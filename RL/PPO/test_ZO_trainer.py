"""Run: python -m unittest RL.PPO.test_ZO_trainer"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml
from torch.nn.utils import parameters_to_vector

from RL.PPO.ZO_trainer import (ZOPolicy, ZerothOrderTrainer, evaluate_trajectory,
                               install_streams, make_environment, paired_step,
                               parameter_partitions, perturbation_distance,
                               pack_state, unpack_state, evaluate_packed_trajectory)


class ZOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        with (Path(__file__).resolve().parents[1] / 'policy_configs/ZO.yaml').open() as stream:
            self.config = yaml.safe_load(stream)
        self.config['model']['scale'] = 1
        self.config['behavior_cloning'].update(num_samples=4, batch_size=2)
        self.config['training'].update(total_iterations=2, evaluation_trajectories=2,
                                       evaluation_length=5, max_cpus=1)
        self.env = dict(name='test', network=[[1, 1]], mu=[[2, 2]], h=[1, 1],
                        num_pool=1, init_queues=[1, 1], lam_type='constant',
                        lam_params={'val': [0.5, 0.5]}, queue_event_options=None)

    def test_partitions_and_scheduler(self):
        policy = ZOPolicy([[1, 1]], scale=1)
        total = sum(p.numel() for p in policy.parameters())
        self.assertEqual(parameter_partitions(policy, 'vanilla', 1), [slice(0, total)])
        parts = parameter_partitions(policy, 'split_layer', 1)
        self.assertEqual([p.stop-p.start for p in parts], [p.numel() for p in policy.parameters()])
        self.assertEqual(len(parameter_partitions(policy, 'partitioned', total)), total)
        with self.assertRaises(ValueError):
            parameter_partitions(policy, 'partitioned', total+1)
        t = self.config['training']
        self.assertAlmostEqual(perturbation_distance(t, 0), t['initial_perturbation'])
        self.assertAlmostEqual(perturbation_distance(t, 1), t['ending_perturbation'])

    def test_work_conservation(self):
        policy = ZOPolicy([[1, 1]], scale=1)
        probs = policy.probabilities(torch.tensor([[0., 10.], [10., 0.]]))
        torch.testing.assert_close(probs, torch.tensor([[[0., 1.]], [[1., 0.]]]))
        self.assertFalse(any('value' in name for name, _ in policy.named_parameters()))

    def test_partition_remainder_goes_to_last(self):
        policy = torch.nn.Linear(98, 1)  # 98 weights + one bias.
        parts = parameter_partitions(policy, 'partitioned', 5)
        self.assertEqual([p.stop-p.start for p in parts], [19, 19, 19, 19, 23])
        self.assertEqual([i for p in parts for i in range(p.start, p.stop)], list(range(99)))

    def test_evaluation_forces_deterministic_actions(self):
        self.config['env']['randomize'] = True
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            policy = trainer.policy
            self.assertTrue(policy.randomize)
            queues = torch.tensor([[1., 1.]])
            expected = policy.probabilities(queues).argmax(-1)
            with patch('torch.multinomial', side_effect=AssertionError('Evaluation sampled an action')):
                self.assertTrue(torch.equal(policy.act(queues, deterministic=True).argmax(-1), expected))
                evaluate_trajectory((self.env, self.config, policy, trainer.states[0], 42))

    def test_arrival_coupling_despite_different_service(self):
        self.env['queue_event_options'] = [[1, 0], [0, 1], [-1, 1], [0, -1]]
        traces = []
        for action in (torch.tensor([[[1., 0.]]]), torch.tensor([[[0., 1.]]])):
            env = make_environment(self.env, self.config, 42)
            env.reset(init_queues=torch.tensor([[2., 2.]]))
            install_streams(env, 123)
            trace = []
            with torch.no_grad():
                while len(trace) < 8:
                    paired_step(env, action)
                    if env.st_argmin.index < env.q:
                        trace.append((env.st_argmin.index, float(env.env_state.time)))
            traces.append(trace)
        self.assertEqual([e[0] for e in traces[0]], [e[0] for e in traces[1]])
        torch.testing.assert_close(torch.tensor([e[1] for e in traces[0]]),
                                   torch.tensor([e[1] for e in traces[1]]))

    def test_reproducibility_and_pool(self):
        self.env['num_pool'] = 2
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            self.assertEqual(trainer.policy.s, 2)
            job = (self.env, self.config, trainer.policy, trainer.states[0], 42)
            a, b = evaluate_trajectory(job), evaluate_trajectory(job)
            self.assertEqual(a[0], b[0])
            torch.testing.assert_close(a[1].queues, b[1].queues)
            torch.testing.assert_close(a[1].arrival_times, b[1].arrival_times)

    def test_acceptance_and_baseline_state_carryover(self):
        self.config['training'].update(mode='split_layer', update_ratio=0.5)
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            before = parameters_to_vector(trainer.policy.parameters()).detach().clone()
            initial_time = trainer.states[0].time.clone()
            n = len(trainer.seeds)
            calls = []

            def fake(job, *, return_state=True):
                index = len(calls)
                calls.append(copy.deepcopy(job[3]))
                candidate = (index // n) % (len(trainer.partitions)+1)
                state = copy.deepcopy(job[3])
                return (10. if candidate == 0 else (9. if candidate == 1 else 11.),
                        state._replace(time=state.time + (1 if candidate == 0 else 100))
                        if return_state else None)

            with patch.object(trainer, 'pretrain'), patch('RL.PPO.ZO_trainer.evaluate_trajectory', side_effect=fake):
                trainer.train()
            after = parameters_to_vector(trainer.policy.parameters()).detach()
            first = trainer.partitions[0]
            self.assertTrue(torch.all((after[first]-before[first]).abs() > 0))
            torch.testing.assert_close(after[first.stop:], before[first.stop:])
            torch.testing.assert_close(trainer.states[0].time, initial_time+2)
            self.assertTrue((Path(temp)/'run/original_000001.pt').exists())

    def test_all_modes_and_spawn(self):
        for mode in ('vanilla', 'partitioned', 'split_layer'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                self.config['training'].update(mode=mode, partition_count=3,
                                               max_cpus=2 if mode == 'vanilla' else 1)
                trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
                trainer.train()
                history = [json.loads(line) for line in (Path(temp)/'run/history.jsonl').read_text().splitlines()]
                self.assertEqual(len(history), 2)
                self.assertTrue((Path(temp)/'run/final_policy.pt').exists())

    def test_packed_state_preserves_exact_continuation(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            env = make_environment(self.env, self.config, 42)
            env.reset(init_queues=torch.tensor([[0., 100.]]))
            state = env.env_state
            packed = pack_state(state)
            restored = unpack_state(packed)
            for name in ('queues', 'time', 'arrival_times'):
                torch.testing.assert_close(getattr(restored, name), getattr(state, name),
                                           rtol=0, atol=0)
            for original_queue, restored_queue in zip(state.service_times, restored.service_times):
                self.assertEqual(len(original_queue), len(restored_queue))
                for original, restored_job in zip(original_queue, restored_queue):
                    torch.testing.assert_close(original, restored_job, rtol=0, atol=0)
            job = (self.env, self.config, trainer.policy, state, 42)
            expected_score, expected = evaluate_trajectory(job)
            packed_job = (self.env, self.config, trainer.policy, packed, 42)
            score, ending = evaluate_packed_trajectory((*packed_job, True))
            actual = unpack_state(ending)
            self.assertEqual(score, expected_score)
            for name in ('queues', 'time', 'arrival_times'):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                           rtol=0, atol=0)
            next_job = (self.env, self.config, trainer.policy)
            self.assertEqual(evaluate_trajectory((*next_job, actual, 43))[0],
                             evaluate_trajectory((*next_job, expected, 43))[0])
            self.assertEqual(evaluate_packed_trajectory((*packed_job, False)),
                             (expected_score, None))

    def test_repository_reentrant_environment(self):
        path = Path(__file__).resolve().parents[2] / 'configs/env/reentrant_2.yaml'
        with path.open() as stream:
            env_config = yaml.safe_load(stream)
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(env_config, self.config, Path(temp)/'run')
            score, state = evaluate_trajectory((env_config, self.config, trainer.policy,
                                                trainer.states[0], 42))
            self.assertGreaterEqual(score, 0)
            self.assertTrue(torch.isfinite(state.queues).all())

    def test_dummy_arrival_is_not_counted(self):
        self.env['queue_event_options'] = [[1, 0], [0, 0], [-1, 0], [0, -1]]
        self.config['training']['evaluation_length'] = 1
        env = make_environment(self.env, self.config, 42)
        env.reset(init_queues=torch.zeros(1, 2))
        initial = env.env_state._replace(arrival_times=torch.tensor([[1., 0.25]]))
        policy = ZOPolicy(env.network[0], scale=1)
        score, state = evaluate_trajectory((self.env, self.config, policy, initial, 42))
        # The dummy event at t=.25 must not end the one-arrival evaluation.
        self.assertEqual(float(state.time), 1.0)
        torch.testing.assert_close(state.queues, torch.tensor([[1., 0.]]))
        self.assertTrue(torch.isinf(state.arrival_times[0, 1]))
        self.assertEqual(score, 0.0)

    def test_pretrain_saves_reusable_initial_policy(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            trainer.pretrain()
            saved = torch.load(Path(temp)/'run/initial_policy.pt', weights_only=True)
            restored = ZOPolicy(trainer.policy.network, scale=self.config['model']['scale'])
            restored.load_state_dict(saved)
            for key, value in trainer.policy.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[key], value)
            self.assertFalse((Path(temp)/'run/original_000000.pt').exists())

    def test_default_output_directory_and_override(self):
        from RL.PPO import ZO_train
        for override in (None, 'custom-output'):
            args = ['ZO_train.py', 'ZO', 'reentrant_2']
            if override:
                args.extend(['--output-dir', override])
            with patch('sys.argv', args), patch.object(ZO_train, 'ZerothOrderTrainer') as trainer:
                ZO_train.main()
                expected = Path(override) if override else ZO_train.ROOT / 'RL/PPO/reentrant_2'
                self.assertEqual(trainer.call_args.args[2], expected)
                trainer.return_value.train.assert_called_once()


if __name__ == '__main__':
    unittest.main()
