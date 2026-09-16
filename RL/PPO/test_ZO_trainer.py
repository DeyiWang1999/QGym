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
                               parameter_partitions, perturbation_ratio,
                               pack_state, unpack_state, evaluate_packed_trajectory,
                               evaluate_pretrain_trajectory)


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
                        lam_params={'val': [0.5, 0.5]}, queue_event_options=None, test_T=5)

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
        self.assertAlmostEqual(perturbation_ratio(t, 0), t['initial_perturbation_ratio'])
        self.assertAlmostEqual(perturbation_ratio(t, 1), t['ending_perturbation_ratio'])

    def test_work_conservation(self):
        policy = ZOPolicy([[1, 1]], scale=1)
        probs = policy.probabilities(torch.tensor([[0., 10.], [10., 0.]]))
        torch.testing.assert_close(probs, torch.tensor([[[0., 1.]], [[1., 0.]]]))
        self.assertFalse(any('value' in name for name, _ in policy.named_parameters()))

    def test_behavior_cloning_can_be_disabled(self):
        self.config['behavior_cloning'] = {'enabled': False}
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            initial = copy.deepcopy(trainer.policy.state_dict())
            with patch('torch.optim.Adam', side_effect=AssertionError('Cloning must not run')):
                trainer.train()
            saved = torch.load(Path(temp)/'run/initial_policy.pt', weights_only=True)
            for key in initial:
                if key in ('mean_queue_length', 'std_queue_length'):
                    continue
                torch.testing.assert_close(saved[key], initial[key], rtol=0, atol=0)
            history = [json.loads(line) for line in (Path(temp)/'run/history.jsonl').read_text().splitlines()]
            self.assertEqual(len(history), 2)
            self.assertEqual(history[0]['para_maxima'], history[1]['para_maxima'])

    def test_partition_remainder_goes_to_last(self):
        policy = torch.nn.Linear(98, 1)  # 98 weights + one bias.
        parts = parameter_partitions(policy, 'partitioned', 5)
        self.assertEqual([p.stop-p.start for p in parts], [19, 19, 19, 19, 23])
        self.assertEqual([i for p in parts for i in range(p.start, p.stop)], list(range(99)))

    def test_evaluation_samples_even_when_randomize_is_false(self):
        self.config['env']['randomize'] = False
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            policy = trainer.policy
            self.assertFalse(policy.randomize)
            queues = torch.tensor([[1., 1.]])
            expected = policy.probabilities(queues).argmax(-1)
            with patch('torch.multinomial', wraps=torch.multinomial) as sample:
                self.assertTrue(torch.equal(policy.act(queues, deterministic=True).argmax(-1), expected))
                sample.assert_not_called()
                evaluate_trajectory((self.env, self.config, policy, trainer.states[0], 42))
                self.assertGreater(sample.call_count, 0)
                sample.reset_mock()
                evaluate_pretrain_trajectory((self.env, self.config, policy, 42))
                self.assertEqual(sample.call_count, self.env['test_T'])

    def test_pretrain_statistics_and_normalization_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            states = [pack_state(state) for state in trainer.states]
            results = [torch.tensor([[float(i), float(i + 4)]]).numpy() for i in range(100)]
            with patch('RL.PPO.ZO_trainer.evaluate_pretrain_trajectory', side_effect=results) as evaluate:
                trainer.pre_train_eval()
            self.assertEqual(evaluate.call_count, 100)
            self.assertEqual([call.args[0][-1] for call in evaluate.call_args_list], list(range(42, 142)))
            expected = torch.tensor([49.5, 53.5])
            torch.testing.assert_close(trainer.policy.mean_queue_length, expected.mean())
            torch.testing.assert_close(trainer.policy.std_queue_length, expected.std())
            queues = torch.tensor([[0., 10.]])
            torch.testing.assert_close(trainer.policy.standardize_queues(queues),
                                       (queues - expected.mean()) / (expected.std() + 1e-8))
            # Normalized negative inputs must not change the raw nonempty mask.
            torch.testing.assert_close(trainer.policy.probabilities(queues), torch.tensor([[[0., 1.]]]))
            restored = ZOPolicy(trainer.policy.network, scale=1)
            restored.load_state_dict(torch.load(Path(temp)/'run/initial_policy.pt', weights_only=True))
            torch.testing.assert_close(restored.probabilities(queues), trainer.policy.probabilities(queues))
            for before, after in zip(states, trainer.states):
                torch.testing.assert_close(torch.from_numpy(before[0]), after.queues)
                torch.testing.assert_close(torch.from_numpy(before[1]), after.time)

    def test_cloning_precedes_pretrain_evaluation(self):
        self.config['behavior_cloning']['enabled'] = True
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
            before = parameters_to_vector(trainer.policy.parameters()).detach().clone()
            actual_evaluate = trainer.pre_train_eval
            def check():
                self.assertFalse(torch.equal(before, parameters_to_vector(trainer.policy.parameters())))
                self.assertEqual(float(trainer.policy.mean_queue_length), 0.)
                actual_evaluate()
            with patch.object(trainer, 'pre_train_eval', side_effect=check) as evaluate:
                trainer.train()
                evaluate.assert_called_once()

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
            active_count = sum(bool(before[part].abs().max()) for part in trainer.partitions)
            calls = []

            def fake(job, *, return_state=True):
                index = len(calls)
                calls.append(copy.deepcopy(job[3]))
                candidate = (index // n) % (active_count+1)
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
            history = [json.loads(line) for line in (Path(temp)/'run/history.jsonl').read_text().splitlines()]
            for iteration, record in enumerate(history):
                checkpoint = torch.load(Path(temp)/f'run/original_{iteration:06d}.pt', weights_only=False)
                baseline = copy.deepcopy(trainer.policy)
                baseline.load_state_dict(checkpoint['policy_state_dict'])
                vector = parameters_to_vector(baseline.parameters()).detach()
                for part, maximum, distance in zip(trainer.partitions, record['para_maxima'], record['perturbation_distances']):
                    self.assertAlmostEqual(maximum, float(before[part].abs().max()))
                    self.assertAlmostEqual(distance, record['perturbation_ratio'] * maximum)
                if iteration == 0:
                    self.assertEqual(record['perturbation_distances'][1], 0.0)  # Zero bias.
            last_base = vector
            self.assertEqual(history[0]['para_maxima'], history[1]['para_maxima'])
            self.assertNotEqual(float(last_base[first].abs().max()), history[1]['para_maxima'][0])
            torch.testing.assert_close((after[first] - last_base[first]).abs(),
                                       torch.full_like(after[first], history[-1]['perturbation_distances'][0] * 0.5))

    def test_initially_zero_partitions_skip_perturbation_and_evaluation(self):
        for mode in ('vanilla', 'partitioned', 'split_layer'):
            for all_zero in (False, True):
                with self.subTest(mode=mode, all_zero=all_zero), tempfile.TemporaryDirectory() as temp:
                    self.config['training'].update(mode=mode, partition_count=3)
                    trainer = ZerothOrderTrainer(self.env, self.config, Path(temp)/'run')
                    with torch.no_grad():
                        for parameter in trainer.policy.parameters():
                            parameter.fill_(0. if all_zero else 1.)
                        vector = parameters_to_vector(trainer.policy.parameters()).detach().clone()
                        vector[trainer.partitions[0]] = 0
                        torch.nn.utils.vector_to_parameters(vector, trainer.policy.parameters())
                    active = [i for i, part in enumerate(trainer.partitions)
                              if bool(vector[part].abs().max())]
                    calls = []

                    def evaluate(job):
                        candidate = parameters_to_vector(job[2].parameters()).detach()
                        for i, part in enumerate(trainer.partitions):
                            if i not in active:
                                torch.testing.assert_close(candidate[part], vector[part], rtol=0, atol=0)
                        calls.append(job)
                        return (10. if job[-1] else 9., job[3] if job[-1] else None)

                    with patch.object(trainer, 'pretrain'), patch.object(trainer, 'pre_train_eval'), \
                            patch('RL.PPO.ZO_trainer.evaluate_packed_trajectory', side_effect=evaluate), \
                            patch('torch.randint', wraps=torch.randint) as perturb:
                        trainer.train()
                    iterations = self.config['training']['total_iterations']
                    self.assertEqual(perturb.call_count, iterations * len(active))
                    self.assertEqual(len(calls), iterations * len(trainer.seeds) * (len(active) + 1))
                    final = parameters_to_vector(trainer.policy.parameters()).detach()
                    for i, part in enumerate(trainer.partitions):
                        if i not in active:
                            torch.testing.assert_close(final[part], vector[part], rtol=0, atol=0)
                        else:
                            self.assertFalse(torch.equal(final[part], vector[part]))
                    history = [json.loads(line) for line in (Path(temp)/'run/history.jsonl').read_text().splitlines()]
                    for record in history:
                        self.assertEqual(record['active_partitions'], active)
                        self.assertEqual(len(record['scores']), len(active) + 1)
                        self.assertEqual(record['accepted'], [True] * len(active))

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

    def test_nearly_simultaneous_arrivals_do_not_make_negative_clock(self):
        env = make_environment(self.env, self.config, 42)
        env.reset(init_queues=torch.zeros(1, 2))
        install_streams(env, 42)
        # Both round UP to the same float32 value, beyond the second arrival.
        clocks = torch.tensor([[1.00000007, 1.00000008]], dtype=torch.float64)
        env.env_state = env.env_state._replace(arrival_times=clocks)
        action = torch.tensor([[[1., 0.]]])
        first = paired_step(env, action)[-1]['event_time']
        self.assertEqual(float(first), float(clocks[0, 0]))
        self.assertEqual(env.st_argmin.index, 0)
        self.assertTrue((env.env_state.arrival_times >= 0).all())
        second = paired_step(env, action)[-1]['event_time']
        self.assertEqual(env.st_argmin.index, 1)
        self.assertEqual(float(second), float(clocks[0, 1] - clocks[0, 0]))
        self.assertGreaterEqual(float(second), 0)

    def test_service_work_update_keeps_double_precision(self):
        self.env['mu'] = [[1.3, 1.3]]
        env = make_environment(self.env, self.config, 42)
        env.reset(init_queues=torch.tensor([[1., 0.]]))
        install_streams(env, 42)
        work = torch.tensor([[1.30000006, 2.]], dtype=torch.float64)
        env.env_state = env.env_state._replace(
            service_times=[[work.clone()], []],
            arrival_times=torch.tensor([[10., 1.00000007]], dtype=torch.float64))
        action = torch.tensor([[[1., 0.]]])
        dt = paired_step(env, action)[-1]['event_time']
        self.assertEqual(env.st_argmin.index, 1)
        residual = env.env_state.service_times[0][0][0, 0]
        expected = work[0, 0] - dt * env.mu[0, 0, 0]
        self.assertGreater(float(residual), 0)
        self.assertEqual(float(residual), float(expected))
        self.assertGreater(float(paired_step(env, action)[-1]['event_time']), 0)

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
