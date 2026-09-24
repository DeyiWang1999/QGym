"""Run: python -m unittest RL.PPO.test_ZO_trainer_hospital"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from main.env import allocator
from RL.PPO.ZO_trainer_hospital import (
    ZOPolicy, ZerothOrderTrainer, evaluate_trajectory,
    evaluate_pretrain_trajectory, make_environment, pack_state,
    unpack_state, evaluate_packed_trajectory,
)


class HospitalZOTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        root = Path(__file__).resolve().parents[2]
        self.config = yaml.safe_load((root / 'RL/policy_configs/ZO.yaml').read_text())
        self.config['model']['scale'] = 1
        self.config['behavior_cloning'].update(num_samples=4, batch_size=2)
        self.config['pretrain_evaluation']['enabled'] = False
        self.config['training'].update(total_iterations=1, evaluation_trajectories=2,
                                       evaluation_length=5, max_cpus=1)
        self.env = dict(name='test', network=[[1, 1], [1, 0]],
                        mu=[[10, 9], [9, 0]], h=[1, 1], server_pool_size=[2, 2],
                        num_pool=7, init_queues=[2, 2], lam_type='constant',
                        lam_params={'val': [0.5, 0.5]}, queue_event_options=None,
                        test_T=5)

    def test_integer_apportionment_masks_and_capacity(self):
        policy = ZOPolicy([[1, 1, 0], [0, 1, 1], [1, 1, 1]], [5, 3, 0], scale=1)
        with torch.no_grad():
            for parameter in policy.parameters():
                parameter.zero_()
        queues = torch.tensor([[20., 20., 20.], [0., 0., 1.], [0., 0., 0.]])
        action = policy.act(queues, deterministic=False)
        torch.testing.assert_close(action, torch.tensor([
            [[3, 2, 0], [0, 2, 1], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 3], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ]))
        torch.testing.assert_close(action, policy.act(queues, deterministic=True))

    def test_invalid_pool_sizes(self):
        for pools in ([1], [-1, 2], [1.5, 2], [float('nan'), 2]):
            with self.subTest(pools=pools), self.assertRaisesRegex(ValueError, 'server_pool_size'):
                ZOPolicy([[1, 1], [1, 0]], pools, scale=1)

    def test_integer_actions_serve_distinct_patients_at_per_bed_rate(self):
        services = [[torch.tensor([[10.]], dtype=torch.float64) for _ in range(4)]]
        rates = torch.tensor([[[2.]]], dtype=torch.float64)
        for beds in (1, 3):
            work, _, count = allocator(torch.tensor([[[float(beds)]]]), rates, services)
            self.assertEqual(count, [beds])
            self.assertEqual([float(x) for x in work[0]], [2.] * beds)

    def test_global_teacher_coordinates_pools_and_respects_demand(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            queues = torch.tensor([[2., 2.], [0., 0.], [1., 0.]])
            action = trainer.teacher_allocations(queues)
            # Local argmax would send both pools to class 0. The global optimum
            # reserves that class for its dedicated pool and serves both classes.
            torch.testing.assert_close(action[0], torch.tensor([[0., 2.], [2., 0.]]))
            self.assertTrue((action.sum(1) <= queues).all())
            self.assertTrue((action.sum(2) <= torch.tensor([2, 2])).all())
            self.assertTrue((action == action.round()).all())
            self.assertEqual(float(action[:, 1, 1].sum()), 0.)
            self.assertEqual(float(action[1].sum()), 0.)

    def test_teacher_uses_holding_costs(self):
        self.env.update(network=[[1, 1]], mu=[[1, 1]], h=[1, 5], server_pool_size=[2])
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            torch.testing.assert_close(trainer.teacher_allocations(torch.tensor([[4., 2.]])),
                                       torch.tensor([[[0., 2.]]]))

    def test_cloning_changes_policy_and_checkpoints_restore_capacities(self):
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            before = trainer.policy.action_net.weight.detach().clone()
            trainer.pretrain()
            self.assertFalse(torch.equal(before, trainer.policy.action_net.weight))
            restored = ZOPolicy(self.env['network'], [0, 0], scale=1)
            restored.load_state_dict(torch.load(trainer.output_dir / 'initial_policy.pt', weights_only=True))
            queues = torch.tensor([[10., 10.]])
            torch.testing.assert_close(restored.act(queues), trainer.policy.act(queues))
            torch.testing.assert_close(restored.server_pool_size, torch.tensor([2, 2]))

    def test_trajectory_and_pretraining_use_pool_rows(self):
        env = make_environment(self.env, self.config, 42)
        self.assertEqual(env.s, 2)  # num_pool must not replicate hospital pools.
        self.assertEqual(env.action_space.high.tolist(), [[2., 2.], [2., 2.]])
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            job = (self.env, self.config, trainer.policy, trainer.states[0], 42)
            score, state = evaluate_trajectory(job)
            packed_score, packed = evaluate_packed_trajectory(
                (*job[:3], pack_state(job[3]), 42, True))
            self.assertEqual(score, packed_score)
            torch.testing.assert_close(state.queues, unpack_state(packed).queues)
            means = evaluate_pretrain_trajectory((*job[:3], 42))
            self.assertTrue(torch.isfinite(torch.as_tensor(means)).all())

    def test_training_modes_and_spawn(self):
        for mode in ('vanilla', 'partitioned', 'split_layer'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp:
                config = copy.deepcopy(self.config)
                config['training'].update(mode=mode, max_cpus=2 if mode == 'vanilla' else 1)
                config['training']['policy_diagnostics'] = dict(enabled=True, num_samples=4)
                trainer = ZerothOrderTrainer(self.env, config, Path(temp) / 'run')
                trainer.train()
                history = json.loads((trainer.output_dir / 'history.jsonl').read_text())
                self.assertTrue(all(torch.isfinite(torch.tensor(history['scores']))))
                self.assertEqual(history['policy_diagnostics']['servers'], 2)
                self.assertTrue((trainer.output_dir / 'final_policy.pt').is_file())

    def test_training_resets_all_trajectories_at_test_horizon(self):
        self.env['test_T'] = 11  # Floor division gives two iterations per reset.
        self.config['training'].update(total_iterations=5, mode='vanilla',
                                       evaluation_seeds=[3012, 3000])
        self.config['behavior_cloning']['enabled'] = False
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            trainer.train()
            checkpoints = [torch.load(trainer.output_dir / f'original_{i:06d}.pt',
                                      weights_only=False) for i in range(5)]
            self.assertEqual([c['evaluation_seeds'] for c in checkpoints],
                             [[3012, 3000], [3012, 3000], [3013, 3014],
                              [3013, 3014], [3015, 3016]])
            for state in checkpoints[0]['initial_states']:
                torch.testing.assert_close(state.queues, torch.tensor([[2., 2.]]))
            for state in checkpoints[1]['initial_states']:
                self.assertGreater(float(state.time), 0.)
            for checkpoint in (checkpoints[2], checkpoints[4]):
                for state, seed in zip(checkpoint['initial_states'], checkpoint['evaluation_seeds']):
                    self.assertEqual(float(state.queues.sum()), 0.)
                    self.assertEqual(float(state.time), 0.)
                    self.assertTrue(all(not jobs for jobs in state.service_times))
                    env = make_environment(self.env, self.config, seed)
                    env.reset()
                    torch.testing.assert_close(state.arrival_times, env.env_state.arrival_times)

    def test_repository_hospital(self):
        root = Path(__file__).resolve().parents[2]
        env_config = yaml.safe_load((root / 'configs/env/hospital.yaml').read_text())
        with tempfile.TemporaryDirectory() as temp:
            trainer = ZerothOrderTrainer(env_config, self.config, Path(temp) / 'run')
            self.assertEqual(trainer.policy.s, 13)
            self.assertEqual(int(trainer.policy.server_pool_size.sum()), 497)
            score, state = evaluate_trajectory(
                (env_config, self.config, trainer.policy, trainer.states[0], 42))
            self.assertGreaterEqual(score, 0.)
            self.assertTrue(torch.isfinite(state.queues).all())

    def test_entry_point_defaults_and_override(self):
        from RL.PPO import ZO_train_hospital
        for override in (None, 'custom-hospital-output'):
            args = ['ZO_train_hospital.py']
            if override:
                args.extend(['--output-dir', override])
            with patch('sys.argv', args), patch.object(ZO_train_hospital, 'ZerothOrderTrainer') as trainer:
                ZO_train_hospital.main()
                self.assertEqual(trainer.call_args.args[0]['name'], 'hospital')
                expected = Path(override) if override else ZO_train_hospital.ROOT / 'RL/PPO/hospital'
                self.assertEqual(trainer.call_args.args[2], expected)
                trainer.return_value.train.assert_called_once()


if __name__ == '__main__':
    unittest.main()
