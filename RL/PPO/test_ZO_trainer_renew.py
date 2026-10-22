"""Run: python -m unittest RL.PPO.test_ZO_trainer_renew"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch.nn.utils import parameters_to_vector

from RL.PPO import ZO_trainer_renew as renewal


class RenewalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = yaml.safe_load(
            (Path(__file__).resolve().parents[1] / 'policy_configs/ZO.yaml').read_text())
        self.config['model']['scale'] = 1
        self.config['behavior_cloning']['enabled'] = False
        self.config['pretrain_evaluation'] = {'enabled': False}
        self.config['training'].update(
            mode='partitioned', partition_count=2, total_iterations=1,
            evaluation_trajectories=2, evaluation_seeds=[142, 143],
            evaluation_length=2, max_cpus=1,
            policy_diagnostics={'enabled': False})
        self.env = dict(name='test', network=[[1, 1]], mu=[[1, 1]], h=[1, 1],
                        num_pool=1, init_queues=[5, 5], lam_type='constant',
                        lam_params={'val': [.1, 0.]}, queue_event_options=None, test_T=5)

    def deterministic_rollout(self, cap, samples=None):
        config = copy.deepcopy(self.config)
        config['training']['evaluation_length'] = cap
        env = renewal.make_environment(self.env, config, 142)
        env.draw_inter_arrivals_core = lambda instance, time: torch.tensor([[10., float('inf')]])
        env.draw_service_core = lambda instance, time: torch.ones((1, 2))
        policy = renewal.ZOPolicy([[1, 1]], scale=1)
        # The supplied state is deliberately nonempty; renewal evaluation resets it.
        env.reset(init_queues=torch.tensor([[5., 5.]]))
        state = copy.deepcopy(env.env_state)
        with patch.object(renewal, 'make_environment', return_value=env):
            return renewal.evaluate_trajectory(
                (self.env, config, policy, state, 142), sample_count=samples)

    def test_completed_cycle_uses_cost_over_duration(self):
        score, state, samples = self.deterministic_rollout(2, samples=7)
        self.assertAlmostEqual(score, 1 / 11)
        self.assertEqual(float(state.time), 11.)
        self.assertEqual(int(state.queues.sum()), 0)
        self.assertEqual(samples.shape, (7, 2))
        self.assertTrue(np.all(samples.sum(axis=1) <= 1))

    def test_event_cap_discards_incomplete_cycle(self):
        score, state = self.deterministic_rollout(1)
        self.assertIsNone(score)
        self.assertEqual(float(state.time), 10.)
        self.assertEqual(int(state.queues.sum()), 1)

    def run_mocked_training(self, scores):
        with tempfile.TemporaryDirectory() as temp:
            trainer = renewal.ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            before = parameters_to_vector(trainer.policy.parameters()).detach().clone()
            initial_states = [renewal.pack_state(state) for state in trainer.states]
            # Return a nonempty terminal state to detect accidental state carryover.
            terminal = copy.deepcopy(trainer.states[0])
            terminal.queues.fill_(9)
            packed = renewal.pack_state(terminal)
            results = [(score, packed) for score in scores]
            with patch.object(renewal, 'evaluate_packed_trajectory', side_effect=results):
                trainer.train()
            after = parameters_to_vector(trainer.policy.parameters()).detach().clone()
            for old, state in zip(initial_states, trainer.states):
                np.testing.assert_array_equal(old[0], state.queues.numpy())
            record = json.loads((trainer.output_dir / 'history.jsonl').read_text())
            return before, after, trainer.partitions, record

    def test_failed_perturbation_skips_only_its_partition(self):
        before, after, parts, record = self.run_mocked_training([10., None, None, None, 5., 7.])
        self.assertEqual(record['scores'], [10., None, 6.])
        self.assertEqual(record['valid_trajectory_counts'], [1, 0, 2])
        self.assertEqual(record['accepted'], [False, True])
        self.assertFalse(record['skipped_cycle'])
        torch.testing.assert_close(before[parts[0]], after[parts[0]], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[parts[1]], after[parts[1]]))

    def test_failed_baseline_skips_all_updates(self):
        before, after, _, record = self.run_mocked_training([None, None, 1., 2., 3., 4.])
        self.assertEqual(record['accepted'], [False, False])
        self.assertTrue(record['skipped_cycle'])
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_spawn_training_and_diagnostics(self):
        self.config['training'].update(
            max_cpus=2, evaluation_length=1,
            policy_diagnostics={'enabled': True, 'num_samples': 7})
        with tempfile.TemporaryDirectory() as temp:
            trainer = renewal.ZerothOrderTrainer(self.env, self.config, Path(temp) / 'run')
            trainer.train()
            record = json.loads((trainer.output_dir / 'history.jsonl').read_text())
            self.assertEqual(record['valid_trajectory_counts'], [0, 0, 0])
            self.assertTrue(record['skipped_cycle'])
            self.assertEqual(record['policy_diagnostics']['sampled_states'], 7)
            self.assertTrue((trainer.output_dir / 'final_policy.pt').is_file())


if __name__ == '__main__':
    unittest.main()
