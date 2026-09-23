"""Run: python -m unittest RL.PPO.test_ZO_eval_hospital"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cloudpickle
import numpy as np
import torch
import yaml

from RL.PPO import ZO_eval_hospital as hospital
from RL.PPO.ZO_trainer_hospital import evaluate_pretrain_trajectory
from RL.utils import eval as evaluation


class HospitalEvaluationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = yaml.safe_load(hospital.DEFAULT_POLICY_CONFIG.read_text())
        self.config['model']['scale'] = 1
        self.env_config = yaml.safe_load((hospital.ROOT / 'configs/env/hospital.yaml').read_text())
        self.env_config['test_T'] = 20
        self.env_config['num_pool'] = 3  # Must not expand hospital rows.
        self.policy = hospital.EvaluationPolicy(
            self.env_config['network'], self.env_config['server_pool_size'], scale=1)
        self.policy.update_mean_std(3., 2.)
        self.checkpoint = dict(config=self.config, env_config=self.env_config,
                               evaluation_seeds=[42, 43],
                               policy_state_dict=self.policy.state_dict())

    def test_default_seed_reads_yaml_and_override(self):
        self.assertEqual(hospital.evaluation_seeds(), list(range(42, 142)))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ZO.yaml'
            path.write_text('env:\n  test_seed: 123\n')
            with patch.object(hospital, 'DEFAULT_POLICY_CONFIG', path):
                self.assertEqual(hospital.evaluation_seeds(), list(range(123, 223)))
                self.assertEqual(hospital.evaluation_seeds(500), list(range(500, 600)))
        for seed in (-1, 2**32 - 99, True, 1.5):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                hospital.evaluation_seeds(seed)

    def test_overlap_override_is_preserved(self):
        used = hospital.validate_seeds(self.checkpoint, range(42, 142))
        self.assertIn(42, used)
        self.assertIn(self.config['env']['train_seed'], used)

    def test_policy_restores_integer_allocations_and_normalization(self):
        restored = hospital.load_policy(self.checkpoint)
        queues = torch.full((1, 8), 100.)
        actual, _ = restored.predict(queues)
        expected, _ = self.policy.predict(queues)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual.sum(-1)[0], self.env_config['server_pool_size'])
        self.assertTrue((actual == np.rint(actual)).all())
        self.assertGreater(actual.max(), 1)
        self.assertEqual(float(restored.mean_queue_length), 3.)
        self.assertFalse(restored.training)
        invalid = copy.deepcopy(self.checkpoint)
        invalid['env_config']['server_pool_size'][0] += 1
        with self.assertRaisesRegex(ValueError, 'server_pool_size'):
            hospital.load_policy(invalid)

    def test_shared_rollout_matches_hospital_pretrain_and_milestones(self):
        env = hospital.make_evaluation_environment(self.env_config, self.config, 42)
        self.assertEqual(env.s, 13)
        self.assertEqual(env.mu.dtype, torch.float64)
        packed = cloudpickle.dumps(env)
        with patch.object(evaluation, '_eval_policy', self.policy, create=True):
            first = evaluation._eval_trajectory(packed, 20, 42)
            second = evaluation._eval_trajectory(packed, 20, 999)
            for left, right in zip(first, second):
                np.testing.assert_array_equal(left, right)
            expected = evaluate_pretrain_trajectory(
                (self.env_config, self.config, self.policy, 42))
            np.testing.assert_array_equal(first[1] / first[2], expected)
            milestones = [4, 8, 12, 16, 20]
            snapshots = evaluation._eval_trajectory(packed, 20, 42, milestones)
            for step in milestones:
                prefix = evaluation._eval_trajectory(packed, step, 42)
                for actual, reference in zip(snapshots[step], prefix):
                    np.testing.assert_array_equal(actual, reference)

    def test_spawn_callback(self):
        envs = [hospital.make_evaluation_environment(self.env_config, self.config, seed)
                for seed in (42, 43)]
        callback = hospital.parallel_eval(
            SimpleNamespace(policy=self.policy), envs, 1, 20, 'softmax', 42,
            torch.zeros(1, 8), 2, 'cpu', 1, False, 'WC', False, 'hospital', False,
            seed_actions=True, milestones=[4, 8, 12, 16, 20])
        result = callback.eval()
        self.assertTrue(all(torch.isfinite(value).all() for value in result))
        self.assertEqual(len(callback.milestone_metrics), 5)
        self.assertEqual(callback.milestone_metrics[-1]['total_queue_mean'], float(result[6]))

    def test_cli_final_checkpoint_seeds_and_output(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            torch.save(self.checkpoint, directory / 'original_000000.pt')
            torch.save(self.policy.state_dict(), directory / 'final_policy.pt')
            selected = hospital.load_selected_checkpoint(directory / 'final_policy.pt')
            hospital.load_policy(selected)
            for override in (None, 700):
                args = ['ZO_eval_hospital.py', temp, '--indices', '0', '--include-final',
                        '--ranking-check']
                if override is not None:
                    args.extend(['--test-seed', str(override)])
                with patch('sys.argv', args), patch.object(hospital, 'parallel_eval') as callback:
                    callback.return_value.eval.return_value = [torch.tensor(1.)] * 8
                    callback.return_value.milestone_metrics = [dict(events=100, total_queue_mean=1.)]
                    hospital.main()
                    self.assertEqual(callback.call_count, 2)
                    kwargs = callback.call_args.kwargs
                    first_seed = 42 if override is None else override
                    self.assertEqual(kwargs['test_seed'], first_seed)
                    self.assertEqual([env.seed for env in kwargs['eval_env']],
                                     list(range(first_seed, first_seed + 100)))
                    self.assertEqual(kwargs['milestones'], [20, 40, 60, 80, 100])
                record = json.loads((directory / 'ppo_ranking_evaluation.json').read_text())
                self.assertEqual(record['environment_seeds'], list(range(first_seed, first_seed + 100)))
                self.assertEqual(record['action_mode'], 'integer_pool_allocation')
                self.assertEqual([r['checkpoint'] for r in record['results']],
                                 ['original_000000.pt', 'final_policy.pt'])
                self.assertFalse(record['results'][0]['seed_overlap_enforced'])

    def test_check_only_does_not_start_rollouts(self):
        with tempfile.TemporaryDirectory() as temp:
            torch.save(self.checkpoint, Path(temp) / 'original_000000.pt')
            with patch('sys.argv', ['ZO_eval_hospital.py', temp, '--indices', '0', '--check-only']), \
                    patch.object(hospital, 'parallel_eval') as callback:
                hospital.main()
                callback.assert_not_called()


if __name__ == '__main__':
    unittest.main()
