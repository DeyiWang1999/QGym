import copy
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cloudpickle
import numpy as np
import torch
import yaml
from RL.PPO.ZO_eval import EvaluationPolicy, ROOT, validate_seeds, evaluation_schedule
from RL.PPO.ZO_trainer import evaluate_pretrain_trajectory
from RL.utils import eval as evaluation
from RL.utils.rl_env import load_rl_p_env


class EvaluationTests(unittest.TestCase):
    def test_schedule_follows_network_test_horizon(self):
        for name, base in [('reentrant_9', 300000), ('reentrant_2', 12000)]:
            config = dict(name=name, test_T=base)
            self.assertEqual(evaluation_schedule(config, True),
                             (5 * base, [base * i for i in range(1, 6)]))
            self.assertEqual(evaluation_schedule(config, False), (base, None))
        for invalid in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                evaluation_schedule(dict(test_T=invalid), True)

    def setUp(self):
        self.config = yaml.safe_load((ROOT / 'RL/policy_configs/ZO.yaml').read_text())
        self.env_config = yaml.safe_load((ROOT / 'configs/env/reentrant_9.yaml').read_text())

    def test_seed_separation_includes_normalization_and_explicit_seeds(self):
        checkpoint = dict(config=copy.deepcopy(self.config), evaluation_seeds=[3003])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_seeds(checkpoint, range(3003, 3103))
        checkpoint['evaluation_seeds'] = list(range(42, 52))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            validate_seeds(checkpoint, [141])
        validate_seeds(checkpoint, range(3003, 3103))

    def test_shared_rollout_reproducible_and_matches_zo_pretrain(self):
        torch.set_num_threads(1)
        env = load_rl_p_env(self.env_config, 1., 1, 3003, 'WC', torch.device('cpu'))
        policy = EvaluationPolicy(env.network[0], scale=1)
        evaluation._eval_policy = policy
        packed = cloudpickle.dumps(env)
        first = evaluation._eval_trajectory(packed, 30, 3003)
        torch.manual_seed(12345)
        second = evaluation._eval_trajectory(packed, 30, 3003)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.env_config['test_T'] = 30
        expected = evaluate_pretrain_trajectory((self.env_config, self.config, policy, 3003))
        np.testing.assert_array_equal(first[1] / first[2], expected)

    def test_milestones_match_independent_prefix_rollouts(self):
        torch.set_num_threads(1)
        env = load_rl_p_env(self.env_config, 1., 1, 3003, 'WC', torch.device('cpu'))
        evaluation._eval_policy = EvaluationPolicy(env.network[0], scale=1)
        packed = cloudpickle.dumps(env)
        milestones = [10, 20, 30, 40, 50]
        snapshots = evaluation._eval_trajectory(packed, 50, 3003, milestones)
        self.assertEqual(list(snapshots), milestones)
        for step in milestones:
            expected = evaluation._eval_trajectory(packed, step, 3003)
            for actual, reference in zip(snapshots[step], expected):
                np.testing.assert_array_equal(actual, reference)

    def test_spawn_callback(self):
        torch.set_num_threads(1)
        envs = [load_rl_p_env(copy.deepcopy(self.env_config), 1., 1, seed, 'WC', torch.device('cpu'))
                for seed in (3003, 3004)]
        policy = EvaluationPolicy(envs[0].network[0], scale=1)
        callback = evaluation.parallel_eval(
            SimpleNamespace(policy=policy), envs, 1, 10, 'softmax', 3003,
            torch.zeros(1, 27), 2, 'cpu', 1, False, 'WC', False,
            'reentrant_9', False, seed_actions=True, milestones=[2, 4, 6, 8, 10])
        result = callback.eval()
        self.assertTrue(all(torch.isfinite(value) for value in result))
        self.assertAlmostEqual(float(result[6]), float(result[0]) * 27, places=5)
        self.assertEqual(len(callback.milestone_metrics), 5)
        self.assertEqual(callback.milestone_metrics[-1]['total_queue_mean'], float(result[6]))
        self.assertEqual(callback.milestone_metrics[-1]['total_queue_std'], float(result[7]))


if __name__ == '__main__':
    unittest.main()
