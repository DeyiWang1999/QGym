"""Tests for paired event/renewal checkpoint collection."""
import copy
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import tempfile
import unittest

import cloudpickle
import numpy as np
import torch
from RL.PPO import ZO_eval_renew_compare as comparison
from RL.PPO.test_ZO_eval_renew import environment, Policy


class ComparisonTests(unittest.TestCase):
    def make_env(self):
        return environment(
            lambda env, time: torch.tensor([[10., float('inf')]]),
            lambda env, time: torch.ones(1, 2))

    def test_both_targets_in_either_order(self):
        for base, final_steps, final_returns in [(1, 60, 30), (20, 60, 30), (40, 120, 60)]:
            with self.subTest(base=base):
                sim = comparison.RenewalEnvironment(self.make_env())
                result = comparison.collect_trajectory(sim, Policy(), base, 3)
                self.assertEqual(result['final_events'], final_steps)
                self.assertEqual(result['final_returns'], final_returns)
                self.assertEqual([r['events'] for r in result['event_checkpoints']],
                                 [base, 2 * base, 3 * base])
                self.assertEqual([r['events'] for r in result['return_checkpoints']], [20, 40, 60])
                self.assertEqual([r['s'] for r in result['return_checkpoints']], [1, 2, 3])
                self.assertEqual([r['returns'] for r in result['return_checkpoints']], [10, 20, 30])
                np.testing.assert_allclose(
                    [r['average_cost'] for r in result['return_checkpoints']],
                    [10 / 101, 20 / 201, 30 / 301])
                self.assertEqual(len(result['return_checkpoints']), 3)

    def test_summary_ci_and_plot(self):
        result = comparison.collect_trajectory(
            comparison.RenewalEnvironment(self.make_env()), Policy(), 2, 1)
        second = copy.deepcopy(result)
        for key in ('event_checkpoints', 'return_checkpoints'):
            result[key][0]['average_cost'] = 2.
            second[key][0]['average_cost'] = 6.
        second['return_checkpoints'][0]['events'] = 8
        summary = comparison.summarize([result, second], 1)
        self.assertEqual(summary['return_checkpoints'][0]['mean_events'], 14.)
        self.assertEqual(summary['event_checkpoints'][0]['mean_events'], 2.)
        self.assertEqual(summary['return_checkpoints'][0]['mean_average_cost'], 4.)
        self.assertAlmostEqual(summary['return_checkpoints'][0]['ci95_half_width'], 3.92)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'plot.png'
            comparison.plot_results(dict(summary=summary, checkpoint='test/policy.pt',
                                         environment_seeds=[142, 143]), path)
            self.assertGreater(path.stat().st_size, 1000)
        second['return_checkpoints'] = []
        with self.assertRaises(ValueError):
            comparison.summarize([result, second], 1)

    def test_spawn_worker(self):
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context('spawn'),
                                 initializer=comparison._init_worker,
                                 initargs=(cloudpickle.dumps(Policy()),
                                           cloudpickle.dumps(self.make_env()), 1, 3)) as pool:
            first, second = list(pool.map(comparison._run_seed, [142, 142]))
        self.assertEqual(first, second)
        self.assertEqual(first['final_events'], 60)
        self.assertEqual(first['seed'], 142)


if __name__ == '__main__':
    unittest.main()
