"""Focused tests for renewal stopping and event-indexed common randomness."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import cloudpickle
import numpy as np
import torch
from RL.PPO import ZO_eval_renew as renewal


def environment(arrivals=None, service=None):
    return SimpleNamespace(
        seed=142, batch=1, q=2, s=1, state=None,
        network=torch.ones(1, 1, 2), mu=torch.ones(1, 1, 2), h=torch.ones(2),
        queue_event_options=torch.cat((torch.eye(2), -torch.eye(2))),
        draw_inter_arrivals_core=arrivals or (
            lambda env, time: torch.tensor(env.state.exponential(10., (1, 2)))),
        draw_service_core=service or (
            lambda env, time: torch.tensor(env.state.exponential(1., (1, 2)))))


class Policy:
    def set_training_mode(self, mode):
        pass

    def predict(self, queues):
        action = np.zeros((1, 1, 2))
        action[0, 0, int(np.argmax(queues))] = 1
        return action, None


class RenewalTests(unittest.TestCase):
    def test_arrival_calendar_independent_of_service_order(self):
        left, right = [renewal.RenewalEnvironment(environment()) for _ in range(2)]
        traces = [[], []]
        for sim, trace, queue in zip((left, right), traces, (0, 1)):
            for _ in range(80):
                previous = sim.next_arrival.copy()
                sim.step([[1, 0]] if queue == 0 else [[0, 1]])
                for q in np.flatnonzero(previous != sim.next_arrival):
                    trace.append((q, previous[q]))
        for q in range(2):
            a = [t for i, t in traces[0] if i == q]
            b = [t for i, t in traces[1] if i == q]
            n = min(len(a), len(b))
            self.assertGreater(n, 5)
            np.testing.assert_array_equal(a[:n], b[:n])

    def test_server_work_does_not_depend_on_selected_queue(self):
        left, right = [renewal.RenewalEnvironment(environment()) for _ in range(2)]
        for sim, q in ((left, 0), (right, 1)):
            sim.jobs[q] = [None]
            sim.next_arrival[:] = 1e-6
            sim.step([[1, 0]] if q == 0 else [[0, 1]])
        self.assertEqual(left.jobs[0][0], right.jobs[1][0])

    def test_first_return_cap_and_milestones(self):
        env = environment(
            lambda env, time: torch.tensor([[10., float('inf')]]),
            lambda env, time: torch.ones(1, 2))
        renewal._policy = Policy()
        result, snapshots, returned = renewal._trajectory(cloudpickle.dumps(env), 20, [1, 10, 20])
        self.assertTrue(returned)
        self.assertEqual(result[3], 2)
        self.assertEqual(result[2], 11.)
        self.assertEqual(snapshots[1][3], 1)
        self.assertEqual(snapshots[10][3], 2)
        self.assertEqual(snapshots[20][3], 2)
        capped, _, returned = renewal._trajectory(cloudpickle.dumps(env), 1)
        self.assertFalse(returned)
        self.assertEqual(capped[3], 1)
        self.assertEqual(result[0], 1.)

    def test_routing_leaves_external_clocks_unchanged(self):
        env = environment(service=lambda env, time: torch.ones(1, 2))
        env.queue_event_options[2] = torch.tensor([-1., 1.])
        sim = renewal.RenewalEnvironment(env)
        sim.jobs[0] = [None]
        sim.next_arrival[:] = [10., 20.]
        sim.step([[1, 0]])
        np.testing.assert_array_equal(sim.next_arrival, [10., 20.])
        np.testing.assert_array_equal(sim.queues(), [0., 1.])

    def test_multiple_returns_and_cap_at_an_earlier_empty_state(self):
        env = environment(
            lambda env, time: torch.tensor([[10., float('inf')]]),
            lambda env, time: torch.ones(1, 2))
        renewal._policy = Policy()
        result, snapshots, reached = renewal._trajectory(
            cloudpickle.dumps(env), 20, [2, 4, 6, 20], empty_returns=3)
        self.assertTrue(reached)
        self.assertEqual(result[2:5], (31., 6, 3))
        np.testing.assert_allclose(result[5], [1 / 11, 1 / 10, 1 / 10])
        self.assertEqual(result[0], 3.)
        self.assertEqual(snapshots[2][4], 1)
        self.assertEqual(snapshots[4][4], 2)
        self.assertEqual(snapshots[20][3:5], (6, 3))
        self.assertEqual(len(snapshots[2][5]), 1)
        capped, _, reached = renewal._trajectory(
            cloudpickle.dumps(env), 4, empty_returns=3)
        self.assertFalse(reached)
        self.assertEqual(capped[2:5], (21., 4, 2))
        partial, _, _ = renewal._trajectory(cloudpickle.dumps(env), 5, empty_returns=3)
        self.assertEqual(partial[5], capped[5])  # Unfinished third cycle excluded.

    def test_cycle_statistics_pool_cycles_equally(self):
        results = [(2., np.array([2., 4.]), 2., 2, 1, (1.,)),
                   (12., np.array([12., 16.]), 4., 6, 2, (3., 8.))]
        metrics = renewal.summarize(results)
        self.assertEqual(metrics['completed_cycle_count'], 3)
        self.assertEqual(metrics['cycle_holding_cost_mean'], 4.)
        self.assertAlmostEqual(metrics['cycle_holding_cost_std'], np.std([1., 3., 8.], ddof=1))
        empty = renewal.summarize([(0., np.zeros(2), 1., 1, 0, ())])
        self.assertEqual(empty['completed_cycle_count'], 0)
        self.assertIsNone(empty['cycle_holding_cost_mean'])
        self.assertIsNone(empty['cycle_holding_cost_std'])

    def test_metrics_length_sample_std(self):
        metrics = renewal.summarize([(2., np.array([2., 4.]), 2., 2),
                                     (12., np.array([12., 16.]), 4., 6)])
        self.assertEqual(metrics['simulation_steps_mean'], 4.)
        self.assertAlmostEqual(metrics['simulation_steps_std'], np.std([2., 6.], ddof=1))
        self.assertEqual(metrics['total_queue_mean'], 5.)

    def test_spawn_worker(self):
        env = environment(
            lambda env, time: torch.tensor([[10., float('inf')]]),
            lambda env, time: torch.ones(1, 2))
        metrics, _, lengths, returned = renewal.evaluate(Policy(), [env], 10, None, empty_returns=3)
        self.assertEqual(lengths, [6])
        self.assertEqual(returned, [True])
        self.assertEqual(metrics['simulation_steps_std'], 0.)
        self.assertEqual(metrics['completed_cycle_count'], 3)
        self.assertAlmostEqual(metrics['cycle_holding_cost_mean'], np.mean([1 / 11, .1, .1]))
        self.assertAlmostEqual(metrics['cycle_holding_cost_std'], np.std([1 / 11, .1, .1], ddof=1))


if __name__ == '__main__':
    unittest.main()
