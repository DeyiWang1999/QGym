"""Evaluate ZO actors until S returns to empty, capped at a configurable multiple of test_T.

Randomness is indexed by external queue arrival and by server's newly started
job (first service assignment). Service draws are unit-rate WORK, as in QGym:
duration = work / mu[server, queue]. Equal work does not imply equal duration
when policies choose different service rates. Preempted jobs retain their work,
including when moved to another server; resuming them consumes no new draw.
External arrival calendars use absolute float64 times and never change on routing.
Metrics retain ZO_eval's mean/std of per-trajectory time averages. Event lengths
are reported separately from elapsed simulated time. Capped paths are identified.
"""
import argparse
import copy
import gc
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cloudpickle
import numpy as np
import torch
from RL.PPO.ZO_eval import (
    EvaluationPolicy, evaluation_schedule as original_evaluation_schedule,
    load_selected_checkpoint, validate_seeds,
)
from RL.utils.rl_env import load_rl_p_env


def evaluation_schedule(env_config, ranking_check, horizon_multiplier=10):
    """Scale the cap; optionally report each original-horizon milestone."""
    if type(horizon_multiplier) is not int or horizon_multiplier < 1:
        raise ValueError("horizon_multiplier must be a positive integer")
    base, _ = original_evaluation_schedule(env_config, False)
    milestones = [base * i for i in range(1, horizon_multiplier + 1)] if ranking_check else None
    return horizon_multiplier * base, milestones


class RenewalEnvironment:
    """CPU, batch-one event simulator with QGym's FIFO/fastest-server allocation.

Use the original distribution callbacks, but give each queue/server its own RNG.
Only an actual external arrival advances that queue's arrival RNG. Only first
service of a job advances a server RNG. Drawing unused vector entries within a
callback cannot shift another source's stream.
    """

    def __init__(self, env):
        self.env = env
        self.network = env.network[0].numpy()
        self.mu = env.mu[0].double().numpy()
        self.h = env.h.double().numpy()
        self.delta = env.queue_event_options.double().numpy()
        self.q, self.s = env.q, env.s
        if env.batch != 1 or self.delta.shape != (2 * self.q, self.q):
            raise ValueError('Renewal evaluation requires batch=1 and 2*q event rows')
        for q, row in enumerate(self.delta[:self.q]):
            if not (np.array_equal(row, np.eye(self.q)[q]) or not row.any()):
                raise ValueError('External rows must add one source job or disable arrivals')
        for q, row in enumerate(self.delta[self.q:]):
            if row[q] != -1 or np.any(np.delete(row, q) < 0) or np.any(row != row.astype(int)):
                raise ValueError('Service rows must remove one source job and route integer jobs')
        eligible = self.network > 0
        if not np.isfinite(self.mu[eligible]).all() or np.any(self.mu[eligible] <= 0):
            raise ValueError('Eligible service rates must be finite and positive')
        self.arrival_rng = [np.random.RandomState(
            np.random.SeedSequence([env.seed, 0, q]).generate_state(1)[0])
            for q in range(self.q)]
        self.service_rng = [np.random.RandomState(
            np.random.SeedSequence([env.seed, 1, s]).generate_state(1)[0])
            for s in range(self.s)]
        self.time = 0.
        # Match ZO_eval: reset to empty, regardless of config.init_queues.
        self.jobs = [[] for _ in range(self.q)]
        self.next_arrival = np.array([self.draw_arrival(q) for q in range(self.q)])
        self.steps = 0

    def draw(self, callback, rng, q):
        previous = self.env.state
        self.env.state = rng
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                values = callback(self.env, torch.tensor([[self.time]], dtype=torch.float64))
            return float(values[0, q])
        finally:
            self.env.state = previous

    def draw_arrival(self, q):
        if not self.delta[q].any():
            return np.inf
        value = self.draw(self.env.draw_inter_arrivals_core, self.arrival_rng[q], q)
        if np.isnan(value) or value <= 0:
            raise ValueError('Interarrival times must be positive or infinity')
        return value

    def queues(self):
        return np.array([len(jobs) for jobs in self.jobs], dtype=np.float64)

    def step(self, action):
        action = np.asarray(action).reshape(self.s, self.q) * self.network
        if not np.all((action == 0) | (action == 1)) or np.any(action.sum(1) > 1):
            raise ValueError('Expected at most one selected queue per server (num_pool=1)')
        before = self.queues()
        allocated = []
        for q in range(self.q):
            servers = np.flatnonzero(action[:, q] > 0).tolist()
            servers.sort(key=lambda s: -self.mu[s, q])
            for j, s in enumerate(servers[:len(self.jobs[q])]):
                if self.jobs[q][j] is None:
                    # Service callbacks have identical marginals for all queues.
                    # Always take component zero: selecting component q would
                    # break the coupling when policies choose different queues.
                    work = self.draw(self.env.draw_service_core, self.service_rng[s], 0)
                    if not np.isfinite(work) or work <= 0:
                        raise ValueError('Service work must be finite and positive')
                    self.jobs[q][j] = work
                allocated.append((self.jobs[q][j] / self.mu[s, q], q, j, s))
        completion = min(allocated, default=(np.inf, 0, 0, 0))
        arrival_q = int(self.next_arrival.argmin())
        arrival_dt = max(0., self.next_arrival[arrival_q] - self.time)
        external = arrival_dt <= completion[0]
        dt = arrival_dt if external else completion[0]
        if not np.isfinite(dt):
            raise RuntimeError('No finite event exists before a return to empty')
        for _, q, j, s in allocated:
            self.jobs[q][j] = max(0., self.jobs[q][j] - dt * self.mu[s, q])
        # Assign from the calendar directly at arrivals: no policy-dependent drift.
        self.time = float(self.next_arrival[arrival_q]) if external else self.time + dt
        if external:
            self.jobs[arrival_q].append(None)
            self.next_arrival[arrival_q] = self.time + self.draw_arrival(arrival_q)
        else:
            _, q, j, _ = completion
            self.jobs[q].pop(j)
            for destination, count in enumerate(self.delta[self.q + q]):
                if count > 0:
                    self.jobs[destination].extend([None] * int(count))
        self.steps += 1
        # Preserve existing evaluator's post-event queue integral convention;
        # holding costs use the pre-event queues, as in the original simulator.
        return float(before @ self.h) * dt, self.queues() * dt


def _init_worker(policy_bytes):
    global _policy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _policy = cloudpickle.loads(policy_bytes)
    _policy.set_training_mode(False)


def _trajectory(env_bytes, cap, milestones=None, empty_returns=1):
    if type(empty_returns) is not int or empty_returns < 1:
        raise ValueError('empty_returns must be a positive integer')
    env = cloudpickle.loads(env_bytes)
    torch.manual_seed(env.seed)
    sim = RenewalEnvironment(env)
    cost, integral = 0., np.zeros(sim.q)
    snapshots = {}
    return_count = 0
    with torch.no_grad():
        for step in range(1, cap + 1):
            was_nonempty = any(sim.jobs)
            action, _ = _policy.predict(sim.queues().reshape(1, -1))
            increment, queue_increment = sim.step(action)
            cost += increment
            integral += queue_increment
            # Count transitions back to empty; the initial empty state does not count.
            # Continue the same trajectory and random streams between returns.
            if was_nonempty and not any(sim.jobs):
                return_count += 1
            if milestones and step in milestones:
                snapshots[step] = (cost, integral.copy(), sim.time, step, return_count)
            if return_count >= empty_returns:
                break
    result = (cost, integral.copy(), sim.time, sim.steps, return_count)
    # Each milestone means min(S-th return, milestone cap); stopped paths freeze.
    for milestone in milestones or ():
        snapshots.setdefault(milestone, result)
    return result, snapshots, return_count >= empty_returns


def _std(values):
    return float(np.std(values, ddof=1)) if np.size(values) > 1 else 0.


def summarize(results):
    times = np.array([r[2] for r in results])
    steps = np.array([r[3] for r in results])
    if np.any(times <= 0):
        raise RuntimeError('Cannot compute time averages with zero elapsed time')
    queue_means = np.stack([r[1] for r in results]) / times[:, None]
    per_queue = queue_means.mean(0)
    totals = queue_means.sum(1)
    costs = np.array([r[0] for r in results]) / times
    metrics = dict(queue_mean=float(per_queue.mean()), queue_std_across_queues=_std(per_queue),
                   time_mean=float(times.mean()), time_max=float(times.max()),
                   time_min=float(times.min()), time_std=_std(times),
                   total_queue_mean=float(totals.mean()), total_queue_std=_std(totals),
                   total_holding_cost_mean=float(costs.mean()), total_holding_cost_std=_std(costs),
                   simulation_steps_mean=float(steps.mean()), simulation_steps_std=_std(steps))
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Nonfinite evaluation metrics')
    return metrics


def evaluate(policy, envs, cap, milestones, empty_returns=1):
    workers = min(48, os.cpu_count() or 1, len(envs))
    with ProcessPoolExecutor(max_workers=workers,
                             mp_context=multiprocessing.get_context('spawn'),
                             initializer=_init_worker,
                             initargs=(cloudpickle.dumps(policy),)) as executor:
        futures = [executor.submit(_trajectory, cloudpickle.dumps(env), cap, milestones, empty_returns)
                   for env in envs]
        paths = [future.result() for future in futures]
    metrics = summarize([path[0] for path in paths])
    snapshots = [dict(events=step, **summarize([path[1][step] for path in paths]))
                 for step in milestones or ()]
    return metrics, snapshots, [path[0][3] for path in paths], [path[2] for path in paths]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dir', type=Path, nargs='?', default=Path(__file__).parent / 'reentrant_9')
    # parser.add_argument('--test-seed', type=int, default=42)
    parser.add_argument('--test-seed', type=int, default=142)
    parser.add_argument('--indices', type=int, nargs='+', default=list(range(0, 101, 10)))
    parser.add_argument('--include-final', action='store_true', help='Also evaluate final_policy.pt using original_000000.pt metadata')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--empty-returns', '--S', '-S', type=int, default=1, metavar='S',
                        help='Stop after S returns to all-empty queues (positive integer; default: 1)')
    parser.add_argument('--horizon-multiplier', type=int, default=10, metavar='K',
                        help='Cap each trajectory at K * test_T events (positive integer; default: 10)')
    parser.add_argument('--ranking-check', action='store_true',
                        help="Report stopped trajectories at each test_T milestone up to the configured cap")
    parser.add_argument('--check-only', action='store_true', help='Validate files and seed separation without rollouts')
    args = parser.parse_args()
    if args.empty_returns < 1:
        parser.error('--empty-returns / --S must be a positive integer')
    if args.horizon_multiplier < 1:
        parser.error('--horizon-multiplier must be a positive integer')
    seeds = list(range(args.test_seed, args.test_seed + 400))
    if not 0 <= seeds[0] <= seeds[-1] < 2**32:
        parser.error('Test seeds must be uint32 values')
    paths = [args.checkpoint_dir / f'original_{i:06d}.pt' for i in args.indices]
    if args.include_final:
        paths.append(args.checkpoint_dir / 'final_policy.pt')
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error('Missing checkpoints: ' + ', '.join(missing) +
                     '. A 100-iteration run ends at original_000099.pt; use --indices 0 10 20 30 40 50 60 70 80 90 --include-final for the final updated actor.')
    jobs = []
    for path in paths:
        checkpoint = load_selected_checkpoint(path)
        used = validate_seeds(checkpoint, seeds)
        env_config = checkpoint['env_config']
        evaluation_schedule(env_config, args.ranking_check, args.horizon_multiplier)
        if env_config['num_pool'] != 1:
            raise ValueError(f'{path}: PPO-compatible evaluation currently requires num_pool=1')
        jobs.append(path)
        eval_t, milestones = evaluation_schedule(env_config, args.ranking_check, args.horizon_multiplier)
        print(f'{path.name}: {env_config["name"]}; base test_T={env_config["test_T"]}; '
              f'cap={eval_t} events; target empty returns={args.empty_returns}; milestones={milestones}', flush=True)
        del checkpoint, env_config
        gc.collect()
    print(f'Validated {len(jobs)} policies; {len(seeds)} environments; seeds {seeds[0]}–{seeds[-1]}', flush=True)
    if args.check_only:
        return
    torch.set_num_threads(1)
    output = args.output or args.checkpoint_dir / (
        'ppo_ranking_evaluation_renew.json' if args.ranking_check else 'ppo_evaluation_renew.json')
    records = []
    for path in jobs:
        checkpoint = load_selected_checkpoint(path)
        used = validate_seeds(checkpoint, seeds)
        config, env_config = checkpoint['config'], checkpoint['env_config']
        eval_t, milestones = evaluation_schedule(env_config, args.ranking_check, args.horizon_multiplier)
        state = checkpoint['policy_state_dict']
        policy = EvaluationPolicy(state['network'], config['model']['scale'])
        policy.load_state_dict(state, strict=True)
        envs = [load_rl_p_env(copy.deepcopy(env_config), config['env']['env_temp'], 1, seed, 'WC', torch.device('cpu')) for seed in seeds]
        print(f'Evaluating {path.name}', flush=True)
        metrics, milestone_metrics, lengths, returned = evaluate(
            policy, envs, eval_t, milestones, args.empty_returns)
        print(f'{path.name}: {metrics}', flush=True)
        records.append(dict(checkpoint=path.name, metrics=metrics,
                            environment=env_config['name'], base_test_T=env_config['test_T'],
                            events_per_environment=eval_t,
                            horizon_multiplier=args.horizon_multiplier,
                            target_empty_returns=args.empty_returns,
                            excluded_training_seeds=used,
                            trajectory_steps=lengths, returned_empty=returned,
                            returned_empty_count=sum(returned),
                            capped_count=len(returned) - sum(returned)))
        if args.ranking_check:
            records[-1]['milestones'] = milestone_metrics
            for metric in milestone_metrics:
                print(f"{path.name}: {metric}", flush=True)
        environments = {record['environment'] for record in records}
        horizons = {record['events_per_environment'] for record in records}
        output.write_text(json.dumps(dict(environment=next(iter(environments)) if len(environments) == 1 else None,
                                         events_per_environment=next(iter(horizons)) if len(horizons) == 1 else None,
                                         ranking_check=args.ranking_check,
                                         horizon_multiplier=args.horizon_multiplier,
                                         stopping_rule='S_returns_to_empty_or_event_cap',
                                         target_empty_returns=args.empty_returns,
                                         returned_empty_definition='reached_target_empty_returns',
                                         service_coupling='per_server_first_service_unit_rate_work',
                                         milestone_rule='min(Sth_return, milestone_cap)',
                                         environment_seeds=seeds, action_seeds=seeds,
                                         results=records), indent=2) + '\n')
        print(f'Saved {output}', flush=True)
        # evaluate() has already joined its worker pool. Release this policy's parent
        # environments (including simulator reference cycles) before creating more.
        del envs, policy, state, checkpoint, config, env_config
        gc.collect()
        print(f'Finished {path.name}; environments released before next policy', flush=True)


if __name__ == '__main__':
    main()
