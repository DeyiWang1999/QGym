"""Compare fixed-event and renewal checkpoints on the same trajectories.

Defaults: criss_cross_bh_split_1/original_000000.pt, 400 seeds, s=1,...,100.
Stop after BOTH 100*test_T events and 100 returns to empty, with no upper cap.
All estimates use cumulative holding cost / cumulative simulated time.
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cloudpickle
import numpy as np
import torch
from RL.PPO.ZO_eval import EvaluationPolicy, evaluation_schedule, load_selected_checkpoint
from RL.PPO.ZO_eval_renew import RenewalEnvironment
from RL.utils.rl_env import load_rl_p_env


def collect_trajectory(sim, policy, base, checkpoints=100):
    """Record both series without resetting between checkpoints."""
    event_records, return_records = [], []
    cost, returns = 0., 0

    def snapshot():
        if sim.time <= 0 or not np.isfinite(cost) or not np.isfinite(sim.time):
            raise RuntimeError('Nonfinite cost/time or nonpositive checkpoint time')
        return dict(events=sim.steps, cumulative_time=float(sim.time),
                    cumulative_cost=float(cost), average_cost=float(cost / sim.time))

    with torch.no_grad():
        # OR is intentional: continue until BOTH targets have been reached.
        while sim.steps < base * checkpoints or returns < checkpoints:
            was_nonempty = any(sim.jobs)
            action, _ = policy.predict(sim.queues().reshape(1, -1))
            increment, _ = sim.step(action)
            cost += increment
            if was_nonempty and not any(sim.jobs):
                returns += 1
                if returns <= checkpoints:
                    return_records.append(dict(s=returns, **snapshot()))
            if sim.steps <= base * checkpoints and sim.steps % base == 0:
                event_records.append(dict(s=sim.steps // base, **snapshot()))
    return dict(event_checkpoints=event_records, return_checkpoints=return_records,
                final_events=sim.steps, final_time=float(sim.time), final_returns=returns)


def _init_worker(policy_bytes, env_bytes, base, checkpoints):
    global _policy, _env_bytes, _base, _checkpoints
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _policy = cloudpickle.loads(policy_bytes)
    _policy.set_training_mode(False)
    _env_bytes, _base, _checkpoints = env_bytes, base, checkpoints


def _run_seed(seed):
    torch.manual_seed(seed)
    env = cloudpickle.loads(_env_bytes)
    env.seed = seed
    return dict(seed=seed, **collect_trajectory(
        RenewalEnvironment(env), _policy, _base, _checkpoints))


def summarize(trajectories, checkpoints):
    """Pointwise normal-approximation 95% CIs across seed estimates."""
    n = len(trajectories)
    if n < 2:
        raise ValueError('At least two trajectories are required for confidence intervals')
    summary = {}
    for key in ('event_checkpoints', 'return_checkpoints'):
        if any(len(path[key]) != checkpoints for path in trajectories):
            raise ValueError('All trajectories must complete every requested checkpoint')
        series = []
        for index in range(checkpoints):
            costs = np.array([path[key][index]['average_cost'] for path in trajectories])
            events = np.array([path[key][index]['events'] for path in trajectories])
            if not np.isfinite(costs).all():
                raise ValueError('Nonfinite cost estimate')
            mean, std = float(costs.mean()), float(costs.std(ddof=1))
            half_width = float(1.96 * std / np.sqrt(n))
            series.append(dict(s=index + 1, trajectories=n,
                               mean_events=float(events.mean()),
                               std_events=float(events.std(ddof=1)),
                               mean_average_cost=mean, std_average_cost=std,
                               ci95_half_width=half_width,
                               ci95_lower=mean - half_width, ci95_upper=mean + half_width))
        summary[key] = series
    return summary


def plot_results(report, figure_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label, marker in (
            ('event_checkpoints', 'Fixed event count', 'o'),
            ('return_checkpoints', 'Returns to empty', 's')):
        series = report['summary'][key]
        ax.errorbar([r['mean_events'] for r in series],
                    [r['mean_average_cost'] for r in series],
                    yerr=[r['ci95_half_width'] for r in series],
                    fmt=marker + '-', markersize=3, linewidth=1.2,
                    elinewidth=.7, capsize=2, label=label)
    ax.set_xlabel('Number of events (mean across trajectories for empty returns)')
    ax.set_ylabel('Mean cumulative holding cost / cumulative time')
    ax.set_title(f"{Path(report['checkpoint']).parent.name}/{Path(report['checkpoint']).name}\n"
                 f"{len(report['environment_seeds'])} trajectories; pointwise 95% CIs")
    ax.grid(alpha=.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('Must be a positive integer')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint', type=Path, nargs='?', default=
                        Path(__file__).parent / 'criss_cross_bh_split_1' / 'original_000000.pt')
    parser.add_argument('--test-seed', type=int, default=142)
    parser.add_argument('--trajectories', type=positive_int, default=400)
    parser.add_argument('--checkpoints', type=positive_int, default=100,
                        help='Record s=1,...,N for both methods (default: 100)')
    parser.add_argument('--workers', type=positive_int, default=min(48, os.cpu_count() or 1))
    parser.add_argument('--output', type=Path, help='JSON output; PNG uses the same stem')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.trajectories < 2:
        parser.error('--trajectories must be at least 2')
    seeds = list(range(args.test_seed, args.test_seed + args.trajectories))
    if not 0 <= seeds[0] <= seeds[-1] < 2**32:
        parser.error('Seeds must be uint32 values')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}')
    torch.set_num_threads(1)
    checkpoint = load_selected_checkpoint(args.checkpoint)
    config, env_config = checkpoint['config'], checkpoint['env_config']
    base, _ = evaluation_schedule(env_config, False)
    if env_config['num_pool'] != 1:
        parser.error('This evaluator requires num_pool=1, matching ZO_eval_renew.py')
    state = checkpoint['policy_state_dict']
    policy = EvaluationPolicy(state['network'], config['model']['scale'])
    policy.load_state_dict(state, strict=True)
    policy.set_training_mode(False)
    env = load_rl_p_env(copy.deepcopy(env_config), config['env']['env_temp'],
                        1, seeds[0], 'WC', torch.device('cpu'))
    RenewalEnvironment(copy.deepcopy(env))
    print(f'{args.checkpoint}: test_T={base}; {len(seeds)} trajectories; '
          f'each runs until BOTH {args.checkpoints * base} events AND '
          f'{args.checkpoints} returns to empty', flush=True)
    if args.check_only:
        return
    output = args.output or args.checkpoint.with_name(args.checkpoint.stem + '_renew_compare.json')
    if output.suffix.lower() != '.json':
        parser.error('--output must have a .json extension')
    output.parent.mkdir(parents=True, exist_ok=True)
    trajectories = {}
    # Keep completed paths during long runs; final statistics require ALL seeds.
    progress_path = output.with_suffix('.trajectories.jsonl')
    with progress_path.open('w', encoding='utf-8') as progress, ProcessPoolExecutor(
            max_workers=min(args.workers, len(seeds)), mp_context=mp.get_context('spawn'),
            initializer=_init_worker,
            initargs=(cloudpickle.dumps(policy), cloudpickle.dumps(env), base, args.checkpoints)) as pool:
        futures = [pool.submit(_run_seed, seed) for seed in seeds]
        for future in as_completed(futures):
            result = future.result()
            trajectories[result['seed']] = result
            progress.write(json.dumps(result, allow_nan=False) + '\n')
            progress.flush()
            print(f"Completed {len(trajectories)}/{len(seeds)}: seed={result['seed']}, "
                  f"events={result['final_events']}, returns={result['final_returns']}", flush=True)
    ordered = [trajectories[seed] for seed in seeds]
    report = dict(checkpoint=str(args.checkpoint.resolve()), environment=env_config['name'],
                  base_test_T=base, checkpoints=args.checkpoints, environment_seeds=seeds,
                  stopping_rule='both_event_and_return_targets_reached',
                  estimator='cumulative_holding_cost / cumulative_simulated_time',
                  ci95='mean +/- 1.96 * sample_std / sqrt(number_of_trajectories)',
                  summary=summarize(ordered, args.checkpoints), trajectories=ordered)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    figure_path = output.with_suffix('.png')
    plot_results(report, figure_path)
    print(f'Saved {output}\nSaved {figure_path}', flush=True)


if __name__ == '__main__':
    main()
