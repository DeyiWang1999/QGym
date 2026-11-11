"""Evaluate arrival-rate robustness; run from any directory.

    python RL/PPO/reentrant_7_robustness.py
    python RL/PPO/reentrant_7_robustness.py --smoke-test

Full runs use 400 seeds (142..541), the original test_T, and <=48 workers.
JSON and CSV are saved after every completed rate/policy combination.
"""
import argparse
import contextlib
import copy
import csv
import gc
import io
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

# Set before importing numerical libraries, including in spawned workers.
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'RL'))

from RL.PPO.PPO_eval import METRIC_NAMES
from RL.PPO.ZO_eval import EvaluationPolicy, load_selected_checkpoint
from RL.utils.eval import parallel_eval
from RL.utils.rl_env import load_rl_p_env
import cloudpickle
import numpy as np
import torch
import yaml
from utils import routing

RATES = (6, 7, 8, 9.5)
CHECKPOINTS = {
    'ppo': ROOT / 'RL/PPO/reentrant_7_PPO/policy_eval_000002.pkl',
    'split': ROOT / 'RL/PPO/reentrant_7_split/original_000090.pt',
}


def arrival_variants(original):
    """Change only the two external arrivals; preserve small entries exactly."""
    original = np.asarray(original)
    active = ~np.isclose(original, 1e-6, rtol=1e-5, atol=0)
    if original.ndim != 1 or active.sum() != 2:
        raise ValueError('Expected exactly two external arrival entries')
    if not np.allclose(original[active], 9 / 140):
        raise ValueError('Expected original external arrival rates near 9/140')
    variants = []
    for numerator in RATES:
        modified = original.copy()
        modified[active] = numerator / 140
        variants.append((numerator, modified))
    return np.flatnonzero(active).tolist(), variants


class TraditionalPolicy:
    """Same c-mu/c-mu-q priorities and assignment as Trainer.test_epoch."""
    def __init__(self, name, env):
        self.name = name
        self.network, self.mu, self.h = env.network, env.mu, env.h
        self.servers = torch.ones(env.s)

    def set_training_mode(self, mode):
        pass

    def predict(self, observation):
        queues = torch.as_tensor(observation).float()
        priority = self.h * self.mu
        if self.name == 'cmuq':
            priority = priority * queues.unsqueeze(1) * self.network
        values, servers, jobs = routing.pad_pool(
            2 * priority, queues, network=self.network,
            device='cpu', server_pool_size=self.servers)
        return torch.round(routing.linear_assignment_batch(values, servers, jobs)).numpy(), None


def evaluate(name, env_config, seeds):
    policy_name, temp, seed_actions = 'WC', 1.0, False
    if name == 'ppo':
        with CHECKPOINTS[name].open('rb') as source:
            policy = cloudpickle.load(source)
        policy_name = {'WC_Policy': 'WC', 'Vanilla_Policy': 'vanilla'}[type(policy).__name__]
        with (ROOT / 'RL/policy_configs' / f'{policy_name}.yaml').open() as source:
            config = yaml.safe_load(source)
        temp, seed_actions = config['env']['env_temp'], True
    elif name == 'split':
        checkpoint = load_selected_checkpoint(CHECKPOINTS[name])
        if checkpoint['env_config']['name'] != 'reentrant_7':
            raise ValueError('Split checkpoint has the wrong environment')
        config = checkpoint['config']
        state = checkpoint['policy_state_dict']
        policy = EvaluationPolicy(state['network'], config['model']['scale'])
        policy.load_state_dict(state, strict=True)
        temp, seed_actions = config['env']['env_temp'], True

    # The loader mutates its configuration; isolate each environment's copy.
    with contextlib.redirect_stdout(io.StringIO()):
        envs = [load_rl_p_env(copy.deepcopy(env_config), temp, 1, seed,
                              policy_name, torch.device('cpu')) for seed in seeds]
    if name in ('cmu', 'cmuq'):
        policy = TraditionalPolicy(name, envs[0])
    evaluator = parallel_eval(
        model=SimpleNamespace(policy=policy), eval_env=envs, eval_freq=1,
        eval_t=env_config['test_T'], test_policy='softmax', test_seed=seeds[0],
        init_test_queues=torch.tensor([env_config['init_queues']]).float(),
        test_batch=len(seeds), device='cpu', num_pool=1,
        time_f=getattr(policy, 'time_f', False), policy_name=policy_name,
        per_iter_normal_obs=False, env_config_name='reentrant_7', bc=False,
        seed_actions=seed_actions)
    values = evaluator.eval()
    if not all(torch.isfinite(value).all() for value in values):
        raise RuntimeError(f'Nonfinite metrics for {name}')
    return dict(zip(METRIC_NAMES, map(float, values)))


def save_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(output)
    with output.with_suffix('.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['arrival_rate', 'policy', *METRIC_NAMES])
        writer.writeheader()
        for result in report['results']:
            writer.writerow(dict(arrival_rate=result['arrival_rate'],
                                 policy=result['policy'], **result['metrics']))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Results JSON path (CSV alongside)')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Only 2 seeds and 20 events; separate smoke output')
    args = parser.parse_args(argv)
    output = args.output or ROOT / 'RL/PPO' / (
        'reentrant_7_robustness_smoke.json' if args.smoke_test else 'reentrant_7_robustness.json')
    if output.suffix.lower() != '.json':
        parser.error('--output must end in .json')
    for path in CHECKPOINTS.values():
        if not path.is_file():
            parser.error(f'Missing checkpoint: {path}')
    with (ROOT / 'configs/env/reentrant_7.yaml').open() as source:
        base = yaml.safe_load(source)
    original = np.load(ROOT / 'configs/env_data/reentrant_7/reentrant_7_lam.npy')
    indices, variants = arrival_variants(original)
    seeds = list(range(142, 144 if args.smoke_test else 542))
    if args.smoke_test:
        base['test_T'] = 20
    torch.set_num_threads(1)
    report = dict(environment='reentrant_7', smoke_test=args.smoke_test,
                  events_per_environment=base['test_T'], environment_seeds=seeds,
                  max_cpu_workers=48, workers=min(48, os.cpu_count() or 1, len(seeds)),
                  modified_arrival_indices=indices, original_arrival_rates=original.tolist(),
                  action_seeding={'ppo': 'same as environment seed',
                                  'split': 'same as environment seed',
                                  'cmu': 'deterministic', 'cmuq': 'deterministic'},
                  results=[])
    print(f'16 evaluations; {len(seeds)} trajectories each; {base["test_T"]} events each', flush=True)
    for numerator, arrivals in variants:
        for name in ('cmu', 'cmuq', 'ppo', 'split'):
            config = copy.deepcopy(base)
            config['lam_params']['val'] = arrivals.tolist()
            print(f'Evaluating {name}, arrivals={numerator}/140', flush=True)
            metrics = evaluate(name, config, seeds)
            report['results'].append(dict(
                arrival_rate=f'{numerator}/140', arrival_rate_value=numerator / 140,
                arrival_rates=arrivals.tolist(), policy=name,
                checkpoint=str(CHECKPOINTS[name].relative_to(ROOT)) if name in CHECKPOINTS else None,
                metrics=metrics))
            save_report(report, output)
            gc.collect()
    print(f'Saved {output} and {output.with_suffix(".csv")}', flush=True)


if __name__ == '__main__':
    main()
