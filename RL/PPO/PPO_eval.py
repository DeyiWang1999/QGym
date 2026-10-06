"""Evaluate every saved PPO policy with the training evaluation routine.

Example (from RL/PPO):
    python PPO_eval.py reentrant_5_hyper_PPO --test_seed 142
"""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
# Checkpoints saved by train.py reference the top-level policies package.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'RL'))

# Match train.py's suppression of Gym's repeated worker import notice.
try:
    from gym_notices import notices as gym_notices
except ImportError:
    pass
else:
    for version, notice in list(gym_notices.notices.items()):
        if notice.startswith('Gym has been unmaintained since 2022'):
            del gym_notices.notices[version]

import cloudpickle
import torch
import yaml

from RL.utils.eval import parallel_eval
from RL.utils.rl_env import load_rl_p_env


TEST_BATCH = 400
METRIC_NAMES = (
    'queue_mean', 'queue_std_across_queues', 'time_mean', 'time_max',
    'time_min', 'time_std', 'total_queue_mean', 'total_queue_std',
    'total_holding_cost_mean', 'total_holding_cost_std',
)


def evaluate_policy(path, env_config, env_config_name, test_seed, config=None):
    # These are full policy objects, as written by SavingPolicyEval.
    with path.open('rb') as checkpoint:
        policy = cloudpickle.load(checkpoint)
    policy_name = {'WC_Policy': 'WC', 'Vanilla_Policy': 'vanilla'}.get(
        type(policy).__name__)
    if policy_name is None:
        raise ValueError(f'{path}: unsupported policy type {type(policy).__name__}')
    if config is None:
        with (ROOT / 'RL' / 'policy_configs' / f'{policy_name}.yaml').open() as source:
            config = yaml.safe_load(source)
    if config['model']['policy_name'] != policy_name:
        raise ValueError(f'{path}: policy type does not match --config')

    envs = [load_rl_p_env(
        env_config=env_config, temp=config['env']['env_temp'], batch=1,
        seed=seed, policy_name=policy_name, device=torch.device('cpu'),
    ) for seed in range(test_seed, test_seed + TEST_BATCH)]
    evaluator = parallel_eval(
        model=SimpleNamespace(policy=policy), eval_env=envs, eval_freq=1,
        eval_t=env_config['test_T'], test_policy=config['policy']['test_policy'],
        test_seed=test_seed,
        init_test_queues=torch.tensor([env_config['init_queues']]).float(),
        test_batch=TEST_BATCH, device='cpu', num_pool=env_config['num_pool'],
        time_f=policy.time_f, randomize=policy.randomize, policy_name=policy_name,
        per_iter_normal_obs=config['training']['per_iter_normal_obs'],
        env_config_name=env_config_name, bc=False,
    )
    # Call eval directly: never clone behavior or update saved normalization.
    # Leave seed_actions at its training default (False).
    values = evaluator.eval()
    if not all(torch.isfinite(value).all() for value in values):
        raise RuntimeError(f'Nonfinite evaluation metrics for {path}')
    return dict(zip(METRIC_NAMES, map(float, values)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dir', type=Path)
    parser.add_argument('--test_seed', '--test-seed', type=int, default=142,
                        help='First of 400 consecutive environment seeds (default: 142)')
    parser.add_argument('--env-config', help='Environment name; inferred from folder by removing _PPO')
    parser.add_argument('--config', type=Path,
                        help='Training policy YAML; defaults to WC.yaml or vanilla.yaml for the saved policy')
    parser.add_argument('--output', type=Path, help='Optional results JSON path')
    args = parser.parse_args(argv)
    if not 0 <= args.test_seed <= 2**32 - TEST_BATCH:
        parser.error('All 400 test seeds must be uint32 values')
    folder = args.checkpoint_dir
    if not folder.is_dir() and not folder.is_absolute():
        folder = Path(__file__).parent / folder
    if not folder.is_dir():
        parser.error(f'Checkpoint directory not found: {folder}')
    paths = sorted(folder.glob('*.pkl'))
    if not paths:
        parser.error(f'No .pkl policies found in {folder}')
    env_name = args.env_config or folder.resolve().name.removesuffix('_PPO')
    env_path = ROOT / 'configs' / 'env' / f'{env_name}.yaml'
    if not env_path.is_file():
        parser.error(f'Environment config not found: {env_path}; use --env-config')
    with env_path.open() as source:
        env_config = yaml.safe_load(source)
    config = None
    if args.config:
        with args.config.open() as source:
            config = yaml.safe_load(source)
    output = args.output or folder / f'ppo_evaluation_seed_{args.test_seed}.json'
    seeds = list(range(args.test_seed, args.test_seed + TEST_BATCH))
    report = dict(environment=env_name, events_per_environment=env_config['test_T'],
                  environment_seeds=seeds, results=[])
    torch.set_num_threads(1)
    print(f'Evaluating {len(paths)} policies on {TEST_BATCH} trajectories each; '
          f'seeds {seeds[0]} through {seeds[-1]}', flush=True)
    for path in paths:
        print(f'Evaluating {path.name}', flush=True)
        metrics = evaluate_policy(path, env_config, env_name, args.test_seed, config)
        report['results'].append(dict(checkpoint=path.name, metrics=metrics))
        output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'Saved {output}', flush=True)
        gc.collect()


if __name__ == '__main__':
    main()
