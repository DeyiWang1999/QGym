"""Evaluate hospital ZO actors with integer bed allocations and PPO metrics."""
import argparse
import gc
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import yaml
from RL.PPO.ZO_trainer_hospital import ZOPolicy, EventRecorder, make_environment
from RL.utils.eval import parallel_eval


DEFAULT_POLICY_CONFIG = ROOT / 'RL' / 'policy_configs' / 'ZO.yaml'


class EvaluationPolicy(ZOPolicy):
    def set_training_mode(self, mode):
        self.train(mode)

    def predict(self, observation):
        with torch.no_grad():
            return self.act(torch.as_tensor(observation)).cpu().numpy(), None


def evaluation_seeds(first_seed=None):
    """Use the current ZO.yaml evaluation seed unless explicitly overridden."""
    if first_seed is None:
        with DEFAULT_POLICY_CONFIG.open() as stream:
            first_seed = yaml.safe_load(stream)['env']['test_seed']
    if type(first_seed) is not int or not 0 <= first_seed <= 2**32 - 100:
        raise ValueError('Test seeds must be uint32 values (100 consecutive seeds)')
    return list(range(first_seed, first_seed + 100))


def load_policy(checkpoint):
    state = checkpoint['policy_state_dict']
    pools = torch.as_tensor(checkpoint['env_config']['server_pool_size'])
    if not torch.equal(state['server_pool_size'], pools):
        raise ValueError('Checkpoint policy and environment server_pool_size disagree')
    policy = EvaluationPolicy(state['network'], pools, checkpoint['config']['model']['scale'])
    policy.load_state_dict(state, strict=True)
    policy.set_training_mode(False)
    return policy


def make_evaluation_environment(env_config, config, seed):
    """Keep pool rows and per-bed rates, matching hospital pretrain evaluation."""
    env = make_environment(env_config, config, seed)
    env.st_argmin = EventRecorder(env.st_argmin)
    return env


def validate_seeds(checkpoint, seeds):
    """ZO's so-called evaluation trajectories are used to select updates."""
    config = checkpoint['config']
    training = config['training']
    used = set(checkpoint['evaluation_seeds'])
    configured = training.get('evaluation_seeds')
    used.update(configured if configured is not None else range(
        config['env']['test_seed'],
        config['env']['test_seed'] + training['evaluation_trajectories']))
    # Pre-training evaluation determines the actor's normalization statistics.
    used.update(range(config['env']['test_seed'], config['env']['test_seed'] + 100))
    used.add(config['env']['train_seed'])
    overlap = used.intersection(seeds)
    overlap = False
    if overlap:
        raise ValueError(f'Test seeds overlap ZO training/normalization seeds: {sorted(overlap)}')
    return sorted(used)


def load_checkpoint(path):
    # Only load trusted checkpoints: they contain config and simulator objects.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('config', 'env_config', 'evaluation_seeds', 'policy_state_dict'):
        if key not in checkpoint:
            raise ValueError(f'{path}: missing checkpoint metadata {key}')
    return checkpoint


def evaluation_schedule(env_config, ranking_check):
    base = env_config['test_T']
    if isinstance(base, bool) or not isinstance(base, int) or base <= 0:
        raise ValueError('Checkpoint env_config.test_T must be a positive integer')
    milestones = [base * i for i in range(1, 6)] if ranking_check else None
    return (base * 5 if ranking_check else base), milestones


def load_selected_checkpoint(path):
    if path.name == 'final_policy.pt':
        checkpoint = load_checkpoint(path.parent / 'original_000000.pt')
        checkpoint['policy_state_dict'] = torch.load(path, map_location='cpu', weights_only=True)
    else:
        checkpoint = load_checkpoint(path)
    # Training continuation states can be large and are not used for reset evaluation.
    checkpoint.pop('initial_states', None)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dir', type=Path, nargs='?', default=Path(__file__).parent / 'hospital')
    parser.add_argument('--test-seed', type=int, default=None,
                        help='First of 100 seeds; defaults to env.test_seed in RL/policy_configs/ZO.yaml (42)')
    parser.add_argument('--indices', type=int, nargs='+', default=list(range(0, 101, 10)))
    parser.add_argument('--include-final', action='store_true', help='Also evaluate final_policy.pt using original_000000.pt metadata')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ranking-check', action='store_true',
                        help="Run 5 times the checkpoint's test_T; record cumulative queue mean/std every test_T events")
    parser.add_argument('--check-only', action='store_true', help='Validate files, hospital policies, and seed metadata without rollouts')
    args = parser.parse_args()
    try:
        seeds = evaluation_seeds(args.test_seed)
    except ValueError as error:
        parser.error(str(error))
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
        evaluation_schedule(env_config, args.ranking_check)
        # Hospital rows already represent pools; num_pool does not expand them.
        policy = load_policy(checkpoint)
        del policy
        jobs.append(path)
        eval_t, milestones = evaluation_schedule(env_config, args.ranking_check)
        print(f'{path.name}: {env_config["name"]}; base test_T={env_config["test_T"]}; '
              f'{eval_t} events; milestones={milestones}', flush=True)
        del checkpoint, env_config
        gc.collect()
    print(f'Validated {len(jobs)} policies; 100 environments; seeds {seeds[0]}–{seeds[-1]}', flush=True)
    if args.check_only:
        return
    torch.set_num_threads(1)
    output = args.output or args.checkpoint_dir / (
        'ppo_ranking_evaluation.json' if args.ranking_check else 'ppo_evaluation.json')
    records = []
    for path in jobs:
        checkpoint = load_selected_checkpoint(path)
        used = validate_seeds(checkpoint, seeds)
        config, env_config = checkpoint['config'], checkpoint['env_config']
        eval_t, milestones = evaluation_schedule(env_config, args.ranking_check)
        policy = load_policy(checkpoint)
        envs = [make_evaluation_environment(env_config, config, seed) for seed in seeds]
        evaluator = parallel_eval(
            model=SimpleNamespace(policy=policy), eval_env=envs, eval_freq=1,
            eval_t=eval_t, test_policy='softmax', test_seed=seeds[0],
            init_test_queues=torch.tensor([env_config['init_queues']]), test_batch=100,
            device='cpu', num_pool=1, time_f=False, policy_name='WC',
            per_iter_normal_obs=False, env_config_name=env_config['name'], bc=False,
            seed_actions=True, milestones=milestones)
        print(f'Evaluating {path.name}', flush=True)
        values = evaluator.eval()
        if not all(torch.isfinite(value).all() for value in values):
            raise RuntimeError(f'Nonfinite metrics for {path.name}')
        names = ('queue_mean', 'queue_std_across_queues', 'time_mean', 'time_max',
                 'time_min', 'time_std', 'total_queue_mean', 'total_queue_std')
        records.append(dict(checkpoint=path.name, metrics=dict(zip(names, map(float, values))),
                            environment=env_config['name'], base_test_T=env_config['test_T'],
                            events_per_environment=eval_t,
                            excluded_training_seeds=used,
                            seed_overlap_enforced=False,
                            server_pool_size=policy.server_pool_size.tolist()))
        if args.ranking_check:
            records[-1]['milestones'] = evaluator.milestone_metrics
            for metric in evaluator.milestone_metrics:
                print(f"{path.name}: {metric}", flush=True)
        environments = {record['environment'] for record in records}
        horizons = {record['events_per_environment'] for record in records}
        output.write_text(json.dumps(dict(environment=next(iter(environments)) if len(environments) == 1 else None,
                                         events_per_environment=next(iter(horizons)) if len(horizons) == 1 else None,
                                         ranking_check=args.ranking_check,
                                         action_mode='integer_pool_allocation',
                                         environment_seeds=seeds, action_seeds=seeds,
                                         results=records), indent=2) + '\n')
        print(f'Saved {output}', flush=True)
        # eval() has already joined its worker pool. Release this policy's parent
        # environments (including simulator reference cycles) before creating more.
        del evaluator, envs, policy, checkpoint, config, env_config
        gc.collect()
        print(f'Finished {path.name}; environments released before next policy', flush=True)


if __name__ == '__main__':
    main()
