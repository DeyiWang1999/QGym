"""Evaluate saved ZO actors using PPO's simulator, rollout, and metrics."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from RL.PPO.ZO_trainer import ZOPolicy
from RL.utils.eval import parallel_eval
from RL.utils.rl_env import load_rl_p_env


class EvaluationPolicy(ZOPolicy):
    def set_training_mode(self, mode):
        self.train(mode)

    def predict(self, observation):
        return self.act(torch.as_tensor(observation), deterministic=False).numpy(), None


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint_dir', type=Path, nargs='?', default=Path(__file__).parent / 'reentrant_9')
    parser.add_argument('--test-seed', type=int, default=3003)
    parser.add_argument('--indices', type=int, nargs='+', default=list(range(0, 101, 10)))
    parser.add_argument('--include-final', action='store_true', help='Also evaluate final_policy.pt using original_000000.pt metadata')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ranking-check', action='store_true',
                        help="Run 5 times the checkpoint's test_T; record cumulative queue mean/std every test_T events")
    parser.add_argument('--check-only', action='store_true', help='Validate files and seed separation without rollouts')
    args = parser.parse_args()
    seeds = list(range(args.test_seed, args.test_seed + 100))
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
        if path.name == 'final_policy.pt':
            checkpoint = load_checkpoint(args.checkpoint_dir / 'original_000000.pt')
            checkpoint['policy_state_dict'] = torch.load(path, map_location='cpu', weights_only=True)
        else:
            checkpoint = load_checkpoint(path)
        used = validate_seeds(checkpoint, seeds)
        env_config = checkpoint['env_config']
        evaluation_schedule(env_config, args.ranking_check)
        if env_config['num_pool'] != 1:
            raise ValueError(f'{path}: PPO-compatible evaluation currently requires num_pool=1')
        jobs.append((path, checkpoint, used))
        eval_t, milestones = evaluation_schedule(env_config, args.ranking_check)
        print(f'{path.name}: {env_config["name"]}; base test_T={env_config["test_T"]}; '
              f'{eval_t} events; milestones={milestones}', flush=True)
    print(f'Validated {len(jobs)} policies; 100 environments; seeds {seeds[0]}–{seeds[-1]}', flush=True)
    if args.check_only:
        return
    torch.set_num_threads(1)
    output = args.output or args.checkpoint_dir / (
        'ppo_ranking_evaluation.json' if args.ranking_check else 'ppo_evaluation.json')
    records = []
    for path, checkpoint, used in jobs:
        config, env_config = checkpoint['config'], checkpoint['env_config']
        eval_t, milestones = evaluation_schedule(env_config, args.ranking_check)
        state = checkpoint['policy_state_dict']
        policy = EvaluationPolicy(state['network'], config['model']['scale'])
        policy.load_state_dict(state, strict=True)
        envs = [load_rl_p_env(env_config, config['env']['env_temp'], 1, seed, 'WC', torch.device('cpu')) for seed in seeds]
        evaluator = parallel_eval(
            model=SimpleNamespace(policy=policy), eval_env=envs, eval_freq=1,
            eval_t=eval_t, test_policy='softmax', test_seed=args.test_seed,
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
                            excluded_training_seeds=used))
        if args.ranking_check:
            records[-1]['milestones'] = evaluator.milestone_metrics
            for metric in evaluator.milestone_metrics:
                print(f"{path.name}: {metric}", flush=True)
        environments = {record['environment'] for record in records}
        horizons = {record['events_per_environment'] for record in records}
        output.write_text(json.dumps(dict(environment=next(iter(environments)) if len(environments) == 1 else None,
                                         events_per_environment=next(iter(horizons)) if len(horizons) == 1 else None,
                                         ranking_check=args.ranking_check,
                                         environment_seeds=seeds, action_seeds=seeds,
                                         results=records), indent=2) + '\n')
        print(f'Saved {output}', flush=True)


if __name__ == '__main__':
    main()
