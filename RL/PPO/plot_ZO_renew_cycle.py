"""Replay original/perturbed policies for one coupled cycle and plot real time.

Run from any directory. --cycle is zero-based; omitted selects the cycle with
the largest max-minus-min step count across policies in the saved history.
No training or checkpoint files are modified. Requires the training dependencies
and matplotlib. Only load trusted PyTorch checkpoints.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from RL.PPO.ZO_renew_trainer import (
    ZOPolicy, make_environment, install_streams, paired_step,
    coupled_action, parameter_partitions,
)


def reconstruct(checkpoint, history):
    """Replay the dedicated perturbation RNG, then verify the next update."""
    config = checkpoint['config']
    training = config['training']
    env_config = checkpoint['env_config']
    env = make_environment(env_config, config, config['env']['train_seed'])
    policy = ZOPolicy(env.network[0], config['model']['scale'],
                      config['env']['randomize'], env.s // env_config['num_pool'])
    policy.load_state_dict(checkpoint['policy_state_dict'])
    parts = parameter_partitions(policy, training['mode'], training['partition_count'])
    generator = torch.Generator().manual_seed(training['perturbation_seed'])
    iteration = checkpoint['iteration']
    # Directions are not saved in history: replay every earlier RNG draw.
    for i in range(iteration + 1):
        directions = []
        for index in history[i]['active_partitions']:
            part = parts[index]
            directions.append(torch.randint(0, 2, (part.stop - part.start,),
                                             generator=generator).float() * 2 - 1)
    record = history[iteration]
    base = parameters_to_vector(policy.parameters()).detach().clone()
    updated = base.clone()
    candidate = base.clone()
    policies, labels = [copy.deepcopy(policy)], ['Original']
    split = training['mode'] == 'split_vanilla'
    accepted = record['accepted'] * len(directions) if split else record['accepted']
    for index, direction, accept in zip(record['active_partitions'], directions, accepted):
        part = parts[index]
        distance = record['perturbation_distances'][index]
        if not split:
            candidate = base.clone()
        candidate[part] += distance * direction
        if accept:
            updated[part] += training['update_ratio'] * distance * direction
        if not split:
            perturbed = copy.deepcopy(policy)
            vector_to_parameters(candidate.clone(), perturbed.parameters())
            policies.append(perturbed)
            labels.append(f'Perturbed partition {index}')
    if split and directions:
        perturbed = copy.deepcopy(policy)
        vector_to_parameters(candidate.clone(), perturbed.parameters())
        policies.append(perturbed)
        labels.append('Combined perturbation')
    return policies, labels, updated


def replay(policy, checkpoint, seed, max_steps):
    env = make_environment(checkpoint['env_config'], checkpoint['config'], seed)
    install_streams(env, seed)
    generator = torch.Generator().manual_seed(seed)
    times, queues = [0.0], [env.env_state.queues[0].numpy().copy()]
    busy = False
    cost = 0.0
    policy.eval()
    with torch.no_grad():
        while True:
            if max_steps and len(times) - 1 >= max_steps:
                raise RuntimeError('Replay exceeded --max-steps; no partial cycle plotted')
            action = coupled_action(policy, env.env_state.queues, generator)
            info = paired_step(env, action)[-1]
            dt = float(info['event_time'])
            if not math.isfinite(dt) or dt < 0:
                raise RuntimeError(f'Invalid event duration: {dt}')
            times.append(float(env.env_state.time.item()))
            queues.append(env.env_state.queues[0].numpy().copy())
            cost += float(info['cost'])
            nonempty = bool(env.env_state.queues.any())
            busy = busy or nonempty
            if busy and not nonempty:
                break
    return np.array(times), np.array(queues), cost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'RL/PPO/criss_cross_bh_renewZO_s_50')
    parser.add_argument('--iteration', type=int, default=60)
    parser.add_argument('--cycle', type=int, help='Zero-based cycle index (0 through 49 for this run)')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--max-steps', type=int, default=0, help='Optional failure guard; 0 means unlimited')
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(args.run_dir / f'original_{args.iteration:06d}.pt',
                            map_location='cpu', weights_only=False)
    history = {}
    for line in (args.run_dir / 'history.jsonl').read_text().splitlines():
        record = json.loads(line)
        if record['iteration'] in history:
            raise ValueError('Duplicate history iteration; cannot unambiguously replay RNG')
        history[record['iteration']] = record
    policies, labels, updated = reconstruct(checkpoint, history)
    next_path = args.run_dir / f'original_{args.iteration + 1:06d}.pt'
    if next_path.exists():
        next_checkpoint = torch.load(next_path, map_location='cpu', weights_only=False)
        next_policy = copy.deepcopy(policies[0])
        next_policy.load_state_dict(next_checkpoint['policy_state_dict'])
        if not torch.equal(updated, parameters_to_vector(next_policy.parameters()).detach()):
            raise RuntimeError('Reconstructed update does not match the next checkpoint')
        print('Verified reconstructed update against the next checkpoint.')
    seeds = checkpoint['evaluation_seeds']
    steps = np.asarray(history[args.iteration]['trajectory_steps']).reshape(len(policies), len(seeds))
    cycle = int(np.argmax(np.ptp(steps, axis=0))) if args.cycle is None else args.cycle
    if not 0 <= cycle < len(seeds):
        raise ValueError(f'--cycle must be between 0 and {len(seeds) - 1}')
    seed = seeds[cycle]
    traces = []
    for index, (policy, label) in enumerate(zip(policies, labels)):
        times, queues, cost = replay(policy, checkpoint, seed, args.max_steps)
        count = len(times) - 1
        if count != steps[index, cycle]:
            raise RuntimeError(f'{label}: replayed {count} steps; history has {steps[index, cycle]}')
        traces.append((times, queues))
        print(f'{label}: steps={count}, duration={times[-1]:.9f}, average cost={cost/times[-1]:.9f}')
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(16, 7))
    end = max(times[-1] for times, _ in traces)
    peak = max(float(queues.sum(axis=1).max()) for _, queues in traces)
    colors = plt.get_cmap('tab10')
    styles = ['-', '--', '-.', ':']
    for index, (label, (times, queues)) in enumerate(zip(labels, traces)):
        ax.step(times, queues.sum(axis=1), where='post',
                color='black' if index == 0 else colors((index - 1) % 10),
                linestyle=styles[index % len(styles)],
                linewidth=2.0 if index == 0 else 1.3,
                alpha=0.9 if index == 0 else 0.75,
                label=f'{label} (return t={times[-1]:.2f})')
    ax.set_xlim(0, end * 1.015)
    ax.set_ylim(0, max(1, peak * 1.05))
    ax.grid(alpha=0.2)
    ax.set_xlabel('Simulation time')
    ax.set_ylabel('Total queue length (all queues)')
    ax.set_title(f'Iteration {args.iteration} | coupled cycle index {cycle} | seed {seed}\n'
                 'Each trajectory ends at its first return to empty', fontsize=14)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9)
    fig.tight_layout()
    output = args.output or args.run_dir / f'cycle_{args.iteration:06d}_{cycle:02d}.png'
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f'Saved {output.resolve()}')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
