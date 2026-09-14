"""Actor-only, paired-comparison zeroth-order training (CPU only)."""
import copy
import json
import math
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from RL.utils.rl_env import load_rl_p_env
from main.env import Obs


class ZOPolicy(nn.Module):
    """WC PPO's Tanh actor architecture and orthogonal initialization, no critic."""

    def __init__(self, network, scale=10, randomize=False, original_servers=None):
        super().__init__()
        self.register_buffer('network', torch.as_tensor(network).float())
        self.s, self.q = self.network.shape
        if not (self.network.sum(-1) > 0).all():
            raise ValueError('Each server must have at least one compatible queue')
        self.randomize = randomize
        servers = original_servers or self.s
        widths = [self.q, scale * self.q,
                  scale * int(math.sqrt(self.q * servers)), scale * servers]
        layers = []
        for left, right in zip(widths, widths[1:]):
            layer = nn.Linear(left, right)
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2))
            nn.init.zeros_(layer.bias)
            layers.extend([layer, nn.Tanh()])
        self.policy_net = nn.Sequential(*layers)
        self.action_net = nn.Linear(widths[-1], self.s * self.q)
        nn.init.orthogonal_(self.action_net.weight, gain=0.01)
        nn.init.zeros_(self.action_net.bias)

    def probabilities(self, queues):
        queues = queues.float().reshape(-1, self.q)
        logits = self.action_net(self.policy_net(queues)).reshape(-1, self.s, self.q)
        # Mask logits before softmax to avoid underflow leaving a busy server idle.
        allowed = (self.network > 0).unsqueeze(0) & (queues > 0).unsqueeze(1)
        allowed = torch.where(allowed.any(-1, keepdim=True), allowed,
                              (self.network > 0).unsqueeze(0))
        return logits.masked_fill(~allowed, -torch.inf).softmax(-1)

    def act(self, queues, generator=None, *, deterministic=False):
        probs = self.probabilities(queues)
        if self.randomize and not deterministic:
            indices = torch.multinomial(probs.reshape(-1, self.q), 1,
                                        generator=generator).reshape(-1, self.s)
        else:
            indices = probs.argmax(-1)
        return F.one_hot(indices, self.q).float()


class EventRecorder(nn.Module):
    """Record the actual event index, including routed service completions."""

    def __init__(self, original):
        super().__init__()
        self.original = original
        self.index = 0

    def forward(self, times):
        self.index = int(times.argmin())
        return self.original(times)


def make_environment(env_config, config, seed):
    env = load_rl_p_env(copy.deepcopy(env_config), config['env']['env_temp'],
                        1, seed, 'WC', torch.device('cpu'))
    # PPO expands the actor's server rows; expand simulator rows consistently.
    pool = env_config['num_pool']
    env.network = env.network.repeat_interleave(pool, dim=1)
    env.mu = env.mu.repeat_interleave(pool, dim=1)
    env.s = env.network.shape[1]
    return env


def install_streams(env, seed):
    """Key draws by source event, so service ordering cannot shift arrival draws.

    The existing simulator draws whole vectors on routed and external arrivals.
    Separate streams for each source event preserve those sampling laws while
    coupling corresponding events across policies.
    """
    env.st_argmin = EventRecorder(env.st_argmin)
    streams = {}
    for kind in range(2):
        for event in range(2 * env.q):
            streams[kind, event] = np.random.RandomState(
                np.random.SeedSequence([seed, kind, event]).generate_state(1)[0])
    original_arrivals = env.draw_inter_arrivals_core
    original_service = env.draw_service_core

    def draw(original, kind, instance, time):
        previous = instance.state
        instance.state = streams[kind, instance.st_argmin.index]
        try:
            result = original(instance, time)
            if kind == 0:
                instance.zo_last_interarrivals = result
            return result
        finally:
            instance.state = previous

    env.draw_inter_arrivals_core = lambda instance, time: draw(original_arrivals, 0, instance, time)
    env.draw_service_core = lambda instance, time: draw(original_service, 1, instance, time)


def paired_step(env, action):
    """Keep external clocks independent of internally routed arrivals.

    The base simulator also adds an interarrival draw to a routed destination's
    clock. Undo that clock change locally: routing is not an external arrival.
    """
    previous_clocks = env.env_state.arrival_times.clone()
    env.zo_last_interarrivals = None
    result = env.step(action)
    dt = result[-1]['event_time']
    clocks = previous_clocks - dt
    event = env.st_argmin.index
    if event < env.q:
        if env.zo_last_interarrivals is None:
            raise ValueError('External arrival rows must add a job to a queue')
        clocks[0, event] += env.zo_last_interarrivals[0, event]
    env.env_state = env.env_state._replace(arrival_times=clocks)
    return result


def evaluate_trajectory(job):
    env_config, config, policy, initial_state, seed = job
    torch.set_num_threads(1)
    env = make_environment(env_config, config, seed)
    env.env_state = copy.deepcopy(initial_state)
    env.obs = Obs(env.env_state.queues, env.env_state.time)
    install_streams(env, seed)
    integral = elapsed = 0.0
    arrivals = 0
    policy.eval()
    with torch.no_grad():
        while arrivals < config['training']['evaluation_length']:
            queues = env.env_state.queues.clone()
            _, _, _, _, info = paired_step(env, policy.act(queues, deterministic=True))
            dt = float(info['event_time'])
            if not math.isfinite(dt) or dt < 0:
                raise RuntimeError('Invalid event duration')
            integral += float(queues.sum()) * dt
            elapsed += dt
            arrivals += int(env.st_argmin.index < env.q)
    if elapsed <= 0:
        raise RuntimeError('Evaluation must have positive elapsed time')
    return integral / elapsed, copy.deepcopy(env.env_state)


def perturbation_distance(training, iteration):
    start, end = training['initial_perturbation'], training['ending_perturbation']
    fraction = iteration / max(training['total_iterations'] - 1, 1)
    scheduler = training['perturbation_scheduler']
    if scheduler == 'logarithmic':
        # Constant slope in log(distance), with exact configured endpoints.
        return math.exp(math.log(start) + fraction * (math.log(end) - math.log(start)))
    if scheduler == 'linear':
        return start + fraction * (end - start)
    if scheduler == 'cosine':
        return end + (start - end) * (1 + math.cos(math.pi * fraction)) / 2
    raise ValueError(f'Unknown perturbation scheduler: {scheduler}')


def parameter_partitions(policy, mode, count):
    sizes = [p.numel() for p in policy.parameters()]
    total = sum(sizes)
    if mode == 'vanilla':
        return [slice(0, total)]
    if mode == 'partitioned':
        if not isinstance(count, int) or not 1 <= count <= total:
            raise ValueError(f'partition_count must be an integer between 1 and {total}')
        sizes = [total // count] * count
        sizes[-1] += total % count
    elif mode != 'split_layer':
        raise ValueError(f'Unknown training mode: {mode}')
    offset, partitions = 0, []
    for size in sizes:
        partitions.append(slice(offset, offset + size))
        offset += size
    return partitions


class ZerothOrderTrainer:
    def __init__(self, env_config, config, output_dir):
        self.env_config, self.config = copy.deepcopy(env_config), copy.deepcopy(config)
        self.training = config['training']
        t = self.training
        for key in ('total_iterations', 'evaluation_trajectories', 'evaluation_length', 'max_cpus'):
            if not isinstance(t[key], int) or t[key] <= 0:
                raise ValueError(f'{key} must be a positive integer')
        if not 0 < t['ending_perturbation'] <= t['initial_perturbation']:
            raise ValueError('Require 0 < ending_perturbation <= initial_perturbation')
        if not math.isfinite(t['update_ratio']) or t['update_ratio'] < 0:
            raise ValueError('update_ratio must be finite and nonnegative')
        perturbation_distance(t, 0)
        bc = config['behavior_cloning']
        for key in ('epochs', 'num_samples', 'batch_size'):
            if not isinstance(bc[key], int) or bc[key] <= 0:
                raise ValueError(f'behavior_cloning.{key} must be a positive integer')
        if not math.isfinite(bc['learning_rate']) or bc['learning_rate'] <= 0:
            raise ValueError('behavior_cloning.learning_rate must be positive and finite')
        if config['env']['device'] != 'cpu':
            raise ValueError('ZO evaluation supports CPU only')
        torch.set_num_threads(1)
        torch.manual_seed(config['env']['model_seed'])
        np.random.seed(config['env']['model_seed'])
        env = make_environment(env_config, config, config['env']['train_seed'])
        self.policy = ZOPolicy(env.network[0], config['model']['scale'], config['env']['randomize'],
                               env.s // env_config['num_pool'])
        self.partitions = parameter_partitions(self.policy, t['mode'], t['partition_count'])
        self.generator = torch.Generator().manual_seed(t['perturbation_seed'])
        self.seeds = t['evaluation_seeds']
        if self.seeds is None:
            self.seeds = list(range(config['env']['test_seed'],
                                    config['env']['test_seed'] + t['evaluation_trajectories']))
        if (len(self.seeds) != t['evaluation_trajectories'] or
                len(set(self.seeds)) != len(self.seeds) or
                any(not isinstance(s, int) or not 0 <= s < 2**32 for s in self.seeds)):
            raise ValueError('Provide one distinct uint32 evaluation seed per trajectory')
        initial = t.get('initial_queues')
        if initial is not None and len(initial) != len(self.seeds):
            raise ValueError('initial_queues requires one queue vector per trajectory')
        self.states = []
        for index, seed in enumerate(self.seeds):
            queues = torch.tensor([env_config['init_queues'] if initial is None else initial[index]]).float()
            if queues.shape != (1, env.q) or not torch.isfinite(queues).all() or (queues < 0).any() or (queues != queues.floor()).any():
                raise ValueError('Initial queues must be nonnegative integer vectors')
            env.seed = seed
            env.reset(init_queues=queues)
            self.states.append(copy.deepcopy(env.env_state))
        self.workers = min(48, t['max_cpus'], os.cpu_count() or 1,
                           len(self.seeds) * (len(self.partitions) + 1))
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)

    def pretrain(self):
        """vanilla_bc: uniform integer queues 0..100, softmax teacher, MSE/Adam."""
        bc = self.config['behavior_cloning']
        optimizer = torch.optim.Adam(self.policy.parameters(), lr=bc['learning_rate'])
        self.policy.train()
        for _ in range(bc['epochs']):
            for offset in range(0, bc['num_samples'], bc['batch_size']):
                size = min(bc['batch_size'], bc['num_samples'] - offset)
                queues = torch.tensor(np.random.randint(0, 101, (size, self.policy.q))).float()
                target = queues.softmax(-1).unsqueeze(1) * self.policy.network
                target = torch.minimum(target, queues.unsqueeze(1))
                target = target + (target == 0).all(-1, keepdim=True) * self.policy.network
                target = target / target.sum(-1, keepdim=True)
                optimizer.zero_grad()
                loss = F.mse_loss(self.policy.probabilities(queues), target)
                loss.backward()
                optimizer.step()
        self.policy.eval()
        torch.save(self.policy.state_dict(), self.output_dir / 'initial_policy.pt')

    def train(self):
        self.pretrain()
        pool = ProcessPoolExecutor(self.workers, mp_context=mp.get_context('spawn')) if self.workers > 1 else None
        try:
            for iteration in range(self.training['total_iterations']):
                distance = perturbation_distance(self.training, iteration)
                base = parameters_to_vector(self.policy.parameters()).detach().clone()
                torch.save({'iteration': iteration, 'policy_state_dict': self.policy.state_dict(),
                            'config': self.config, 'env_config': self.env_config,
                            'initial_states': self.states, 'evaluation_seeds': self.seeds},
                           self.output_dir / f'original_{iteration:06d}.pt')
                policies, directions = [copy.deepcopy(self.policy)], []
                for part in self.partitions:
                    direction = torch.randint(0, 2, (part.stop - part.start,), generator=self.generator).float() * 2 - 1
                    candidate = base.clone()
                    candidate[part] += distance * direction
                    policy = copy.deepcopy(self.policy)
                    vector_to_parameters(candidate, policy.parameters())
                    policies.append(policy)
                    directions.append(direction)
                jobs = [(self.env_config, self.config, policy, state, seed)
                        for policy in policies for state, seed in zip(self.states, self.seeds)]
                results = list(pool.map(evaluate_trajectory, jobs)) if pool else list(map(evaluate_trajectory, jobs))
                n = len(self.seeds)
                scores = [float(np.mean([r[0] for r in results[i:i+n]]))
                          for i in range(0, len(results), n)]
                if not all(math.isfinite(score) for score in scores):
                    raise RuntimeError('Nonfinite evaluation score')
                self.states = [result[1] for result in results[:n]]
                accepted = [score < scores[0] for score in scores[1:]]
                updated = base.clone()
                for part, direction, accept in zip(self.partitions, directions, accepted):
                    if accept:
                        updated[part] += self.training['update_ratio'] * distance * direction
                vector_to_parameters(updated, self.policy.parameters())
                record = dict(iteration=iteration, perturbation_distance=distance,
                              scores=scores, accepted=accepted)
                with (self.output_dir / 'history.jsonl').open('a') as stream:
                    stream.write(json.dumps(record) + '\n')
                print(record, flush=True)
            torch.save(self.policy.state_dict(), self.output_dir / 'final_policy.pt')
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
