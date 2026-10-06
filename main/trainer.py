import copy
from joblib import Parallel, delayed, cpu_count, parallel_config

import sys
sys.path.append('/user/hc3295/queue-learning')

import numpy as np
import tqdm
from tqdm import trange

import torch
from torch import nn
import torch.nn.functional as F
import gym
import argparse
import math
import os
import torch.optim as optim
import matplotlib.pyplot as plt
import utils.routing as rt
import json
from main.env import DiffDiscreteEventSystem

import yaml
import pdb
from datetime import datetime


from utils.switchplot import *

class Trainer:
    def __init__(self, model_config, env_config, policy, optimizer, draw_service, draw_inter_arrivals, experiment_name):
        self.model_config = model_config
        self.env_config = env_config
        self.policy = policy
        self.optimizer = optimizer
        self.draw_service = draw_service
        self.draw_inter_arrivals = draw_inter_arrivals

        self.test_loss = []


        self.fig_dir = create_plot_dir(self.model_config, self.env_config, experiment_name = experiment_name)
        self.loss_dir = create_loss_dir(self.model_config, self.env_config, experiment_name = experiment_name)

        self.experiment_name = experiment_name



    def train_epoch(self):
        dq = DiffDiscreteEventSystem(self.env_config['network'], self.env_config['mu'], self.env_config['h'], 
                                    queue_event_options = self.env_config['queue_event_options'],
                                    batch = 1, 
                                    temp = self.model_config['env']['env_temp'], seed = self.model_config['env']['train_seed'],
                                    device = torch.device(self.model_config['env']['device']), draw_service = self.draw_service, draw_inter_arrivals = self.draw_inter_arrivals)

        # zero out the optimizer
        self.optimizer.zero_grad()

        # save grads
        back_outs = []
        def action_hook(grad):
            #grad = torch.clamp(grad, -threshold,threshold)
            back_outs.append(grad.tolist())
            #return grad
        
        nn_back_ins = []
        def priority_hook(grad):
            nn_back_ins.append(grad.tolist())

        # Train loop
        obs, state = dq.reset(seed = self.model_config['env']['train_seed'])
        total_cost = torch.tensor([[0.]]*self.model_config['opt']['train_batch']).to(self.model_config['env']['device'])
        time_weight_queue_len = torch.tensor([[0.]]*self.model_config['opt']['train_batch']).to(self.model_config['env']['device'])
        
        for _ in trange(self.env_config['train_T']):
            queues, time = dq.obs.queues, dq.obs.time
            
            pr = self.policy.train_forward(queues, time, dq.network, dq.h.unsqueeze(0).repeat(1,dq.s,1), dq.mu)
            
            pr.register_hook(priority_hook)
            
            # server pools
            pr = pr.repeat_interleave(1, dim = 1)
            
            if self.model_config['policy']['train_policy'] == 'sinkhorn':
                lex = torch.zeros(dq.batch, dq.s, dq.q).to(self.model_config['env']['device'])
                v, s_bar, q_bar = rt.pad_pool(2*pr + lex, queues.detach(), network = dq.network, device = self.model_config['env']['device'], server_pool_size = self.env_config['server_pool_size'])

                pr = rt.Sinkhorn.apply(-v, s_bar, q_bar, 
                                        self.model_config['policy']['sinkhorn']['num_iter'],
                                        self.model_config['policy']['sinkhorn']['temp'],
                                        self.model_config['policy']['sinkhorn']['eps'],
                                        self.model_config['policy']['sinkhorn']['back_temp'],
                                        self.model_config['env']['device'])[:,:dq.s,:dq.q]
                
                
            elif self.model_config['policy']['train_policy'] == 'softmax':
                
                pr = F.softmax(pr) * self.env_config['network'].unsqueeze(0)
                pr = torch.minimum(pr, queues.unsqueeze(1).repeat(1,self.env_config['network'].shape[-2],1)).clip(min = 1e-4)
                pr /= torch.sum(pr, dim = -1).unsqueeze(-1)
            else:
                pass
            
            # print(pr)
            action = pr
            # print(action)
            # print(queues)
            action.register_hook(action_hook)
            
            queues, reward, done, truncated, info = dq.step(action)
            # print(f"info: {info['cost']}", total_cost)
            total_cost = total_cost + info['cost']
            
            
            time_weight_queue_len = time_weight_queue_len + info['queues'] * info['event_time']
        
        # Backward
        
        loss = torch.mean(total_cost / self.env_config['train_T'])
        loss.backward()


        state = dq.obs
        # print(f"total_cost:\t{total_cost}, time:{state.time}")
        print(f"train cost:\t{torch.mean(total_cost / state.time)}")
        print(f"queue lengths: \t{torch.mean(time_weight_queue_len / state.time, dim = 0)}")
        # Gradient clipping and step
        torch.nn.utils.clip_grad_norm_(self.policy.network.parameters(), max_norm = self.model_config['opt']['grad_clip_norm'])
        self.optimizer.step()

        if not self.model_config['env']['train_restart']:
            init_train_queues = queues.detach()

        if self.model_config['env']['print_grads']:
            print("Action Grads")
            print(torch.mean(torch.sum(torch.tensor(back_outs),0),0))

            print("Priority Grads")
            print(torch.mean(torch.sum(torch.tensor(nn_back_ins),0),0))

    def test_epoch(self, epoch):
        jobs = self.create_batch_dq(
            self.model_config['opt']['test_batch'],
            self.model_config['env']['test_seed'])
        workers = min(48, cpu_count(), len(jobs))
        description = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')} - {self.experiment_name}"

        # Loky supports locally defined sampling callbacks (including notebooks).
        # Limit native math libraries as well as PyTorch to one thread per worker.
        with parallel_config(backend='loky', inner_max_num_threads=1):
            results = Parallel(n_jobs=workers, return_as='generator', batch_size=1)(
                delayed(_test_environment)(*job) for job in jobs)
            results = list(tqdm.tqdm(results, total=len(jobs), desc=description))

        test_cost_batch = torch.cat([cost for cost, _ in results])
        test_cost = torch.mean(test_cost_batch)
        test_std = torch.std(test_cost_batch)
        test_queue_len = torch.cat([queues for _, queues in results]).mean(dim=0).tolist()
        self.test_loss.append({'epoch': epoch,
                              'test_loss': float(test_cost),
                              'test_loss_std': float(test_std),
                              'mean_queue_length': test_queue_len})
        print(f"queue lengths: \t{test_queue_len}")
        print(f"test cost: \t{test_cost}")

    def create_batch_dq(self, bs, seed):
        """Prepare independent CPU rollout jobs; environments are built in workers."""
        if bs < 1:
            raise ValueError('test_batch must be at least 1')
        model_config, env_config, policy = _cpu_snapshot(
            (self.model_config, self.env_config, self.policy))
        model_config['env']['device'] = 'cpu'
        return [(model_config, env_config, policy, self.draw_service,
                 self.draw_inter_arrivals, seed + idx) for idx in range(bs)]


def _cpu_snapshot(value):
    """Copy evaluation state without moving or mutating the training policy."""
    value = copy.deepcopy(value)
    seen = set()

    def move(item):
        if isinstance(item, torch.Tensor):
            return item.detach().cpu()
        if id(item) in seen:
            return item
        seen.add(id(item))
        if isinstance(item, dict):
            for key, child in item.items():
                item[key] = move(child)
        elif isinstance(item, list):
            item[:] = [move(child) for child in item]
        elif isinstance(item, tuple):
            return tuple(move(child) for child in item)
        elif isinstance(item, nn.Module):
            item.cpu()
            # Also move non-buffer tensors such as normalization statistics.
            for module in item.modules():
                for key, child in vars(module).items():
                    if not key.startswith('_'):
                        setattr(module, key, move(child))
        elif hasattr(item, '__dict__') and not callable(item):
            for key, child in vars(item).items():
                setattr(item, key, move(child))
        return item

    return move(value)


def _test_environment(model_config, env_config, policy, draw_service,
                      draw_inter_arrivals, seed):
    """Run a complete batch=1 trajectory within one CPU worker."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # A worker may receive several jobs; policy caches must be per environment.
        policy = copy.deepcopy(policy)
        with torch.no_grad():
            dq = DiffDiscreteEventSystem(
                env_config['network'], env_config['mu'], env_config['h'],
                queue_event_options=env_config['queue_event_options'], batch=1,
                temp=model_config['env']['env_temp'], seed=seed, device='cpu',
                draw_service=draw_service, draw_inter_arrivals=draw_inter_arrivals)
            obs, state = dq.reset(seed=seed)
            total_cost = torch.zeros(1, 1)
            time_weight_queue_len = torch.zeros(1, dq.q)
            repeated_h = dq.h.repeat(1, dq.s, 1)
            lex = torch.zeros(1, dq.s, dq.q)

            for step in range(env_config['test_T']):
                queues, time = obs
                repeated_queue = queues.unsqueeze(1).repeat(1, dq.s, 1)
                pr = policy.test_forward(step, queues, time, repeated_queue,
                                         dq.network, dq.mu, repeated_h)
                test_policy = model_config['policy']['test_policy']
                if test_policy in ('sinkhorn', 'linear_assigment'):
                    v, s_bar, q_bar = rt.pad_pool(
                        2 * pr + lex, queues.detach(), network=dq.network,
                        device='cpu', server_pool_size=env_config['server_pool_size'])
                    if test_policy == 'sinkhorn':
                        options = model_config['policy']['sinkhorn']
                        pr = rt.Sinkhorn.apply(
                            -v, s_bar, q_bar, options['num_iter'], options['temp'],
                            options['eps'], options['back_temp'], 'cpu')[:, :dq.s, :dq.q]
                    else:
                        pr = rt.linear_assignment_batch(v, s_bar, q_bar)
                elif test_policy == 'softmax':
                    pr = F.softmax(pr, dim=-1) * dq.network
                    pr = torch.minimum(pr, repeated_queue).clip(min=1e-4)
                    pr /= torch.sum(pr, dim=-1).unsqueeze(-1)

                _, _, _, _, info = dq.step(torch.round(pr))
                obs, state = info['obs'], info['state']
                total_cost += info['cost']
                time_weight_queue_len += info['queues'] * info['event_time']

            return total_cost / state.time, time_weight_queue_len / state.time
    finally:
        # n_jobs=1 runs in the caller; preserve its training thread count.
        torch.set_num_threads(previous_threads)
