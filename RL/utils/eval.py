import sys
import copy
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import cloudpickle
sys.path.append('.../')
from stable_baselines3 import PPO
import numpy as np
import torch
from torch.nn import functional as F
from typing import NamedTuple
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.data import Dataset, DataLoader



class Obs(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor

class EnvState(NamedTuple):
    queues: torch.Tensor
    time: torch.Tensor
    service_times: torch.Tensor
    arrival_times: torch.Tensor


def _cpu_copy(obj):
    """Copy tensor attributes too: policy constants are not all registered buffers."""
    obj = copy.deepcopy(obj)
    if isinstance(obj, torch.nn.Module):
        obj.to('cpu')
        modules = obj.modules()
    else:
        modules = [obj]
    for module in modules:
        for name, value in vars(module).items():
            if isinstance(value, torch.Tensor):
                setattr(module, name, value.detach().cpu())
    return obj


def _init_eval_worker(policy_bytes):
    global _eval_policy
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _eval_policy = cloudpickle.loads(policy_bytes)
    _eval_policy.set_training_mode(False)


def _eval_trajectory(env_bytes, eval_t):
    # Cloudpickle supports the local arrival/service functions stored in each env.
    dq = cloudpickle.loads(env_bytes)
    obs, state = dq.reset(seed=dq.seed)
    total_cost = torch.tensor([[0.]])
    time_weight_queue_len = torch.tensor([[0.]])
    with torch.no_grad():
        for _ in range(eval_t):
            # reset() returns a queue array; step() returns Obs(queues, time).
            # Select queues before conversion so the Obs tuple is never cast.
            batch_queue = torch.as_tensor(obs[0]).reshape(dq.batch, -1)
            raw_actions, _ = _eval_policy.predict(batch_queue)
            action = torch.as_tensor(raw_actions, dtype=torch.float32)
            _, _, _, _, info = dq.step(action[0])
            obs, state = info['obs'], info['state']
            total_cost = total_cost + info['cost']
            time_weight_queue_len = (
                time_weight_queue_len + info['queues'] * info['event_time']
            )
    # Return compact NumPy results rather than sharing worker-owned tensor storage.
    return (total_cost.numpy(), time_weight_queue_len.numpy(), state.time.numpy())


class BCD(Dataset):
    def __init__(self, num_samples, network):
        self.num_samples = num_samples
        self.network = network
        self.s = self.network.shape[0]
        self.q = self.network.shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate random input data
        input_data = np.random.randint(0, 101, self.q)
        obs = torch.tensor(input_data)
        action_probs = F.softmax(torch.tensor(input_data).float(), dim=-1)
        action_probs = action_probs * self.network
        action_probs = torch.minimum(action_probs, obs.unsqueeze(0).repeat(1, self.s, 1))
        zero_mask = torch.all(action_probs == 0, dim=2).reshape(-1, self.s, 1).repeat(1, 1, self.q)
        action_probs = action_probs + zero_mask * self.network
        action_probs = action_probs / torch.sum(action_probs, dim=-1).reshape(-1, self.s, 1)
        output_data = action_probs
        input_tensor = torch.tensor(input_data, dtype=torch.float32).squeeze()
        output_tensor = torch.tensor(output_data, dtype=torch.float32).squeeze()
        
        return input_tensor, output_tensor


class parallel_eval(BaseCallback):
    def __init__(self, model, eval_env, eval_freq, eval_t, test_policy, test_seed, init_test_queues, test_batch, device, num_pool, time_f, policy_name, per_iter_normal_obs, env_config_name, bc, randomize = True, 
                 verbose=1):
        super(parallel_eval, self).__init__(verbose)
        self.model = model
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.eval_t = eval_t
        print('eval_t', eval_t)
        self.test_policy = test_policy
        self.test_seed = test_seed
        self.init_test_queues = init_test_queues
        self.test_batch = test_batch
        self.device = device
        self.num_pool = num_pool
        self.randomize = randomize
        self.time_f = time_f
        self.policy_name = policy_name
        self.test_costs = []  
        self.final_costs = []
        self.per_iter_normal_obs = per_iter_normal_obs
        self.env_config_name = env_config_name
        self.bc = bc
        print(f'eval env config name: {self.env_config_name}')
        self.iter = 0

        self.lex_batch = []
        self.obs_batch = []
        self.state_batch = []
        self.total_cost_batch = []
        self.time_weight_queue_len_batch = []




    
    def behavior_cloning(self):

        print(f'---------------------behavior_cloning---------------------')
        if hasattr(self.model.policy, "log_std"):
            self.optimizer_policy = torch.optim.Adam([
                {'params': self.model.policy.log_std},
                {'params': self.model.policy.features_extractor.parameters()},
                {'params': self.model.policy.pi_features_extractor.parameters()},
                {'params': self.model.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.model.policy.action_net.parameters()}
            ], lr=3e-4)

        else:
            self.model.optimizer_policy = torch.optim.Adam([
                {'params': self.model.policy.features_extractor.parameters()},
                {'params': self.model.policy.pi_features_extractor.parameters()},
                {'params': self.model.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.model.policy.action_net.parameters()}
            ], lr=3e-4)

        # print(f'network: {self.eval_env[0].network}, shape: {self.eval_env[0].network[0].shape}')
        BCD_dataset = BCD(num_samples = 100000, network = self.eval_env[0].network[0])
        BCD_loader = DataLoader(BCD_dataset, batch_size = self.test_batch, shuffle = True)

        for i, (obs, target) in enumerate(BCD_loader):
            self.optimizer_policy.zero_grad()
            action, action_probs = self.model.policy.get_prob_act(obs)
            loss = F.mse_loss(action_probs, target)
            loss.backward()
            self.optimizer_policy.step()

    def pre_train_eval(self):
        print('pre_train_eval')
        if self.bc:
            self.behavior_cloning()
        # self.behavior_cloning()
        q_mean, q_std, t_mean, t_max, t_min, t_std, total_q_mean, total_q_std = self.eval()

        self.model.policy.update_mean_std(mean_queue_length = q_mean, std_queue_length = q_std)
        print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
        print(f"std_queue_length: {self.model.policy.std_queue_length}")

        return True

    def _on_step(self) -> bool:
        if self.per_iter_normal_obs:
            if (self.n_calls) % self.eval_freq == 0:
                q_mean, q_std, t_mean, t_max, t_min, t_std, total_q_mean, total_q_std = self.eval()

                self.model.policy.update_mean_std(mean_queue_length = q_mean, std_queue_length = q_std)
                print(f"mean_queue_length: {self.model.policy.mean_queue_length}")
                print(f"std_queue_length: {self.model.policy.std_queue_length}")

                self.test_costs.append([
                                    self.n_calls // self.eval_freq,
                                    q_mean.item(),
                                    q_std.item(),
                                    total_q_mean.item(),
                                    total_q_std.item()
                                ])
        else:
            if (self.n_calls) % self.eval_freq == 0:
                q_mean, q_std, t_mean, t_max, t_min, t_std, total_q_mean, total_q_std = self.eval()
                print(f"mean_queue_length: {q_mean.item()}")
                print(f"std_queue_length: {q_std.item()}")
                self.test_costs.append([
                                    self.n_calls // self.eval_freq,
                                    q_mean.item(),
                                    q_std.item(),
                                    total_q_mean.item(),
                                    total_q_std.item()
                                ])

        return True
    

    def eval(self):
        self.iter += 1
        print(f'iter: {self.iter}')

        if not self.eval_env:
            raise ValueError('Evaluation requires at least one environment')
        worker_count = min(48, os.cpu_count() or 1, len(self.eval_env))
        policy_bytes = cloudpickle.dumps(_cpu_copy(self.model.policy))

        def serialize_env(dq):
            cpu_env = _cpu_copy(dq)
            cpu_env.device = torch.device('cpu')
            return cloudpickle.dumps(cpu_env)

        # Spawn works on Windows and avoids inheriting training/CUDA thread state.
        with ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_eval_worker,
            initargs=(policy_bytes,),
        ) as executor:
            futures = [executor.submit(_eval_trajectory, serialize_env(dq), self.eval_t)
                       for dq in self.eval_env]
            results = [future.result() for future in futures]

        # Keep input order and the original metric definitions.
        total_cost_batch = [torch.as_tensor(result[0]) for result in results]
        time_weight_queue_len_batch = [torch.as_tensor(result[1]) for result in results]
        time_batch = [torch.as_tensor(result[2]) for result in results]
        test_dq_batch = self.eval_env

        # Test cost metrics
        # pdb.set_trace()
        test_cost_batch = [total_cost_batch[test_dq_idx] / time_batch[test_dq_idx] for test_dq_idx in range(len(test_dq_batch))]
        test_cost = torch.mean(torch.concat(test_cost_batch))
        test_std = torch.std(torch.concat(test_cost_batch))
        trajectory_queue_means = torch.concat([
            queue_integrals / elapsed
            for queue_integrals, elapsed in zip(time_weight_queue_len_batch, time_batch)
        ])
        test_queue_len = torch.mean(trajectory_queue_means, dim=0)
        # Sum queues within each trajectory before measuring spread across trajectories.
        trajectory_total_queues = trajectory_queue_means.sum(dim=-1)
        total_q_mean = trajectory_total_queues.mean()
        total_q_std = (trajectory_total_queues.std() if trajectory_total_queues.numel() > 1
                       else torch.zeros_like(total_q_mean))
        test_queue_len = [float(_item) for _item in test_queue_len.to('cpu').detach().numpy().tolist()]
        
        print(f"queue lengths: \t{test_queue_len}")
        print(f"total queue length mean: \t{total_q_mean}")
        print(f"total queue length std: \t{total_q_std}")
        print(f"test cost: \t{test_cost}")
        print(f"test cost std: \t{test_std}")


        test_queue_len = torch.tensor(test_queue_len)

        q_mean = torch.mean(test_queue_len)
        q_std = torch.std(test_queue_len)
        elapsed_times = torch.cat([elapsed.reshape(-1) for elapsed in time_batch])
        t_mean = torch.mean(elapsed_times)
        t_max = torch.max(elapsed_times)
        t_min = torch.min(elapsed_times)
        # Use sample std across trajectories; a single trajectory has no spread.
        t_std = (torch.std(elapsed_times) if elapsed_times.numel() > 1
                 else torch.zeros_like(t_mean))
        
        return q_mean, q_std, t_mean, t_max, t_min, t_std, total_q_mean, total_q_std

    def construct_batch(self):
        lex_batch = []
        obs_batch = []
        state_batch = []
        total_cost_batch = []
        time_weight_queue_len_batch = []


        for dq_idx in range(self.test_batch):

            dq = self.eval_env[dq_idx]
            lex = torch.zeros(dq.batch, dq.s, dq.q)
            obs, state = dq.reset(seed = dq.seed)
            obs = torch.tensor(obs).to(self.device)
            total_cost = torch.tensor([[0.]])
            time_weight_queue_len = torch.tensor([[0.]])

            lex_batch.append(lex)
            obs_batch.append(obs)
            state_batch.append(state)
            total_cost_batch.append(total_cost)
            time_weight_queue_len_batch.append(time_weight_queue_len)

        
        return lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch
