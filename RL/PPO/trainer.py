import copy
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import cloudpickle

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F
from typing import Union, List
from stable_baselines3.common.utils import explained_variance
from stable_baselines3 import PPO
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv



def _rollout_cpu_copy(obj):
    """Include unregistered policy constants and nested environment tensors."""
    def to_cpu(value):
        if isinstance(value, th.Tensor):
            return value.detach().cpu()
        if isinstance(value, list):
            return [to_cpu(item) for item in value]
        if isinstance(value, tuple):
            items = [to_cpu(item) for item in value]
            return type(value)(*items) if hasattr(value, "_fields") else tuple(items)
        if isinstance(value, dict):
            return {key: to_cpu(item) for key, item in value.items()}
        return value

    obj = copy.deepcopy(obj)
    if isinstance(obj, th.nn.Module):
        obj.to("cpu")
        objects = obj.modules()
    else:
        objects = [obj]
    for item in objects:
        for name, value in vars(item).items():
            if isinstance(item, th.nn.Module) and name in {"_parameters", "_buffers", "_modules"}:
                continue
            if name == "device":
                setattr(item, name, th.device("cpu"))
            else:
                setattr(item, name, to_cpu(value))
    return obj


def _init_rollout_worker(policy_bytes):
    global _rollout_policy
    th.set_num_threads(1)
    th.set_num_interop_threads(1)
    _rollout_policy = cloudpickle.loads(policy_bytes)
    _rollout_policy.set_training_mode(False)
    _rollout_policy.printing = False


def _collect_actor_rollout(env_bytes, n_steps, use_sde, sde_sample_freq, seed,
                           initial_obs, initial_episode_start):
    """Collect a complete trajectory without per-step IPC or callbacks."""
    th.manual_seed(seed)
    np.random.seed(seed)
    dq = cloudpickle.loads(env_bytes)
    policy = _rollout_policy
    # Preserve the original reset side effect while using the carried observation.
    dq.reset(seed=dq.seed)
    dq.reset_env_seed()
    obs = np.asarray(initial_obs).reshape(dq.observation_space.shape)
    episode_start = bool(initial_episode_start)
    trajectory = []
    if use_sde:
        policy.reset_noise(1)
    with th.no_grad():
        for step in range(n_steps):
            if use_sde and sde_sample_freq > 0 and step % sde_sample_freq == 0:
                policy.reset_noise(1)
            actions, values, log_probs = policy(th.as_tensor(obs).unsqueeze(0))
            actions = actions.cpu().numpy()
            clipped_actions = actions
            # Use the policy's Gymnasium space; raw environments use legacy Gym.
            if isinstance(policy.action_space, spaces.Box):
                if policy.squash_output:
                    clipped_actions = policy.unscale_action(actions)
                else:
                    clipped_actions = np.clip(actions, policy.action_space.low, policy.action_space.high)
            new_obs, reward, terminated, truncated, info = dq.step(clipped_actions[0])
            new_obs = np.asarray(new_obs).reshape(dq.observation_space.shape)
            reward = float(np.asarray(reward).reshape(-1)[0])
            # The original collector ignores termination and truncation flags.
            done = False
            trajectory.append((obs.copy(), actions[0].copy(), reward, episode_start,
                               values.cpu().numpy().reshape(-1)[0],
                               log_probs.cpu().numpy().reshape(-1)[0]))
            obs, episode_start = new_obs, done
        last_value = policy.predict_values(th.as_tensor(obs).unsqueeze(0)).item()
    # NumPy results avoid shared tensor storage outliving a worker process.
    return trajectory, obs, episode_start, last_value, cloudpickle.dumps(vars(dq))


def _advance_rollout_callback(callback, skipped_steps):
    callback.n_calls += skipped_steps
    # SB3 wraps user callbacks in CallbackList (e.g. with a progress bar).
    if isinstance(callback, CallbackList):
        for child in callback.callbacks:
            _advance_rollout_callback(child, skipped_steps)


def cosine_lr_schedule(initial_lr, min_lr=1e-5, progress_remaining=1.0, warmup_proportion=0.03):
    """
    Computes the cosine decay of the learning rate with a linear warmup period at the beginning.
    
    :param initial_lr: The initial learning rate.
    :param min_lr: The minimum learning rate.
    :param progress_remaining: The progress remaining (from 1 to 0).
    :param warmup_proportion: The proportion of the total training time to be used for linear warmup.
    :return: The adjusted learning rate based on the cosine schedule with warmup.
    """
    # Ensure progress_remaining is between 0 and 1
    progress_remaining = np.clip(progress_remaining, 0, 1)

    if progress_remaining > (1 - warmup_proportion):
        # Warmup phase: linearly increase LR
        warmup_progress = (1 - progress_remaining) / warmup_proportion
        new_lr = min_lr + (initial_lr - min_lr) * warmup_progress
    else:
        # Adjusted progress considering warmup phase
        adjusted_progress = (progress_remaining - (1 - warmup_proportion)) / (1 - warmup_proportion)
        
        # Cosine decay phase
        cos_decay = 0.5 * (1 + np.cos(np.pi * adjusted_progress))
        decayed = (1 - min_lr / initial_lr) * cos_decay + min_lr / initial_lr
        new_lr = initial_lr * decayed

    return new_lr

class CustomPPOTrainer(PPO):
    def __init__(self, *args, normalize_value, lr_policy, lr_value, min_lr_policy, min_lr_value, amp_value, rescale_v, num_epochs, actors, raw_env = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_value = normalize_value
        self.lr_policy = lr_policy
        self.lr_value = lr_value
        self.min_lr_policy = min_lr_policy
        self.min_lr_value = min_lr_value
        self.amp_value = amp_value
        self.rescale_v = rescale_v
        self.num_epochs = num_epochs
        self.raw_env = raw_env
        self.actors = actors
        self.training_iteration = 0
        self.policy_train_iter = 0
        self.value_train_iter = 0

        # check is self.policy has log_std:
        if hasattr(self.policy, "log_std"):
            self.optimizer_policy = th.optim.Adam([
                {'params': self.policy.log_std},
                {'params': self.policy.features_extractor.parameters()},
                {'params': self.policy.pi_features_extractor.parameters()},
                {'params': self.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.policy.action_net.parameters()}
            ], lr=self.lr_policy)

        else:
            self.optimizer_policy = th.optim.Adam([
                {'params': self.policy.features_extractor.parameters()},
                {'params': self.policy.pi_features_extractor.parameters()},
                {'params': self.policy.mlp_extractor.policy_net.parameters()},
                {'params': self.policy.action_net.parameters()}
            ], lr=self.lr_policy)

        self.optimizer_value = th.optim.Adam([
            {'params': self.policy.vf_features_extractor.parameters()},
            {'params': self.policy.mlp_extractor.value_net.parameters()},
            {'params': self.policy.value_net.parameters()}
        ], lr=self.lr_value)

        ### print the architecture of the both networks:
        print('policy architecture:')
        print(self.policy)
        print('value architecture:')
        print(self.policy.value_net)

        #### Check if there's missing parameters: #####
        all_parameters = set(self.policy.parameters())

        # Collect parameters managed by each optimizer
        optimizer_policy_params = set(param for group in self.optimizer_policy.param_groups for param in group['params'])
        optimizer_value_params = set(param for group in self.optimizer_value.param_groups for param in group['params'])

        # Check for any missing parameters
        missing_params = all_parameters - (optimizer_policy_params | optimizer_value_params)
        assert len(missing_params) == 0, "Some parameters are not being optimized."


    def _update_learning_rate(self, policy_optimizer: th.optim.Optimizer, value_optimizer: th.optim.Optimizer) -> None:
        """
        Update the learning rates for policy and value optimizers separately
        using their respective cosine learning rate schedules based on the current progress remaining.
        """
        # Update policy optimizer learning rate
        lr_policy = cosine_lr_schedule(self.lr_policy, self.min_lr_policy, self._current_progress_remaining)
        for param_group in policy_optimizer.param_groups:
            param_group['lr'] = lr_policy
        self.logger.record("train/learning_rate/policy", lr_policy)

        # Update value optimizer learning rate
        lr_value = cosine_lr_schedule(self.lr_value, self.min_lr_value, self._current_progress_remaining)
        for param_group in value_optimizer.param_groups:
            param_group['lr'] = lr_value
        self.logger.record("train/learning_rate/value", lr_value)

    def train(self) -> None:
            """
            Update policy using the currently gathered rollout buffer.
            """
            # Switch to train mode (this affects batch norm / dropout)
            self.policy.set_training_mode(True)
            print('-----------------------------------------now training-----------------------------------------')
            training_time_start = time.time()
            self.training_iteration += 1    

            # Update optimizer learning rate
            # self._update_learning_rate(self.policy.optimizer)
            self._update_learning_rate(self.optimizer_policy, self.optimizer_value)
            # Compute current clip range
            clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
            clipping_alpha = 1.0 - self.training_iteration / self.num_epochs
            clip_range = max(0.01, clipping_alpha * clip_range)
            # print(f'clip_range: {clip_range}')
            # print(f'num_epochs: {self.num_epochs}')
            # print(f'current_training_iteration: {self.training_iteration}')
            # print(f'clipping_alpha: {clipping_alpha}')
            # print(f'clip_range: {clip_range}')
            
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True
            # train for n_epochs epochs
            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                
                # Get all rollout data for policy training
                for rollout_data in self.rollout_buffer.get():
                
                    actions = rollout_data.actions
                    if isinstance(self.action_space, spaces.Discrete):
                        # Convert discrete action from float to long
                        actions = rollout_data.actions.long().flatten()

                    # Re-sample the noise matrix because the log_std has changed
                    if self.use_sde:
                        self.policy.reset_noise(self.batch_size)

                    log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                                        
                    advantages = rollout_data.advantages
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    policy_loss = policy_loss + self.ent_coef * entropy_loss

                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                        break

                    # Update Policy network
                    self.optimizer_policy.zero_grad()
                    policy_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer_policy.step()

                    self.policy_train_iter += 1

                # Do a complete pass on the rollout buffer for value training
                for rollout_data in self.rollout_buffer.get(self.batch_size):

                    values = self.policy.evaluate_values(rollout_data.observations)
                    values = values.flatten()
                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                        )

                    value_loss = F.mse_loss(rollout_data.returns, values_pred)                    
                    value_loss = self.vf_coef * value_loss
                    value_losses.append(value_loss.item())

                    # Update value network
                    self.optimizer_value.zero_grad()
                    value_loss.backward()
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer_value.step()

                    self.value_train_iter += 1


                if not continue_training:
                    break

            explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

            training_time_end = time.time() 
            print(f'training_time: {training_time_end - training_time_start}')

            # Logs
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))
            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))

            # log lr rate for both
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/policy_updates", self.policy_train_iter)
            self.logger.record("train/value_updates", self.value_train_iter)
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)

    def construct_batch(self):
        lex_batch = []
        obs_batch = []
        state_batch = []
        total_cost_batch = []
        time_weight_queue_len_batch = []


        for dq_idx in range(self.actors):

            dq = self.raw_env[dq_idx]
            lex = th.zeros(dq.batch, dq.s, dq.q)
            obs, state = dq.reset(seed = dq.seed)
            total_cost = th.tensor([[0.]])
            time_weight_queue_len = th.tensor([[0.]])

            lex_batch.append(lex)
            obs_batch.append(obs)
            state_batch.append(state)
            total_cost_batch.append(total_cost)
            time_weight_queue_len_batch.append(time_weight_queue_len)

        
        return lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch

    # def collect_rollouts(
    #     self,
    #     env: VecEnv,
    #     callback: BaseCallback,
    #     rollout_buffer: RolloutBuffer,
    #     n_rollout_steps: int,
    # ) -> bool:
    #     """
    #     Collect experiences using the current policy and fill a ``RolloutBuffer``.
    #     The term rollout here refers to the model-free notion and should not
    #     be used with the concept of rollout used in model-based RL or planning.

    #     :param env: The training environment
    #     :param callback: Callback that will be called at each step
    #         (and at the beginning and end of the rollout)
    #     :param rollout_buffer: Buffer to fill with rollouts
    #     :param n_rollout_steps: Number of experiences to collect per environment
    #     :return: True if function returned with at least `n_rollout_steps`
    #         collected, False if callback terminated rollout prematurely.
    #     """
    #     assert self._last_obs is not None, "No previous observation was provided"
    #     # Switch to eval mode (this affects batch norm / dropout)
    #     self.policy.set_training_mode(False)

    #     n_steps = 0
    #     rollout_buffer.reset()
    #     # Sample new weights for the state dependent exploration
    #     if self.use_sde:
    #         self.policy.reset_noise(env.num_envs)

    #     callback.on_rollout_start()
    #     # collect_roll_out_time_start = time.time()
    #     lex_batch, obs_batch, state_batch, total_cost_batch, time_weight_queue_len_batch = self.construct_batch()
    #     test_dq_batch = self.env

    #     while n_steps < n_rollout_steps:
    #         # start_time = time.time()
    #         if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
    #             # Sample a new noise matrix
    #             self.policy.reset_noise(env.num_envs)

    #         with th.no_grad():
    #             # Convert to pytorch tensor or to TensorDict
    #             self.policy.printing = False
    #             obs_tensor = obs_as_tensor(self._last_obs, self.device)
    #             actions, values, log_probs = self.policy(obs_tensor)
    #             self.policy.printing = False
    #         actions = actions.cpu().numpy()


    #         # Rescale and perform action
    #         clipped_actions = actions

    #         if isinstance(self.action_space, spaces.Box):
    #             if self.policy.squash_output:
    #                 # Unscale the actions to match env bounds
    #                 # if they were previously squashed (scaled in [-1, 1])
    #                 clipped_actions = self.policy.unscale_action(clipped_actions)
    #             else:
    #                 # Otherwise, clip the actions to avoid out of bound error
    #                 # as we are sampling from an unbounded Gaussian distribution
    #                 clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

    #         # Rescale and perform action
    #         new_obs, rewards, dones, _ = env.step(clipped_actions)
    #         new_obs = new_obs.squeeze() 
    #         # print(f'new_obs shape: {new_obs.shape}')
    #         # print(f'rewards shape: {rewards.shape}')     
    #         # print(f'dones: {dones}')  
    #         # print(f'n step: {n_steps} step_time: {dones}')


    #         # print(f'collect_time: {collect_time_end - start_time}')

    #         self.num_timesteps += env.num_envs

    #         # Give access to local variables
    #         callback.update_locals(locals())
    #         if not callback.on_step():
    #             return False

    #         # self._update_info_buffer(infos)
    #         n_steps += 1

    #         if isinstance(self.action_space, spaces.Discrete):
    #             # Reshape in case of discrete action
    #             actions = actions.reshape(-1, 1)


    #         # for idx, done in enumerate(dones):
    #         #     if (
    #         #         done
    #         #         and infos[idx].get("terminal_observation") is not None
    #         #         and infos[idx].get("TimeLimit.truncated", False)
    #         #     ):
    #         #         terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
    #         #         with th.no_grad():
    #         #             terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
    #         #         rewards[idx] += self.gamma * terminal_value
            
    #         # print(f'enumerate_time: {enumerate_time_end - collect_time_end}')
    #         rollout_buffer.add(
    #                 self._last_obs,  # type: ignore[arg-type]
    #                 actions,
    #                 rewards,
    #                 self._last_episode_starts,  # type: ignore[arg-type]
    #                 values,
    #                 log_probs,
    #             )
    #         self._last_obs = new_obs  # type: ignore[assignment]
    #         self._last_episode_starts = dones
    #         # end_time = time.time()

    #         # print(f'collect_per_roll_out_time: {end_time - start_time}')

    #     with th.no_grad():
    #         # Compute value for the last timestep
            
    #         values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        

    #     retunrs_mean, returns_std= rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
    #     #print("update_rollout_stats for policy")
    #     #print(f'returns_mean: {retunrs_mean} returns_std: {returns_std}')
    #     self.policy.update_rollout_stats(retunrs_mean, returns_std)

    #     callback.update_locals(locals())

    #     callback.on_rollout_end()
    #     # collect_roll_out_time_end = time.time()

    #     # print(f'collect_roll_out_time: {collect_roll_out_time_end - collect_roll_out_time_start}')

    #     return True
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Collect actor trajectories on 48 local CPU processes.

        Each actor collects ``n_rollout_steps`` using a frozen CPU policy copy.
        Actors beyond 48 are queued. The callback runs once after collection,
        with its step counter advanced by the full rollout length.
        """
        if n_rollout_steps <= 0:
            raise ValueError("n_rollout_steps must be positive")
        if self.raw_env is None or len(self.raw_env) != self.actors or self.actors < 1:
            raise ValueError("raw_env must contain one environment per actor")
        if env.num_envs != self.actors or rollout_buffer.n_envs != self.actors:
            raise ValueError("Environment and rollout buffer counts must match actors")

        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)
        rollout_buffer.reset()
        callback.on_rollout_start()
        policy_bytes = cloudpickle.dumps(_rollout_cpu_copy(self.policy))
        seeds = np.random.randint(0, 2**32 - 1, size=self.actors, dtype=np.uint32)
        # Spawn avoids inheriting CUDA state and works on Windows as well as Linux.
        with ProcessPoolExecutor(
            max_workers=48,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_rollout_worker,
            initargs=(policy_bytes,),
        ) as executor:
            futures = [
                executor.submit(
                    _collect_actor_rollout,
                    cloudpickle.dumps(_rollout_cpu_copy(dq)),
                    n_rollout_steps, self.use_sde, self.sde_sample_freq,
                    int(seeds[idx]), self._last_obs[idx], self._last_episode_starts[idx],
                )
                for idx, dq in enumerate(self.raw_env)
            ]
            results = [future.result() for future in futures]

        # Restore worker mutations in place so existing raw_env references see them.
        for dq, result in zip(self.raw_env, results):
            dq.__dict__.update(cloudpickle.loads(result[4]))

        # Preserve the buffer's [step, actor] ordering regardless of finish order.
        for n_steps in range(n_rollout_steps):
            observations, actions, rewards, episode_starts, values, log_probs = (
                np.stack(items) for items in zip(*(result[0][n_steps] for result in results))
            )
            rollout_buffer.add(
                observations, actions, rewards, episode_starts,
                th.as_tensor(values), th.as_tensor(log_probs),
            )
        self._last_obs = np.stack([result[1] for result in results])
        dones = np.asarray([result[2] for result in results], dtype=bool)
        self._last_episode_starts = dones
        self.num_timesteps += n_rollout_steps * self.actors
        values = th.as_tensor([result[3] for result in results])
        returns_mean, returns_std = rollout_buffer.compute_returns_and_advantage(
            last_values=values, dones=dones,
        )
        self.policy.update_rollout_stats(returns_mean, returns_std)

        n_steps = n_rollout_steps
        _advance_rollout_callback(callback, n_rollout_steps - 1)
        callback.update_locals(locals())
        if not callback.on_step():
            return False
        callback.on_rollout_end()
        return True
