from typing import List

import gymnasium
import numpy as np
import torch
from sympy import false
from torch import nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util import mathutil

"""
Our implementation of TD-MPC2
See Paper: https://arxiv.org/abs/2310.16828
See GitHub for original implementation: https://github.com/nicklashansen/tdmpc2/tree/main

Author: Daniel Flat
"""


# TODO
class EncoderNet(nn.Module):
    def __init__(self, state_size: int, latent_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),  # add SimNorm
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent_state = self.encoder_net(state)
        return latent_state


# TODO
class DynamicsNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            nn.Linear(latent_size + action_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent_state = self.encoder_net(state)
        return latent_state


# TODO
class RewardNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            nn.Linear(latent_size + action_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, latent_size),
            nn.ReLU(),
            nn.Linear(latent_size, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        latent_state = self.encoder_net(state)
        return latent_state


# TODO
class ActorNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(latent_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2 * action_size),  # mean, std
            nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor_net(state)
        return action


# TODO
class CriticNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.critic_net = nn.Sequential(
            nn.Linear(latent_size + action_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1))

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.hstack([state, action])
        q_value = self.critic_net(input)
        return q_value


#
# class WorldModel(nn.Module):
#     def __init__(self, observation_size: int, action_size: int, hidden_size: int, device: torch.device):
#         super().__init__()
#
#         self._encoder = Encoder()


# df: I am experimenting here to try out if the Agent could also be a nn.Module
# Can be maybe converted then to the parent class
class TDMPC2Agent(Agent, nn.Module):
    def __init__(self, state_space: gymnasium.spaces.box.Box, action_space: gymnasium.spaces.box.Box,
                 agent_settings: dict, td_mpc2_settings: dict,
                 device: torch.device):
        super().__init__(agent_settings = agent_settings, device = device)
        nn.Module.__init__(self)

        self.isEval = None

        self.state_space = state_space
        self.action_space = action_space
        state_size = state_space.shape[0]
        self.action_size = self.get_num_actions(action_space)
        self.min_action = action_space.low[:self.action_size]
        self.max_action = action_space.high[:self.action_size]

        # self.noise = initNoise(action_shape = (self.action_size,), noise_settings = td_mpc2_settings["NOISE"],
        #                        device = self.device)
        # self.noise_factor = td_mpc2_settings["NOISE"]["NOISE_FACTOR"]

        self.horizon = td_mpc2_settings["HORIZON"]
        self.mmpi_iterations = td_mpc2_settings["MMPI_ITERATIONS"]
        self.num_trajectories = td_mpc2_settings["NUM_TRAJECTORIES"]
        self.num_samples = td_mpc2_settings["NUM_SAMPLES"]
        self.num_elites = td_mpc2_settings["NUM_ELITES"]
        self.min_std = td_mpc2_settings["MIN_STD"]
        self.max_std = td_mpc2_settings["MAX_STD"]
        self.temperature = td_mpc2_settings["TEMPERATURE"]
        self.latent_size = td_mpc2_settings["LATENT_SIZE"]
        self.log_std_min = -10
        self.log_std_max = 2
        self.log_std_dif = self.log_std_max - self.log_std_min
        self.entropy_coef = 1e-4
        self.rho = 0.5
        self.enc_lr_scale = 0.3
        self.grad_clip_norm = 20
        self.lr = 3e-4
        self.consistency_coef = 20
        self.reward_coef = 0.1
        self.value_coef = 0.1

        self.encoder_net = EncoderNet(state_size = state_size, latent_size = self.latent_size).to(self.device)
        self.dynamics_net = DynamicsNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)
        self.reward_net = RewardNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)

        self.policy_net = ActorNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)
        self.q1_net = CriticNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)
        self.q2_net = CriticNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)

        # Target nets
        self.q1_target_net = CriticNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)
        self.q2_target_net = CriticNet(latent_size = self.latent_size, action_size = self.action_size).to(self.device)
        self.q1_target_net.load_state_dict(self.q1_net.state_dict())
        self.q1_target_net.eval()
        self.q2_target_net.load_state_dict(self.q2_net.state_dict())
        self.q2_target_net.eval()

        self.optim = torch.optim.Adam([
            {'params': self.encoder_net.parameters(), 'lr': self.lr * self.enc_lr_scale},
            {'params': self.dynamics_net.parameters()},
            {'params': self.reward_net.parameters()},
            {'params': self.q1_net.parameters()},
            {'params': self.q2_net.parameters()}], lr = self.lr)

        self.policy_optim = torch.optim.Adam(self.policy_net.parameters(), lr = self.lr)

        # self.world_model = WorldModel()

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"TDMPC2Agent"

    @torch.no_grad()
    def act(self, state, t0 = False) -> np.ndarray:
        """
        Select an action by planning in the latent space of the world model.

        Args:
            state (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.

        """
        proposed_action = self._plan(state, t0 = t0, is_training = not self.isEval)
        return proposed_action
        #     return self.plan(state, t0 = t0, eval_mode = eval_mode, task = task).cpu()
        # z = self.model.encode(state, task)
        # action, info = self.model.pi(z, task)
        # if eval_mode:
        #     action = info["mean"]
        # return action[0].cpu()

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        losses = []

        for i in range(1, self.opt_iter + 1):
            # Step 01: We take the whole sequence from the batch
            state, action, reward, next_state, done, _ = memory.sample(batch_size = self.batch_size, randomly = False)

            # Step 02: We randomly subsample a horizon-long trajectory for updating
            starting_point = torch.randint(0, state.shape[0] - self.horizon, (1,), device = self.device)
            state = state[starting_point:starting_point + self.horizon + 1]
            action = action[starting_point:starting_point + self.horizon + 1]
            reward = reward[starting_point:starting_point + self.horizon + 1]
            next_state = next_state[starting_point:starting_point + self.horizon + 1]
            done = done[starting_point:starting_point + self.horizon + 1]

            # if starting_point > 195:
            #     print(starting_point)

            # Step 03: Update the agent.
            loss = self._update(state, action, reward, next_state, done)

            # Step 04: Keep track of the loss
            losses.append(loss)

        # Step 05: After some time, update the agent
        # NOTE: HERE, we the update frequency is w.t.r. the total number of episodes
        if episode_i % self.target_net_update_freq == 0:
            self.updateTargetNet(soft_update = False, source = self.q1_net,
                                 target = self.q1_target_net)
            self.updateTargetNet(soft_update = False, source = self.q2_net,
                                 target = self.q2_target_net)

        # at the end, we have to clear the memory again
        memory.clear()

        return losses

    def setMode(self, eval = False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.encoder_net.eval()
            self.policy_net.eval()
            self.q1_net.eval()
            self.q2_net.eval()
        else:
            self.encoder_net.train()
            self.policy_net.train()
            self.q1_net.train()
            self.q2_net.train()

    def import_checkpoint(self, checkpoint: dict) -> None:
        self.encoder_net.critic.load_state_dict(checkpoint["encoder_net"])
        self.policy_net.actor.load_state_dict(checkpoint["policy_net"])
        self.dynamics_net.actor.load_state_dict(checkpoint["dynamics_net"])
        self.reward_net.actor.load_state_dict(checkpoint["reward_net"])
        self.q1_net.load_state_dict(checkpoint["q1_net"])
        self.q2_net.load_state_dict(checkpoint["q2_net"])
        self.q1_target_net.load_state_dict(checkpoint["q1_target_net"])
        self.q2_target_net.load_state_dict(checkpoint["q2_target_net"])

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "encoder_net": self.encoder_net.state_dict(),
            "policy_net": self.policy_net.state_dict(),
            "dynamics_net": self.dynamics_net.state_dict(),
            "reward_net": self.reward_net.state_dict(),
            "q1_net": self.q1_net.state_dict(),
            "q2_net": self.q2_net.state_dict(),
            "q1_target_net": self.q1_target_net.state_dict(),
            "q2_target_net": self.q2_target_net.state_dict(),
        }
        return checkpoint

    # ----------------------- Model Functions -----------------------

    def _min_q_value(self, state: torch.Tensor, action: torch.Tensor, use_target: bool):
        """
        Computes the minimum q value of a state-action pair.
        """
        if use_target:
            next_q1 = self.q1_target_net(state, action)
            next_q2 = self.q2_target_net(state, action)
        else:
            next_q1 = self.q1_net(state, action)
            next_q2 = self.q2_net(state, action)
        min_q_value = torch.min(next_q1, next_q2)
        return min_q_value

    @torch.no_grad()
    def _estimate_q_of_action_sequence(self, latent_state: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Estimate value of a trajectory starting at latent state and executing a given sequence of actions.
        We base this calculation by a monte carlo estimate on a horizon-long trajectory.
        """

        # Step 01: Let's first predict the state and the discounted reward in the num of `horizon` in the future.
        _G, _discount = 0, 1
        for t in range(self.horizon):
            reward = self._predict_reward(latent_state, action_sequence[t])
            latent_state = self._predict_next_state(latent_state, action_sequence[t])
            _G += _discount * reward
            _discount *= self.discount

        # Step 02: Sample an action based on our policy
        action, _ = self._predict_action(latent_state)

        # Step 03: Finally, we compute the q value
        min_q_value = self._min_q_value(latent_state, action, use_target = False)
        final_q_value = _G + _discount * min_q_value
        return final_q_value

    # TODO
    @torch.no_grad()
    def _plan(self, state: torch.Tensor, t0 = False, is_training = False) -> np.ndarray:
        """
        Plan proposed_action sequence of actions using the learned world model.

        Args:
            state (torch.Tensor): Real state from which to plan in the future.
            t0 (bool): Whether this is the first observation in the episode.
            is_training (bool): Whether to use the mean of the action distribution or not. if true, we sample, if not, we take the mean.

        Returns:
            torch.Tensor: Action to take in the environment at the current timestep..
        """
        # Step 01: Sample trajectories based on our policy
        latent_state = self.encoder_net(state)
        trajectory_actions = torch.empty(self.horizon, self.num_trajectories, self.action_size, device = self.device)
        _latent_state = latent_state.repeat(self.num_trajectories, 1)  # we only need this tmp. variable for step 01
        for t in range(self.horizon):
            # first, we predict an action
            proposed_action, _ = self._predict_action(_latent_state)

            # then we predict the next state in the future
            _latent_state = self._predict_next_state(_latent_state, proposed_action)
            trajectory_actions[t] = proposed_action

        # Initialize state and parameters
        latent_state = latent_state.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_size, device = self.device)
        std = torch.full((self.horizon, self.action_size), fill_value = self.max_std, dtype = torch.float32,
                         device = self.device)
        # if it is not the starting sequence. TODO
        # if not t0:
        #     mean[:-1] = self._prev_mean[1:]
        actions = torch.empty(self.horizon, self.num_samples, self.action_size, device = self.device)
        actions[:, :self.num_trajectories] = trajectory_actions

        # Iterate MPPI
        for _ in range(self.mmpi_iterations):
            # Sample gaussian random variable
            gaussian_sample = torch.randn(self.horizon, self.num_samples - self.num_trajectories, self.action_size,
                                          device = std.device)

            # do the reparameterization trick
            reparameterization_action = mean.unsqueeze(1) + std.unsqueeze(1) * gaussian_sample

            normalized_action = torch.clamp(reparameterization_action, torch.tensor(self.min_action),
                                            torch.tensor(self.max_action))
            actions[:, self.num_trajectories:] = normalized_action

            # Compute elite actions
            value = self._estimate_q_of_action_sequence(latent_state, actions)  # .nan_to_num(0)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim = 0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            # score = torch.nn.functional.softmax(self.temperature * elite_value, dim = 0)

            max_value = elite_value.max(0).values
            score = torch.exp(self.temperature * (elite_value - max_value))
            score = score / score.sum(0)

            mean = (score.unsqueeze(0) * elite_actions).sum(dim = 1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim = 1) / (
                    score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        # Select action
        rand_idx = mathutil.gumbel_softmax_sample(
            score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)

        proposed_action, std = actions[0], std[0]

        # self._prev_mean.copy_(mean)

        # If we are in training mode, we want to explore by adding noise to the action
        if is_training:
            proposed_action = proposed_action + std * torch.randn(self.action_size, device = std.device)

        normalized_action = torch.clamp(proposed_action, torch.tensor(self.min_action),
                                        torch.tensor(self.max_action))

        return normalized_action.numpy()

    # TODO
    def _calculate_policy_loss(self, latent_state_sequence: torch.Tensor) -> torch.Tensor:
        """
        calculates the policy net using a sequence of latent states.

        """
        # self.policy_optim.zero_grad()
        action, info = self._predict_action(latent_state_sequence)
        q_value = self._min_q_value(latent_state_sequence, action, use_target = false)
        # self.scale.update(q_value[0])
        # q_value = self.scale(q_value)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.rho, torch.arange(len(q_value), device = self.device))
        pi_loss = (rho * -(self.entropy_coef * info["scaled_entropy"] + q_value).mean()).mean()
        return pi_loss
        # pi_loss.backward()
        # pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_clip_norm)
        # self.policy_optim.step()
        # return pi_loss.item()

        # info = TensorDict({
        #     "pi_loss": pi_loss,
        #     "pi_grad_norm": pi_grad_norm,
        #     "pi_entropy": info["entropy"],
        #     "pi_scaled_entropy": info["scaled_entropy"],
        #     "pi_scale": self.scale.value,
        # })
        # return info

    @torch.no_grad()
    def _td_target(self, next_latent_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
                   use_target) -> torch.Tensor:
        next_action, _ = self._predict_action(next_latent_state)
        target_q = self._min_q_value(next_latent_state, next_action, use_target)
        td_target = reward + self.discount * (1 - done) * target_q
        return td_target

    def _update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
                done: torch.Tensor) -> float:
        # Step 01: Compute targets
        with torch.no_grad():
            next_latent_state = self.encoder_net(next_state)
            td_targets = self._td_target(next_latent_state, reward, done, use_target = True)

        # Step 02: Initialize tensors for latent states
        # `latent_state_rollout` will store the latent states over the rollout horizon
        latent_state_rollout = torch.empty(self.horizon + 1, self.latent_size, device = self.device)
        consistency_loss = 0

        latent_state = self.encoder_net(state[0])
        latent_state_rollout[0] = latent_state

        # Step 03: Perform latent rollout to predict future latent states
        for t, (_action, _next_state) in enumerate(zip(action.unbind(0), next_latent_state.unbind(0))):
            if t == self.horizon:
                break
            # Predict the next latent state given the current latent state and action
            latent_state = self._predict_next_state(latent_state, _action)
            # Accumulate consistency loss (mean squared error between predicted and actual latent states)
            consistency_loss += (self.rho ** t) * torch.nn.functional.mse_loss(latent_state, _next_state)
            # Store the predicted latent state
            latent_state_rollout[t + 1] = latent_state

        # Step 04: Make predictions for Q-values and rewards based on the latent states and actions
        predicted_q_values = self._min_q_value(latent_state_rollout, action, use_target = False)
        predicted_rewards = self._predict_reward(latent_state_rollout, action)

        # Step 05: Compute losses for rewards and Q-values
        reward_loss, value_loss = 0, 0
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, qs_unbind) in enumerate(
                zip(predicted_rewards.unbind(0), reward.unbind(0), td_targets.unbind(0), predicted_q_values.unbind(1))):
            # Compute reward prediction loss
            reward_loss += (self.rho ** t) * torch.nn.functional.mse_loss(rew_pred_unbind, rew_unbind).item()
            # reward_loss += mathutil.soft_ce(rew_pred_unbind, rew_unbind, self.cfg).mean() * self.cfg.rho ** t
            # Compute value loss for all predicted Q-values
            for qs_unbind_unbind in qs_unbind.unbind(0):
                value_loss += (self.rho ** t) * torch.nn.functional.mse_loss(qs_unbind_unbind.unsqueeze(0),
                                                                             td_targets_unbind).item()
                # value_loss += mathutil.soft_ce(qs_unbind_unbind, td_targets_unbind, self.cfg).mean() * self.cfg.rho ** t

        # Step 06: Normalize the losses over the rollout horizon
        consistency_loss = consistency_loss / self.horizon
        reward_loss = reward_loss / self.horizon
        value_loss = value_loss / (2 * self.horizon)  # because we have 2 q networks

        # Step 07: Combine all losses into a total loss using configured coefficients
        total_loss = (
                self.consistency_coef * consistency_loss +
                self.reward_coef * reward_loss +
                self.value_coef * value_loss
        )

        # Update model
        self.optim.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm)
        self.optim.step()
        # self.optim.zero_grad(set_to_none = True)

        # Update policy
        self.policy_optim.zero_grad()
        policy_loss = self._calculate_policy_loss(latent_state_rollout.detach())
        policy_loss.backward()
        policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm)
        self.policy_optim.step()

        return total_loss.item()

        # # Update target Q-functions
        # self.model.soft_update_target_Q()
        #
        # # Return training statistics
        # self.model.eval()
        # info = TensorDict({
        #     "consistency_loss": consistency_loss,
        #     "reward_loss": reward_loss,
        #     "value_loss": value_loss,
        #     "total_loss": total_loss,
        #     "grad_norm": grad_norm,
        # })
        # info.update(policy_loss)
        # return info.detach().mean()

    def _predict_next_state(self, latent_state: torch.Tensor, latent_action: torch.Tensor):
        """
        Predicts the next latent state given a latent state and a latent action.
        """
        input = torch.hstack([latent_state, latent_action])
        next_latent_state = self.dynamics_net(input)
        return next_latent_state

    def _predict_reward(self, latent_state: torch.Tensor, latent_action: torch.Tensor):
        """
        Predicts one-step latent reward given a latent state and a latent action.
        """
        input = torch.hstack([latent_state, latent_action])
        latent_reward = self.reward_net(input)
        return latent_reward

    # TODO
    def _predict_action(self, latent_state: torch.Tensor):
        """
        Samples an action from the policy prior.
        The policy prior is a Gaussian distribution with
        mean and (log) std predicted by a neural network.
        """

        # Step 01: Get the gaussian policy prior from the policy network
        mean, log_std = self.policy_net(latent_state).chunk(2, dim = -1)
        log_std = mathutil.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = mathutil.gaussian_logprob(eps, log_std)

        # Scale log probability by action dimensions
        size = eps.shape[-1]
        scaled_log_prob = log_prob * size

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = mathutil.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = {
            "mean": mean,
            "log_std": log_std,
            "action_prob": 1.,
            "entropy": -log_prob,
            "scaled_entropy": -log_prob * entropy_scale,
        }
        return action, info

        # mean = output[..., :self.action_size]
        # self.policy(latent_state)

        # log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        # eps = torch.randn_like(mean)
        #
        # if self.cfg.multitask:  # Mask out unused action dimensions
        #     mean = mean * self._action_masks[task]
        #     log_std = log_std * self._action_masks[task]
        #     eps = eps * self._action_masks[task]
        #     action_dims = self._action_masks.sum(-1)[task].unsqueeze(-1)
        # else:  # No masking
        #     action_dims = None
        #
        # log_prob = math.gaussian_logprob(eps, log_std)
        #
        # # Scale log probability by action dimensions
        # size = eps.shape[-1] if action_dims is None else action_dims
        # scaled_log_prob = log_prob * size
        #
        # # Reparameterization trick
        # action = mean + eps * log_std.exp()
        # mean, action, log_prob = math.squash(mean, action, log_prob)
        #
        # entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        # info = TensorDict({
        #     "mean": mean,
        #     "log_std": log_std,
        #     "action_prob": 1.,
        #     "entropy": -log_prob,
        #     "scaled_entropy": -log_prob * entropy_scale,
        # })
        # return action, info
    #
    # # TODO
    # def Q(self, z, a, task, return_type = 'min', target = False, detach = False):
    #     """
    #     Predict state-action value.
    #     `return_type` can be one of [`min`, `avg`, `all`]:
    #         - `min`: return the minimum of two randomly subsampled Q-values.
    #         - `avg`: return the average of two randomly subsampled Q-values.
    #         - `all`: return all Q-values.
    #     `target` specifies whether to use the target Q-networks or not.
    #     """
    #
    #     z = torch.cat([z, a], dim = -1)
    #     if target:
    #         qnet = self._target_Qs
    #     elif detach:
    #         qnet = self._detach_Qs
    #     else:
    #         qnet = self._Qs
    #     out = qnet(z)
    #
    #     if return_type == 'all':
    #         return out
    #
    #     qidx = torch.randperm(self.cfg.num_q, device = out.device)[:2]
    #     Q = math.two_hot_inv(out[qidx], self.cfg)
    #     if return_type == "min":
    #         return Q.min(0).values
    #     return Q.sum(0) / 2
