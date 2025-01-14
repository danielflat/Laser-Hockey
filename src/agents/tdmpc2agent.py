from typing import List

import gymnasium
import numpy as np
import torch
from sympy import false
from torch import nn
from torch.nn import functional as F

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util import mathutil

"""
Our implementation of TD-MPC2
See Paper: https://arxiv.org/abs/2310.16828
See GitHub for original implementation: https://github.com/nicklashansen/tdmpc2/tree/main

Author: Daniel Flat
"""

def _log_std_clamp(log_std, min_value = -10, max_value = 2):
    """Clamp log_std to a specific range."""
    return torch.clamp(log_std, min_value, max_value)


# TODO
class EncoderNet(nn.Module):
    def __init__(self, state_size: int, latent_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            nn.Linear(state_size, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size),
            nn.LayerNorm(latent_size)  # SimNorm equivalent TODO# add SimNorm
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encoding the state into a latent space because the model can handle with its one latent space better than the actual one.
        """
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

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the next latent state given the current latent state and the action.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        next_latent_state = self.encoder_net(input)
        return next_latent_state


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

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the reward given the current latent state and the action.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        latent_state = self.encoder_net(input)
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
            nn.Linear(128, 2 * action_size)
        )  # mean, std

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Predict the mean and log_std of the action distribution given the latent state.
        """
        output = self.actor_net(latent_state)
        return output


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

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the q-value given the latent state and the action.
        THIS IS JUST AN ESTIMATE, NOT THE REAL Q-VALUE.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        q_value = self.critic_net(input)
        return q_value


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
        self.min_action_torch = torch.tensor(self.min_action, device = self.device)
        self.max_action = action_space.high[:self.action_size]
        self.max_action_torch = torch.tensor(self.max_action, device = self.device)

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

        # if self.USE_COMPILE:
        #     logging.info("Start compiling the TD-MPC2 agent.")
        #     self._update = torch.compile(self._update)
        #     logging.info("Finished compiling the TD-MPC2 agent")

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"TDMPC2Agent"

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> np.ndarray:
        """
        Select an action by planning some steps in the future and take the best estimated action.

        During training, we add noise to the proposed action.
        """
        proposed_action = self._plan(state)
        # if not self.isEval:
        #     noise = self.noise.sample() * self.noise_factor
        #     proposed_action += noise
        #     proposed_action = np.clip(proposed_action, self.min_action, self.max_action)

        return proposed_action

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        losses = []

        # Step 01: We take the whole sequence from the batch
        state, action, reward, next_state, done, _ = memory.sample(batch_size = self.batch_size, randomly = False)

        for i in range(1, self.opt_iter + 1):
            # Step 02: We randomly subsample a horizon-long trajectory for updating
            starting_point = torch.randint(0, state.shape[0] - self.horizon, (1,), device = self.device)
            state = state[starting_point:starting_point + self.horizon + 1]
            action = action[starting_point:starting_point + self.horizon + 1]
            reward = reward[starting_point:starting_point + self.horizon + 1]
            next_state = next_state[starting_point:starting_point + self.horizon + 1]
            done = done[starting_point:starting_point + self.horizon + 1]

            # Step 03: Update the agent.
            loss = self._update(state, action, reward, next_state, done)

            # Step 04: Keep track of the loss
            losses.append(loss)

            # Step 05: After some time, update the agent
            if episode_i % self.target_net_update_freq == 0:
                self.updateTargetNet(soft_update = self.use_soft_updates, source = self.q1_net,
                                     target = self.q1_target_net)
                self.updateTargetNet(soft_update = self.use_soft_updates, source = self.q2_net,
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
        latent_space (latent_state)
        action_sequence (num_samples, horizon, action_size)
        """

        # Step 01: Let's first predict the state and the discounted reward in the num of `horizon` in the future.
        _G, _discount = 0, 1
        latent_state = latent_state.unsqueeze(0).repeat(action_sequence.shape[0], 1)  # expand the latent space
        for action in action_sequence.unbind(1):
            reward = self.reward_net(latent_state, action)
            latent_state = self.dynamics_net(latent_state, action)
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
    def _plan(self, state: torch.Tensor) -> np.ndarray:
        """
        Plan proposed_action sequence of action_sequence_samples using the learned world model.

        Args:
            state (torch.Tensor): Real state from which to plan in the future.
            is_training (bool): Whether to use the mean of the action distribution or not. if true, we sample, if not, we take the mean.

        Returns:
            torch.Tensor: Action to take in the environment at the current timestep.
        """
        # Step 01: Sample trajectories based on our policy
        latent_state = self.encoder_net(state)
        mean, log_std = self.policy_net(latent_state).chunk(2, dim = -1)  # Use the policy network as a prior
        std = log_std.exp()

        # reparameterization trick: We take the mean and std as priors for planning
        # we plan #horizon steps in the future by sampling some future actions
        action_sequence_samples = mean + std * torch.randn(self.num_samples, self.horizon, self.action_size,
                                                           device = self.device)

        # we evaluate our action sequences by looking for the one with the highest q-value estimate
        q_values = self._estimate_q_of_action_sequence(latent_state, action_sequence_samples)

        # sample the next immediate action w.r.t. the highest q-value estimate
        best_idx = q_values.argmax(dim = 0)
        best_action = action_sequence_samples[best_idx, 0].squeeze(dim = 0)  # Best immediate action

        # we normalize the action
        normalized_action = torch.clamp(best_action, self.min_action_torch, self.max_action_torch)
        return normalized_action.cpu().numpy()

    def _calculate_policy_loss(self, latent_state_sequence: torch.Tensor) -> torch.Tensor:
        """
        calculates the policy net using a sequence of latent states.

        """
        # self.policy_optim.zero_grad()
        action, info = self._predict_action(latent_state_sequence)
        q_value = self._min_q_value(latent_state_sequence, action, use_target = false)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.rho, torch.arange(len(q_value), device = self.device))
        pi_loss = (rho * -(self.entropy_coef * info["scaled_entropy"] + q_value).mean()).mean()
        return pi_loss

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
            td_target = self._td_target(next_latent_state, reward, done, use_target = True)

        latent_state = self.encoder_net(state)
        q1_value = self.q1_net(latent_state, action)
        q2_value = self.q2_net(latent_state, action)
        q_loss = F.mse_loss(q1_value, td_target) + F.mse_loss(q2_value, td_target)

        latent_state_rollout = self._rollout(latent_state, action)
        reward_loss = self._reward_loss(latent_state_rollout, reward)
        consistency_loss = self._consistency_loss(latent_state_rollout, next_latent_state)

        total_loss = (
                self.consistency_coef * consistency_loss
                + self.reward_coef * reward_loss
                + self.value_coef * q_loss
        )

        self.optim.zero_grad()
        total_loss.backward()
        if self.use_norm_clipping:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip_norm,
                                                             foreach = self.use_clip_foreach)
        self.optim.step()

        # Update policy
        self.policy_optim.zero_grad()
        policy_loss = self._calculate_policy_loss(latent_state_rollout.detach())
        policy_loss.backward()
        if self.use_norm_clipping:
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip_norm,
                                                              foreach = self.use_clip_foreach)
        self.policy_optim.step()

        return total_loss.item()


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

    @torch.no_grad()
    def _td_target(self, next_latent_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
                   use_target) -> torch.Tensor:
        next_action, _ = self._predict_action(next_latent_state)
        target_q = self._min_q_value(next_latent_state, next_action, use_target)
        td_target = reward + self.discount * (1 - done) * target_q
        return td_target

    def _rollout(self, latent_state, actions):
        latent_rollout = []
        _latent_state = latent_state[0]
        for action in actions.unbind(dim = 0):
            _latent_state = self.dynamics_net(_latent_state, action)
            latent_rollout.append(_latent_state)
        return torch.stack(latent_rollout, dim = 0)

    def _reward_loss(self, rollout_latent_space, reward):
        with torch.no_grad():
            predicted_action, _ = self._predict_action(rollout_latent_space)
        predicted_rewards = self.reward_net(rollout_latent_space, predicted_action)
        return F.mse_loss(predicted_rewards, reward)

    def _consistency_loss(self, latent_state_rollout, next_latent_state):
        return F.mse_loss(latent_state_rollout, next_latent_state)
