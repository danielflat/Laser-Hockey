from __future__ import annotations

import copy

import gymnasium
import logging
import numpy as np
import torch
from sympy import false
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util import mathutil
from src.util.layerutil import NormedLinear
from src.util.noiseutil import initNoise

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
            NormedLinear(in_features=state_size, out_features=latent_size, activation_function="Mish"),
            NormedLinear(in_features=latent_size, out_features=latent_size, activation_function="SimNorm"),
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
            NormedLinear(in_features = latent_size + action_size, out_features = latent_size,
                         activation_function = "Mish"),
            # NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "SimNorm"),
        )

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the next latent state given the current latent state and the action.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        next_latent_state = self.encoder_net(input)
        return next_latent_state


class RewardNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.encoder_net = nn.Sequential(
            NormedLinear(in_features = latent_size + action_size, out_features = latent_size,
                         activation_function = "Mish"),
            # NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            nn.Linear(latent_size, 1, bias = False),
        )

    def forward(self, latent_state: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        """
        Predict the reward given the current latent state and the action.
        """
        input = torch.cat([latent_state, latent_action], dim = -1)
        latent_state = self.encoder_net(input)
        return latent_state


class ActorNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.actor_net = nn.Sequential(
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            nn.Linear(latent_size, 2 * action_size, bias = True),
        )  # mean, std

    def forward(self, latent_state: torch.Tensor) -> torch.Tensor:
        """
        Predict the mean and log_std of the action distribution given the latent state.
        """
        output = self.actor_net(latent_state)
        return output


class CriticNet(nn.Module):
    def __init__(self, latent_size: int, action_size: int):
        super().__init__()

        self.critic_net = nn.Sequential(
            NormedLinear(in_features = latent_size + action_size, out_features = latent_size, dropout = 0.01,
                         activation_function = "Mish"),
            NormedLinear(in_features = latent_size, out_features = latent_size, activation_function = "Mish"),
            nn.Linear(latent_size, 1, bias = True))

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
# TODO: Bad interface. Extend it with more parameters
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

        self.noise = initNoise(action_shape = (self.action_size,), noise_settings = td_mpc2_settings["NOISE"],
                               device = self.device)
        self.noise_factor = td_mpc2_settings["NOISE"]["NOISE_FACTOR"]
        self.use_own_noise = td_mpc2_settings["USE_OWN_NOISE"]

        self.horizon = td_mpc2_settings["HORIZON"]
        self.mmpi_iterations = td_mpc2_settings["MMPI_ITERATIONS"]
        self.num_trajectories = td_mpc2_settings["NUM_TRAJECTORIES"]
        self.num_samples = td_mpc2_settings["NUM_SAMPLES"]
        self.num_elites = td_mpc2_settings["NUM_ELITES"]
        self.min_std = td_mpc2_settings["MIN_STD"]
        self.max_std = td_mpc2_settings["MAX_STD"]
        self.temperature = td_mpc2_settings["TEMPERATURE"]
        self.latent_size = td_mpc2_settings["LATENT_SIZE"]
        self.log_std_min = td_mpc2_settings["LOG_STD_MIN"]
        self.log_std_max = td_mpc2_settings["LOG_STD_MAX"]
        self.log_std_dif = self.log_std_max - self.log_std_min
        self.entropy_coef = td_mpc2_settings["ENTROPY_COEF"]
        self.enc_lr_scale = td_mpc2_settings["ENC_LR_SCALE"]
        self.lr = td_mpc2_settings["OPTIMIZER"]["LEARNING_RATE"]
        self.consistency_coef = td_mpc2_settings["CONSISTENCY_COEF"]
        self.reward_coef = td_mpc2_settings["REWARD_COEF"]
        self.q_coef = td_mpc2_settings["Q_COEF"]

        # MPPI: we save the mean from the previous planning in this variable to make the planning better
        self._prior_mean = torch.zeros(self.horizon - 1, self.action_size, device = self.device)

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

        self.optim = self.initOptim(optim = td_mpc2_settings["OPTIMIZER"],
                                    parameters = [
                                        {'params': self.encoder_net.parameters(), 'lr': self.lr * self.enc_lr_scale},
                                        {'params': self.dynamics_net.parameters()},
                                        {'params': self.reward_net.parameters()},
                                        {'params': self.q1_net.parameters()},
                                        {'params': self.q2_net.parameters()}])

        self.policy_optim = self.initOptim(optim = td_mpc2_settings["OPTIMIZER"],
                                           parameters = self.policy_net.parameters())

        self.consistency_criterion = self.initLossFunction(loss_name = td_mpc2_settings["CONSISTENCY_LOSS_FUNCTION"])
        self.reward_criterion = self.initLossFunction(loss_name = td_mpc2_settings["REWARD_LOSS_FUNCTION"])
        self.q_criterion = self.initLossFunction(loss_name = td_mpc2_settings["Q_LOSS_FUNCTION"])

        if self.USE_COMPILE:
            logging.info("Start compiling the TD-MPC2 agent.")
            self._update = torch.compile(self._update)
            logging.info("Finished compiling the TD-MPC2 agent")

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

        return proposed_action

    def optimize(self, memory: ReplayMemory, episode_i: int) -> Dict[str, Any]:
        statistics_episode = []

        for i in range(1, self.opt_iter + 1):
            # Step 01: We randomly subsample a horizon-long trajectory for updating
            state, action, reward, next_state, done = memory.sample_horizon(batch_size = self.batch_size,
                                                                            horizon = self.horizon)

            # Step 02: Update the agent.
            statistics_iter = self._update(state, action, reward, next_state, done)

            # Step 03: Keep track of the loss
            statistics_episode.append(statistics_iter)

            # Step 04: After some time, update the agent
            if i % self.target_net_update_freq == 0:
                self.updateTargetNet(soft_update = self.use_soft_updates, source = self.q1_net,
                                     target = self.q1_target_net)
                self.updateTargetNet(soft_update = self.use_soft_updates, source = self.q2_net,
                                     target = self.q2_target_net)

        # We sum up the statistics from each optimization iteration by accumulating statistical quantities
        # such that we can log it
        sum_up_statistics_training_iter = {
            "Avg. Total Loss": np.array([episode["total_loss"] for episode in statistics_episode]).mean(),
            "Avg. Policy Loss": np.array([episode["policy_loss"] for episode in statistics_episode]).mean(),
            "Avg. Consistency Loss": np.array(
                [episode["consistency_loss"] for episode in statistics_episode]).mean(),
            "Avg. Reward Loss": np.array([episode["reward_loss"] for episode in statistics_episode]).mean(),
            "Avg. Q Loss": np.array([episode["q_loss"] for episode in statistics_episode]).mean(),
            "Avg. Total Grad Norm": np.array([episode["total_grad_norm"] for episode in statistics_episode]).mean(),
            "Avg. Policy Grad Norm": np.array(
                [episode["policy_grad_norm"] for episode in statistics_episode]).mean(),
        }

        return sum_up_statistics_training_iter

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
        self.encoder_net.load_state_dict(checkpoint["encoder_net"])
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.dynamics_net.load_state_dict(checkpoint["dynamics_net"])
        self.reward_net.load_state_dict(checkpoint["reward_net"])
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

    # TODO
    @torch.no_grad()
    def _plan(self, state: torch.Tensor) -> np.ndarray:
        """
        Plan proposed_action sequence of action_sequence_samples using the learned world model.

        Args:
            state (torch.Tensor): Real state from which to plan in the future.

        Returns:
            torch.Tensor: Action to take in the environment at the current timestep.
        """
        # Step 01: we use our prior policy net to give some action proposals
        latent_state = self.encoder_net(state)
        pi_actions = torch.empty(self.horizon, self.num_trajectories, self.action_size, device = self.device)
        _latent_state = latent_state.repeat(self.num_trajectories, 1)
        for t in range(self.horizon - 1):
            pi_actions[t], _ = self._predict_action(_latent_state)
            _latent_state = self.dynamics_net(_latent_state, pi_actions[t])
        pi_actions[-1], _ = self._predict_action(_latent_state)

        # Step 02: The rest of the action proposals are done by the MPPI algorithm.
        # Initialize state and parameters to prepare MPPI.
        latent_state = latent_state.repeat(self.num_samples, 1)
        mean = torch.zeros(self.horizon, self.action_size, device = self.device)
        mean[:-1] = self._prior_mean  # get the mean from the planning before
        std = torch.full((self.horizon, self.action_size), self.max_std, dtype = torch.float, device = self.device)
        actions = torch.empty(self.horizon, self.num_samples, self.action_size, device = self.device)
        actions[:, :self.num_trajectories] = pi_actions

        # Iterate MPPI
        for _ in range(self.mmpi_iterations):
            # Step 03: Sample random actions by using the mean and std.
            r = torch.randn(self.horizon, self.num_samples - self.num_trajectories, self.action_size,
                            device = std.device)
            actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
            actions_sample = actions_sample.clamp(self.min_action_torch, self.max_action_torch)
            actions[:, self.num_trajectories:] = actions_sample

            # Compute elite actions
            value = self._estimate_q_of_action_sequence(latent_state, actions)
            elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim = 0).indices
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update score, mean and std. parameters for the next iteration
            score = torch.softmax(self.temperature * elite_value, dim = 0)
            mean = (score.unsqueeze(0) * elite_actions).sum(dim = 1) / (score.sum(0) + 1e-9)
            std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim = 1) / (
                    score.sum(0) + 1e-9)).sqrt()
            std = std.clamp(self.min_std, self.max_std)

        # Select action by sampling the one with the highest score w.r.t. gumbel noise perturbation
        gumbel_noise = torch.distributions.Gumbel(0, 1).sample(score.shape).to(self.device)
        noisy_scores = (score + gumbel_noise).squeeze()
        selected_index = torch.argmax(noisy_scores, dim = 0)
        planned_action_sequence = elite_actions[:, selected_index, :]
        planned_action, std = planned_action_sequence[0], std[0]

        # save the mean for the next planning
        self._prior_mean = copy.deepcopy(mean[1:])

        # in training mode, we add some noise
        if not self.isEval:
            if self.use_own_noise:
                planned_action = planned_action + self.noise_factor * torch.tensor(self.noise.sample(),
                                                                                   dtype=torch.float32,
                                                                               device=self.device)  # using pink noise
            else:
                planned_action = planned_action + std * torch.randn(self.action_size,
                                                                    device=self.device)  # adding some noise based on the std.
        return planned_action.clamp(self.min_action_torch, self.max_action_torch).cpu().numpy()

    def _update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
                done: torch.Tensor) -> Dict[str, int | float | bool | Any]:
        # Step 01: Compute targets
        with torch.no_grad():
            next_latent_state = self.encoder_net(next_state)
            td_target = self._td_target(next_latent_state, reward, done, use_target = True)

        # Step 02: We encode the horizon into the latent space
        latent_state = self.encoder_net(state)

        # Step 03: Next, we roll out our latent_state with our sequence of actions for the prediction of the next state #horizon-steps in the future
        # At the same time, we already calculate our prediction loss
        _latent_state = latent_state[:, 0]
        prediction_rollout = [_latent_state]
        _discount = 1
        consistency_loss = 0
        # We go over all actions of the horizon sequence
        for t, (_action, _next_latent_state) in enumerate(
                zip(action.unbind(dim = 1), next_latent_state.unbind(dim = 1))):
            _latent_state = self.dynamics_net(_latent_state, _action)
            prediction_rollout.append(_latent_state)
            # We calc. the discounted consistency loss.
            consistency_loss += _discount * self.consistency_criterion(_latent_state, _next_latent_state)
            _discount *= self.discount
        prediction_rollout = torch.stack(prediction_rollout, dim = 1)
        # prediction_rollout = self._rollout(latent_state[:, 0], action)

        # Step 04: We predict the q_value and the reward of the last rollout
        _prediction_rollout = prediction_rollout[:, :-1]  # we throw away the last latent rollout to do predictions
        q1_prediction = self.q1_net(_prediction_rollout, action)
        q2_prediction = self.q2_net(_prediction_rollout, action)
        reward_prediction = self.reward_net(_prediction_rollout, action)

        # Step 05: We calculate the reward and q loss
        reward_loss = 0
        q_loss = 0
        _discount = 1
        for t, (rew_pred_unbind, rew_unbind, td_targets_unbind, q1_unbind, q2_unbind) in enumerate(
                zip(reward_prediction.unbind(1), reward.unbind(1), td_target.unbind(1), q1_prediction.unbind(1),
                    q2_prediction.unbind(1))):
            reward_loss += _discount * self.reward_criterion(rew_pred_unbind, rew_unbind)
            q_loss += _discount * (self.q_criterion(q1_unbind, td_targets_unbind)) + (
                self.q_criterion(q2_unbind, td_targets_unbind))
            _discount *= self.discount

        # Step 08: Normalize the losses
        consistency_loss = consistency_loss / self.horizon
        reward_loss = reward_loss / self.horizon
        q_loss = q_loss / (2 * self.horizon)  # 2* because we have two q nets

        # Model objective loss: See formula 3
        total_loss = (((self.consistency_coef * consistency_loss)
                       + (self.reward_coef * reward_loss))
                      + (self.q_coef * q_loss))

        self.optim.zero_grad()
        total_loss.backward()
        if self.use_norm_clipping:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.norm_clipping_value,
                                                             foreach = self.use_clip_foreach)
        else:
            total_grad_norm = None
        self.optim.step()

        # Update policy
        self.policy_optim.zero_grad()
        policy_loss = self._calculate_policy_loss(prediction_rollout.detach())
        policy_loss.backward()
        if self.use_norm_clipping:
            policy_grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.norm_clipping_value,
                                                              foreach = self.use_clip_foreach)
        else:
            policy_grad_norm = None
        self.policy_optim.step()

        statistics = {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "consistency_loss": consistency_loss.item(),
            "reward_loss": reward_loss.item(),
            "q_loss": q_loss.item(),
            "total_grad_norm": total_grad_norm.item(),
            "policy_grad_norm": policy_grad_norm.item(),
        }

        return statistics

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
        # latent_state = latent_state.unsqueeze(0).repeat(action_sequence.shape[0], 1)  # expand the latent space
        for action in action_sequence.unbind(0):
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

    def _calculate_policy_loss(self, latent_state_sequence: torch.Tensor) -> torch.Tensor:
        action, info = self._predict_action(latent_state_sequence)
        q_value = self._min_q_value(latent_state_sequence, action, use_target = false)

        # Loss is a weighted sum of horizon Q-values
        policy_loss = 0
        _discount = 1
        for _q_value, _entropy in zip(q_value.unbind(1), info["scaled_entropy"].unbind(1)):
            policy_loss += -_discount * torch.mean(self.entropy_coef * _entropy + _q_value)
            _discount *= self.discount
        return policy_loss

    @torch.no_grad()
    def _td_target(self, next_latent_state: torch.Tensor, reward: torch.Tensor, done: torch.Tensor,
                   use_target) -> torch.Tensor:
        next_action, _ = self._predict_action(next_latent_state)
        target_q = self._min_q_value(next_latent_state, next_action, use_target)
        td_target = reward + self.discount * (1 - done) * target_q
        return td_target

    def reset(self):
        self._prior_mean = torch.zeros(self.horizon - 1, self.action_size, device = self.device)
