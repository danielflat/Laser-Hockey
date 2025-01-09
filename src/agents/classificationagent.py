import logging
import os
from typing import List, Tuple

import numpy as np

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path

"""
This is a classification version of the dqn agent regarding the paper

"Stop Regressing: Training Value Functions via
Classification for Scalable Deep RL" https://arxiv.org/pdf/2403.03950

Author: Daniel Flat
"""

import torch
import torch.special
import torch.nn as nn
import torch.nn.functional as F


class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(min_value, max_value, num_bins + 1, dtype = torch.float32)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma))
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim = -1)


class QFunction(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates all Q values for the given state
        """
        x = self.network(x)
        return x

    def QValue(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Q value for the given state and action
        """
        all_q_values = self.forward(state)
        q_value = all_q_values.gather(dim = 1, index = action)
        return q_value

    def maxQValue(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(state)
            max_q_value = torch.max(q_values, dim = 1)[0].unsqueeze(-1)
            return max_q_value  # (batch_size, 1)

    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(state)
            greedyAction = torch.max(q_values, dim = 1)[1]
            return greedyAction


class ClassificationAgent(Agent):
    def __init__(self, state_shape: Tuple[int, ...], action_size: int, agent_settings: dict,
                 classification_settings: dict,
                 device: torch.device):
        super().__init__(agent_settings = agent_settings, device = device)

        self.isEval = None

        self.state_shape = state_shape
        self.action_size = action_size

        # Define the Q-Network
        self.Q = n(state_size = state_shape[0],
                   hidden_size = 128,
                   action_size = action_size)
        self.Q.to(self.device)

        # If you want to use a target network, it is defined here
        if self.use_target_net:
            self.targetQ = QFunction(state_size = state_shape[0],
                                     hidden_size = 128,
                                     action_size = action_size)
            self.targetQ.to(self.device)
            self.targetQ.eval()  # Set it always to Eval mode
            self.updateTargetNet(soft_update = False, source = self.Q, target = self.targetQ)  # Copy the Q network

        # Define the Optimizer
        self.optimizer = self.initOptim(optim = classification_settings["OPTIMIZER"], parameters = self.Q.parameters())

        # Define Loss function
        self.criterion = self.initLossFunction(loss_name = classification_settings["LOSS_FUNCTION"])

        # Activate torch.compile if wanted
        if self.USE_COMPILE:
            self.Q = torch.compile(self.Q)
            if self.use_target_net:
                self.targetQ = torch.compile(self.targetQ)

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        This function is used to train and optimize the Q Network with the help of the replay memory.
        :return: A list of all losses during optimization
        """
        assert self.isEval == False, "Make sure to put the model in training mode before calling the opt. routine"

        losses = []
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            self.optimizer.zero_grad()

            state, action, reward, next_state, done, info = memory.sample(self.batch_size, randomly = True)
            # We have to convert the actions to a long tensor
            action = action.to(dtype = torch.long)

            # Forward step
            if self.USE_BF_16:
                with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                    loss = self.forward_pass(state, action, reward, next_state, done)
            else:
                loss = self.forward_pass(state, action, reward, next_state, done)

            # Track the loss
            losses.append(loss.item())

            # Backward step
            loss.backward()

            # if we want to clip our gradients
            if self.use_gradient_clipping:
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(parameters = self.Q.parameters(),
                                                clip_value = self.gradient_clipping_value,
                                                foreach = self.use_clip_foreach)
            self.optimizer.step()

            # Update the target net after some iterations again
            if self.use_target_net and i % self.target_net_update_freq == 0:
                self.updateTargetNet(soft_update = self.use_soft_updates, source = self.Q, target = self.targetQ)

        # after each optimization, we want to decay epsilon
        self.adjust_epsilon(episode_i)

        return losses

    def act(self, state: torch.Tensor) -> int:
        """
        The Agent chooses an action.
        In Evaluation mode, we always exploit the best action.
        In Training mode, we sample an action based on epsilon greedy with the given epsilon hyperparam.
        :param state: The state
        :return: The action TODO: Yet only as discrete variable available
        """
        with torch.no_grad():
            # In evaluation mode, we always exploit
            if self.isEval:
                return self.Q.greedyAction(state.unsqueeze(0)).item()

            # In training mode, use epsilon greedy action sampling
            rdn = np.random.random()
            if rdn <= self.epsilon:
                # Exploration
                return np.random.randint(low = 0, high = self.action_size)
            else:
                # Exploitation
                return self.Q.greedyAction(state.unsqueeze(0)).item()

    def forward_pass(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
                     done: torch.Tensor) -> torch.Tensor:
        # Step 01: Calculate the predicted q value
        predicted_q_value = self.Q.QValue(state, action)

        # Step 02: Calculate the td target
        td_target = self.calc_td_target(reward, done, next_state)

        # Step 03: Finally, we calculate the loss
        loss = self.criterion(predicted_q_value, td_target)
        return loss

    def calc_td_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        if self.use_target_net:
            return reward + (1 - done) * self.discount * self.targetQ.maxQValue(next_state)
        else:
            return reward + (1 - done) * self.discount * self.Q.maxQValue(next_state)

    def setMode(self, eval = False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.Q.eval()
        else:
            self.Q.train()

    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok = True)
        torch.save(self.Q.state_dict(), file_path)
        logging.info(f"Q network weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        self.Q.load_state_dict(torch.load(file_name))
        logging.info(f"Q network weights loaded successfully!")
