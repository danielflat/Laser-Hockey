import logging

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.constants import EXPONENTIAL, LINEAR


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
        q_value = all_q_values.gather(dim=1, index=action)
        return q_value

    def maxQValue(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(state)
            max_q_value = torch.max(q_values, dim=1)[0].unsqueeze(-1)
            return max_q_value  # (batch_size, 1)

    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.forward(state)
            greedyAction = torch.max(q_values, dim=1)[1]
            return greedyAction



class DQNAgent(Agent):
    def __init__(self, state_shape: tuple[int, ...], action_size: int, options: dict, optim:dict, hyperparams:dict,
                 device: device):
        super().__init__()

        self.isEval = None

        self.state_shape = state_shape
        self.action_size = action_size

        # Hyperparams
        self.opt_iter = hyperparams["OPT_ITER"]
        self.batch_size = hyperparams["BATCH_SIZE"]
        self.discount = hyperparams["DISCOUNT"]
        self.epsilon = hyperparams["EPSILON"]
        self.epsilon_start = hyperparams["EPSILON"]
        self.epsilon_min = hyperparams["EPSILON_MIN"]
        self.epsilon_decay = hyperparams["EPSILON_DECAY"]
        self.gradient_clipping_value = hyperparams["GRADIENT_CLIPPING_VALUE"]
        self.target_net_update_freq = hyperparams["TARGET_NET_UPDATE_FREQ"]
        self.tau = hyperparams["TAU"]

        # Options
        self.use_target_net = options["USE_TARGET_NET"]
        self.use_soft_updates = options["USE_SOFT_UPDATES"]
        self.use_gradient_clipping = options["USE_GRADIENT_CLIPPING"]
        self.epsilon_decay_strategy = options["EPSILON_DECAY_STRATEGY"]
        self.device: device = device
        self.use_clip_foreach = options["USE_CLIP_FOREACH"]
        self.USE_BF_16 = options["USE_BF16"]


        # Define the Q-Network
        self.Q = QFunction(state_size=state_shape[0],
                           hidden_size=128,
                           action_size=action_size)
        self.Q.to(self.device)

        # If you want to use a target network, it is defined here
        if self.use_target_net:
            self.targetQ = QFunction(state_size=state_shape[0],
                                     hidden_size=128,
                                     action_size=action_size)
            self.targetQ.to(self.device)
            self.targetQ.eval() # Set it always to Eval mode
            self.updateTargetNet(soft_update=False)  # Copy the Q network

        # Define the Optimizer
        self.optimizer = self.initOptim(optim=optim, parameters=self.Q.parameters())

        # Define Loss function
        self.criterion = self.initLossFunction(loss_name = options["LOSS_FUNCTION"])


    def updateTargetNet(self, soft_update) -> None:
        """
        Updates the target network with the weights of the original one
        """
        assert self.use_target_net == True, "You must use have 'self.use_target == True' to call 'updateTargetNet()'"

        if soft_update:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′ where θ′ are the target net weights
            target_net_state_dict = self.targetQ.state_dict()
            origin_net_state_dict = self.Q.state_dict()
            for key in origin_net_state_dict:
                target_net_state_dict[key] = origin_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                            1 - self.tau)
            self.targetQ.load_state_dict(target_net_state_dict)
        else:
            # Do a hard parameter update. Copy all values from the origin to the target network
            self.targetQ.load_state_dict(self.Q.state_dict())

    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        """
        This function is used to train and optimize the Q Network with the help of the replay memory.
        :return: A list of all losses during optimization
        """
        assert self.isEval == False, "Make sure to put the model in training mode before calling the opt. routine"

        losses = []
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            self.optimizer.zero_grad()

            state, action, reward, next_state, done, info = memory.sample(self.batch_size)

            # Forward step
            if self.USE_BF_16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
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
                torch.nn.utils.clip_grad_value_(parameters=self.Q.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
            self.optimizer.step()

            # Update the target net after some iterations again
            if self.use_target_net and i % self.target_net_update_freq == 0:
                self.updateTargetNet(soft_update=self.use_soft_updates)

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

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.Q.eval()
        else:
            self.Q.train()

    def saveModel(self, fileName: str) -> None:
        """
        Saves the model parameters of the agent.
        """
        torch.save(self.Q.state_dict(), fileName)
        logging.info(f"Q network weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        self.Q.load_state_dict(torch.load(file_name))
        logging.info(f"Q network weights loaded successfully!")


    def adjust_epsilon(self, episode_i: int) -> None:
        """
        Here, we decay epsilon w.r.t. the number of run episodes
        :param episode_i: The ith episode
        """
        if self.epsilon_decay_strategy == LINEAR:
            new_candidate = np.maximum(self.epsilon_min, self.epsilon_start - (episode_i/self.epsilon_decay) * (self.epsilon_start - self.epsilon_min))
            self.epsilon = new_candidate.item()
        elif self.epsilon_decay_strategy == EXPONENTIAL:
            new_candidate = np.maximum(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** episode_i))
            self.epsilon = new_candidate.item()
        else:
            raise NotImplementedError(f"The epsilon decay strategy '{self.epsilon_decay_strategy}' is not supported! Please choose another one!")

    def calc_td_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        if self.use_target_net:
            return reward + (1 - done) * self.discount * self.targetQ.maxQValue(next_state)
        else:
            return reward + (1 - done) * self.discount * self.Q.maxQValue(next_state)

    def forward_pass(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        # Step 01: Calculate the predicted q value
        predicted_q_value = self.Q.QValue(state, action)

        # Step 02: Calculate the td target
        td_target = self.calc_td_target(reward, done, next_state)

        # Step 03: Finally, we calculate the loss
        loss = self.criterion(predicted_q_value, td_target)
        return loss





