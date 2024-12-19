import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.settings import ADAMW, L1, SUPPORTED_LOSS_FUNCTIONS, SUPPORTED_OPTIMIZERS


class QFunction(nn.Module):
    def __init__(self, state_size, hidden_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
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
    def __init__(self, state_shape: tuple[int, ...], action_size: int, options: dict, optim:dict, hyperparams:dict):
        super().__init__()

        self.isEval = None

        self.state_shape = state_shape
        self.action_size = action_size

        # Hyperparams
        self.opt_iter = hyperparams["OPT_ITER"]
        self.batch_size = hyperparams["BATCH_SIZE"]
        self.discount = hyperparams["DISCOUNT"]
        self.epsilon = hyperparams["EPSILON"]

        # Options
        self.use_target_net = options["USE_TARGET_NET"]
        self.target_net_update_iter = options["TARGET_NET_UPDATE_ITER"]


        # Define the Q-Network
        self.Q = QFunction(state_size=state_shape[0],
                           hidden_size=128,
                           action_size=action_size)

        # If you want to use a target network, it is defined here
        if self.use_target_net:
            self.targetQ = QFunction(state_size=state_shape[0],
                                     hidden_size=128,
                                     action_size=action_size)
            # Copy the Q network
            self.updateTargetNet()

        # Define the Optimizer
        optim_name = optim["OPTIM_NAME"]
        if optim_name in SUPPORTED_OPTIMIZERS:
            if optim_name == ADAMW:
                self.optimizer = torch.optim.AdamW(self.Q.parameters(), lr = optim["LEARNING_RATE"], betas=optim["BETAS"], eps=optim["EPS"], weight_decay=optim["WEIGHT_DECAY"], fused=optim["USE_FUSION"])
        else:
            raise NotImplemented(f"The optimizer '{optim_name}' is not supported! Please choose another one!")

        # Define Loss function
        loss_name = options["LOSS_FUNCTION"]
        if loss_name in SUPPORTED_LOSS_FUNCTIONS:
            if loss_name == L1:
                self.criterion = nn.L1Loss()
        else:
            raise NotImplemented(f"The Loss function '{loss_name}' is not supported! Please choose another one!")


    def updateTargetNet(self) -> None:
        """
        Updates the target network with the weights of the original one
        """
        assert self.use_target_net == True, "You must use have 'self.use_target == True' to call 'updateTargetNet()'"
        self.targetQ.load_state_dict(self.Q.state_dict())

    def optimize(self, memory: ReplayMemory) -> list[float]:
        """
        This function is used to train and optimize the Q Network with the help of the replay memory.
        :return: A list of all losses during optimization
        """
        assert self.isEval == False, "Make sure to put the model in training mode before calling the opt. routine"

        losses = []
        for i in range(self.opt_iter):
            # Update the target net after some iterations again
            if self.use_target_net and i % self.target_net_update_iter == 0:
                self.updateTargetNet()

            self.optimizer.zero_grad()

            state, action, reward, next_state, done, info = memory.sample(self.batch_size)

            # Forward step
            predicted_q_value = self.Q.QValue(state, action)
            if self.use_target_net:
                td_target = reward + (1 - done) * self.discount * self.targetQ.maxQValue(state)
            else:
                td_target = reward + (1 - done) * self.discount * self.Q.maxQValue(state)

            loss = self.criterion(predicted_q_value, td_target)
            losses.append(loss.item())

            # Backward step
            loss.backward()
            self.optimizer.step()

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
            if self.use_target_net:
                self.targetQ.eval() # df: should not be necessary, but safe is safe
        else:
            self.Q.train()
            if self.use_target_net:
                self.targetQ.train()





