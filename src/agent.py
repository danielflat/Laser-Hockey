from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from src.replaymemory import ReplayMemory
from src.util.constants import ADAM, ADAMW, EXPONENTIAL, L1, LINEAR, MSELOSS, SMOOTHL1, SUPPORTED_LOSS_FUNCTIONS, \
    SUPPORTED_OPTIMIZERS


class Agent(ABC):

    def __init__(self, agent_settings: dict, device: torch.device):
        # Hyperparams
        self.epsilon = agent_settings["EPSILON"]
        self.epsilon_start = agent_settings["EPSILON"]
        self.epsilon_min = agent_settings["EPSILON_MIN"]
        self.epsilon_decay = agent_settings["EPSILON_DECAY"]
        self.opt_iter = agent_settings["OPT_ITER"]
        self.batch_size = agent_settings["BATCH_SIZE"]
        self.discount = agent_settings["DISCOUNT"]
        self.gradient_clipping_value = agent_settings["GRADIENT_CLIPPING_VALUE"]
        self.target_net_update_freq = agent_settings["TARGET_NET_UPDATE_FREQ"]
        self.tau = agent_settings["TAU"]

        # Options
        self.epsilon_decay_strategy = agent_settings["EPSILON_DECAY_STRATEGY"]
        self.use_target_net = agent_settings["USE_TARGET_NET"]
        self.use_soft_updates = agent_settings["USE_SOFT_UPDATES"]
        self.use_gradient_clipping = agent_settings["USE_GRADIENT_CLIPPING"]
        self.epsilon_decay_strategy = agent_settings["EPSILON_DECAY_STRATEGY"]
        self.device: device = device
        self.use_clip_foreach = agent_settings["USE_CLIP_FOREACH"]
        self.USE_BF_16 = agent_settings["USE_BF16"]

    @abstractmethod
    def act(self, x: torch.Tensor) -> int:
        pass

    @abstractmethod
    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        pass

    @abstractmethod
    def setMode(self, eval: bool = False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        pass

    @abstractmethod
    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """
        pass

    @abstractmethod
    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        pass

    def adjust_epsilon(self, episode_i: int) -> None:
        """
        Here, we decay epsilon w.r.t. the number of run episodes
        :param episode_i: The ith episode
        """
        if self.epsilon_decay_strategy == LINEAR:
            new_candidate = np.maximum(self.epsilon_min, self.epsilon_start - (episode_i / self.epsilon_decay) * (
                        self.epsilon_start - self.epsilon_min))
            self.epsilon = new_candidate.item()
        elif self.epsilon_decay_strategy == EXPONENTIAL:
            new_candidate = np.maximum(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** episode_i))
            self.epsilon = new_candidate.item()
        else:
            raise NotImplementedError(
                f"The epsilon decay strategy '{self.epsilon_decay_strategy}' is not supported! Please choose another one!")

    def initOptim(self, optim: dict, parameters) -> torch.optim:
        """
        Initialize the optimizer based on the config.py
        :param optim: the optim config
        :param parameters: the parameters of the network to optimize
        :return: the optimizer
        """
        optim_name = optim["OPTIM_NAME"]
        if optim_name in SUPPORTED_OPTIMIZERS:
            if optim_name == ADAMW:
                return torch.optim.AdamW(parameters, lr=optim["LEARNING_RATE"], betas=optim["BETAS"],
                                         eps=optim["EPS"], weight_decay=optim["WEIGHT_DECAY"],
                                         fused=optim["USE_FUSION"])
            elif optim_name == ADAM:
                return torch.optim.Adam(parameters, lr=optim["LEARNING_RATE"], betas=optim["BETAS"],
                                         eps=optim["EPS"], weight_decay=optim["WEIGHT_DECAY"],
                                         fused=optim["USE_FUSION"])
        else:
            raise NotImplemented(f"The optimizer '{optim_name}' is not supported! Please choose another one!")

    def initLossFunction(self, loss_name: str):
        """
        Initialize the loss function based on the config.py
        :param loss_name: The name of the loss function
        :return: the loss function object
        """
        if loss_name in SUPPORTED_LOSS_FUNCTIONS:
            if loss_name == L1:
                return nn.L1Loss()
            elif loss_name == SMOOTHL1:
                return nn.SmoothL1Loss()
            elif loss_name == MSELOSS:
                return nn.MSELoss()
        else:
            raise NotImplemented(f"The Loss function '{loss_name}' is not supported! Please choose another one!")

    def updateTargetNet(self, soft_update: bool, source: nn.Module, target: nn.Module) -> None:
        """
        Updates the target network with the weights of the original one
        """
        assert self.use_target_net == True, "You must use have 'self.use_target == True' to call 'updateTargetNet()'"

        if soft_update:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′ where θ′ are the target net weights
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            # Do a hard parameter update. Copy all values from the origin to the target network
            target.load_state_dict(source.state_dict())
