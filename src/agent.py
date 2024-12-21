from abc import ABC, abstractmethod

import torch
from torch import nn

from src.replaymemory import ReplayMemory
from src.util.constants import ADAM, ADAMW, L1, SMOOTHL1, SUPPORTED_LOSS_FUNCTIONS, SUPPORTED_OPTIMIZERS


class Agent(ABC):

    @abstractmethod
    def act(self, x:torch.Tensor) -> int:
        pass

    @abstractmethod
    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        pass

    @abstractmethod
    def setMode(self, eval:bool = False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        pass

    @abstractmethod
    def saveModel(self, fileName: str) -> None:
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
        else:
            raise NotImplemented(f"The Loss function '{loss_name}' is not supported! Please choose another one!")