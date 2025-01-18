import gymnasium
import logging
import numpy as np
import os
import torch
from abc import ABC, abstractmethod
from torch import nn
from typing import List

from src.replaymemory import ReplayMemory
from src.util.constants import ADAM, ADAMW, CROSS_ENTROPY_LOSS, EXPONENTIAL, L1_LOSS, LINEAR, MSE_LOSS, SMOOTH_L1_LOSS, \
    SUPPORTED_LOSS_FUNCTIONS, \
    SUPPORTED_OPTIMIZERS
from src.util.directoryutil import get_path


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
        self.norm_clipping_value = agent_settings["NORM_CLIPPING_VALUE"]
        self.target_net_update_freq = agent_settings["TARGET_NET_UPDATE_FREQ"]
        self.tau = agent_settings["TAU"]

        # Options
        self.epsilon_decay_strategy = agent_settings["EPSILON_DECAY_STRATEGY"]
        self.use_target_net = agent_settings["USE_TARGET_NET"]
        self.use_soft_updates = agent_settings["USE_SOFT_UPDATES"]
        self.use_gradient_clipping = agent_settings["USE_GRADIENT_CLIPPING"]
        self.use_norm_clipping = agent_settings["USE_NORM_CLIPPING"]
        self.epsilon_decay_strategy = agent_settings["EPSILON_DECAY_STRATEGY"]
        self.device: device = device
        self.use_clip_foreach = agent_settings["USE_CLIP_FOREACH"]
        self.USE_BF_16 = agent_settings["USE_BF16"]
        self.USE_COMPILE = agent_settings["USE_COMPILE"]

    @abstractmethod
    def act(self, x: torch.Tensor) -> int:
        pass

    @abstractmethod
    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        pass

    @abstractmethod
    def setMode(self, eval: bool = False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        pass

    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """

        checkpoint = self.export_checkpoint()

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:09}.pth")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok = True)

        torch.save(checkpoint, file_path)
        logging.info(f"Training Iter: {iteration}: Checkpoint of {self.__repr__()} saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        try:
            checkpoint = torch.load(file_name, map_location = self.device)
            self.import_checkpoint(checkpoint)
            logging.info(f"Model for {self.__repr__()} loaded successfully from {file_name}")
        except FileNotFoundError:
            logging.error(f"Error: File {file_name} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {str(e)}")

    @abstractmethod
    def import_checkpoint(self, checkpoint: dict) -> None:
        pass

    @abstractmethod
    def export_checkpoint(self) -> dict:
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

    def initOptim(self, optim: dict, parameters, disable_weight_decay: bool = False) -> torch.optim:
        """
        Initialize the optimizer based on the config.py
        :param optim: the optim config
        :param parameters: the parameters of the network to optimize
        :return: the optimizer
        """
        optim_name = optim["OPTIM_NAME"]
        weight_decay = optim["WEIGHT_DECAY"] if not disable_weight_decay else 0.0
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
            if loss_name == L1_LOSS:
                return nn.L1Loss()
            elif loss_name == SMOOTH_L1_LOSS:
                return nn.SmoothL1Loss()
            elif loss_name == MSE_LOSS:
                return nn.MSELoss()
            elif loss_name == CROSS_ENTROPY_LOSS:
                return nn.CrossEntropyLoss()
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

    def get_num_actions(self, action_space):
        if type(action_space) == gymnasium.spaces.box.Box:
            action_size = action_space.shape[0]
        elif type(action_space) == gymnasium.spaces.discrete.Discrete:
            action_size = 1
        else:
            action_size = action_space.n

        # In Hockey, the action size is 8, but we have 2 players -> Therefore the *real* action size is 4
        if action_size == 8:
            return 4
        return action_size
