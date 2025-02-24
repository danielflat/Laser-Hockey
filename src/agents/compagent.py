import numpy as np
import torch

from hockey.hockey_env import BasicOpponent
from src.agent import Agent
from src.replaymemory import ReplayMemory


class CompAgent(Agent):
    def __init__(self, is_Weak, agent_settings: dict, device: torch.device = None):
        # we don't need the agent_settings and the device here. We just call it here to prevent a warning
        super().__init__(agent_settings, device)

        self.is_Weak = is_Weak
        self.basic_opponent = BasicOpponent(weak = self.is_Weak)

    def __repr__(self):
        """
        For printing purposes only
        """
        return "WeakAgent" if self.is_Weak else "StrongAgent"

    def act(self, state: torch.Tensor) -> np.ndarray:
        return self.basic_opponent.act(state.cpu().numpy())

    def optimize(self, memory: ReplayMemory, episode_i: int) -> None:
        # nothing to do
        return None

    def setMode(self, eval = False) -> None:
        # nothing to do
        pass

    def saveModel(self, model_name: str, iteration: int) -> None:
        raise NotImplementedError

    def loadModel(self, file_name: str) -> None:
        pass

    def import_checkpoint(self, checkpoint: dict) -> None:
        raise NotImplementedError

    def export_checkpoint(self) -> dict:
        raise NotImplementedError

    def reset(self):
        pass
