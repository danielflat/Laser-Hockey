import torch

from src.agent import Agent
from src.replaymemory import ReplayMemory


class RandomAgent(Agent):
    def __init__(self, env, agent_settings: dict, device: torch.device = None):
        # we don't need the agent_settings and the device here. We just call it here to prevent a warning
        super().__init__(agent_settings, device)

        self.env = env

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"RandomAgent"

    def act(self, state):
        return self.env.action_space.sample()

    def optimize(self, memory: ReplayMemory, episode_i: int) -> None:
        # nothing to do
        return None

    def setMode(self, eval = False) -> None:
        # nothing to do
        pass

    def saveModel(self, model_name: str, iteration: int) -> None:
        raise NotImplementedError

    def loadModel(self, file_name: str) -> None:
        raise NotImplementedError

    def import_checkpoint(self, checkpoint: dict) -> None:
        raise NotImplementedError

    def export_checkpoint(self) -> dict:
        raise NotImplementedError

    def reset(self):
        pass
