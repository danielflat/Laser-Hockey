import torch
from abc import ABC, abstractmethod


class Agent(ABC):

    def __init__(self):
        pass

    # @abstractmethod
    # def forward(self, x: torch.Tensor):
    #     pass

    # def predict(self, x: torch.Tensor):
    #     with torch.no_grad():
    #         return self.forward(x)