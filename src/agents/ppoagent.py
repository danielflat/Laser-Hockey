from src.agent import Agent
import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy_Net():
    def __init__(self, observation_size, action_size, hidden_size = 128):
        actor = torch.nn.Sequential(
            torch.nn.Linear(observation_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, action_size)
        )

        critic = torch.nn.Sequential(
            torch.nn.Linear(observation_size + action_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1)
        )


class PPOAgent(Agent):
    def __init__(self, observation_size, action_size):
        super().__init__()
        policy_net = Policy_Net()
        target_net = Policy_Net()