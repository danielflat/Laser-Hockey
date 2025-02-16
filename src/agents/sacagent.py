import logging

import torch
from torch import nn
from torch.distributions import Normal

from src.agent import Agent

"""
WARNING: This is just the inference model for the SAC agent. The original code is on the other repository.
"""


class Actor(nn.Module):
    def __init__(
            self,
            state_dim: int,
            action_dim: int,
            num_layers: int,
            hidden_dim: int,
            epsilon: float = 1e-6
    ):
        super(Actor, self).__init__()

        self.epsilon = epsilon

        # Observation -> Latent State Space
        layers = [
            nn.Flatten(),
            nn.Linear(in_features=state_dim, out_features=hidden_dim),
            nn.ReLU()
        ]

        for _ in range(num_layers):
            layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
            layers.append(nn.ReLU())

        self.actor_latent = nn.Sequential(*layers)

        # Latent State Space -> Action Space as Diagonal Normal
        self.actor_mu = nn.Linear(in_features=hidden_dim, out_features=action_dim)
        self.actor_log_std = nn.Linear(in_features=hidden_dim, out_features=action_dim)

    def _sample(self, mu, log_std):
        normal_dist = Normal(mu, log_std.exp())

        # Sample with reparametrization trick
        z = normal_dist.rsample()

        # Calculate log probability along each (independent) dimension
        log_prob = normal_dist.log_prob(z)

        # Calculate log probability of diagonal normal
        log_prob = normal_dist.log_prob(z).sum(dim=-1)

        return z, log_prob

    def _squash(self, action, log_prob):
        action = torch.tanh(action)
        # Squash Correction from original paper
        log_prob -= torch.sum(torch.log(1 - action ** 2 + self.epsilon), dim=1)
        return action, log_prob

    def forward(self, x, deterministic=False):
        latent = self.actor_latent(x)

        mu = self.actor_mu(latent)

        log_std = self.actor_log_std(latent).clamp(min=-20, max=2)

        action, log_prob = self._sample(mu, log_std)

        # Squash to [-1, 1]
        if deterministic:
            return self._squash(mu, log_prob)[0]
        action, log_prob = self._squash(action, log_prob)

        return action, log_prob


class SACAgent(Agent):
    def __init__(self, checkpoint_name: str, agent_settings: dict, device: torch.device):
        super().__init__(agent_settings, device)

        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        state_dict = checkpoint["state_dict"]
        logging.info(f"Model for {self.__repr__()} loaded successfully from {checkpoint_name}")
        self.actor = Actor(state_dim=18, action_dim=4, num_layers=1, hidden_dim=256).to(self.device)
        actor_state_dict = {k.replace("actor.", ""): v for k, v in state_dict.items() if k.startswith("actor.")}
        self.actor.load_state_dict(actor_state_dict)

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"SACAgent"

    @torch.no_grad()
    def act(self, observation):
        return self.actor(observation.unsqueeze(0).to(self.device), deterministic=True).flatten().cpu().numpy()

    def reset(self):
        pass

    def optimize(self, memory, episode_i):
        raise NotImplementedError

    def setMode(self, eval=False):
        pass

    def export_checkpoint(self) -> dict:
        raise NotImplementedError

    def import_checkpoint(self, checkpoint: dict) -> None:
        raise NotImplementedError
