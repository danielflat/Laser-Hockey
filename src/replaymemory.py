import random
from collections import deque
from typing import Any

import torch
from torch import Tensor


class ReplayMemory:
    def __init__(self, capacity) -> None:
        """
        Initialize the replay memory with a given capacity.
        :param capacity: Maximum number of transitions to store.
        """
        self.storage = deque([], maxlen=capacity)

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
             next_state: torch.Tensor, done: Tensor, info: dict) -> None:
        """
        Save a transition in the replay memory
        """
        self.storage.append((state, action, reward, next_state, done, info))

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[Any]]:
        """
        Sample a batch of transitions from the memory.
        """
        if batch_size > len(self.storage):
            raise ValueError("The batch size is larger than the memory elements.")

        batch = random.sample(self.storage, batch_size)
        states, actions, rewards, next_states, dones, infos = zip(*batch)

        # Convert them to the right shapes
        states = torch.vstack(states)
        actions = torch.vstack(actions)
        rewards = torch.vstack(rewards)
        next_states = torch.vstack(next_states)
        dones = torch.vstack(dones)
        infos = list(infos)

        return states, actions, rewards, next_states, dones, infos



    def __len__(self) -> int:
        """
        Return the current size of the memory
        """
        return len(self.storage)

    def clear(self) -> None:
        """
        Clears all transitions from the memory
        """
        self.storage.clear()
