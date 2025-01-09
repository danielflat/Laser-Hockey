import logging
import random
from collections import deque
from typing import Any, List, Tuple

import torch
from torch import Tensor


class ReplayMemory:
    def __init__(self, capacity, device: torch.device) -> None:
        """
        Initialize the replay memory with a given capacity.
        :param capacity: Maximum number of transitions to store.
        """
        self.storage = deque([], maxlen=capacity)
        self.device = device

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
             next_state: torch.Tensor, done: Tensor, info: dict) -> None:
        """
        Save a transition in the replay memory
        """
        self.storage.append((state, action, reward, next_state, done, info))

    def sample(self, batch_size: int, randomly: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[Any]]:
        """
        Sample a batch of transitions from the memory.
        """

        # If there are not enough elements in the memory, take all elements of the storage for now
        if batch_size > len(self.storage):
            batch_size = len(self.storage)
            logging.warning("The batch size was larger than the memory elements!")

        if randomly:
            # Random batch
            batch = random.sample(self.storage, batch_size)
        else:
            # Sequential batch
            batch = [self.storage[i] for i in range(batch_size)]
        states, actions, rewards, next_states, dones, infos = zip(*batch)

        # Convert them to the right shapes
        states = torch.vstack(states).to(self.device)
        actions = torch.vstack(actions).to(self.device)
        rewards = torch.vstack(rewards).to(self.device)
        next_states = torch.vstack(next_states).to(self.device)
        dones = torch.vstack(dones).to(self.device)
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
