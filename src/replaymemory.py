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

    def sample_horizon(self, horizon: bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, List[Any]]:
        """
        Samples a trajectory of a random episode from the memory.
        This trajectory starts at a random position in the episode and goes #horizon steps in the future
        returns the tuple of the horizon trajectory.
        :returns
            e.g. if horizon = 3,
            horizon_states tensor(horizon + 1, state_size)
            horizon_actions tensor(horizon + 1, action_size)
            horizon_rewards tensor(horizon + 1, 1)
            horizon_dones tensor(horizon + 1, 1)
            horizon_next_state tensor(horizon + 1, state_size)
            horizon_infos list(horizon + 1, info_length)
        """

        # Step 01: We sample a random episode
        batch = random.sample(self.storage, 1)[0]
        states, actions, rewards, next_states, dones, infos = batch
        episode_length = states.shape[0]

        # # Step 02: If the episode is smaller than the required batch size, throw an error
        # if batch_size > episode_length:
        #     raise Exception("Required batch size is larger than the episode length!")

        # Step 03: We sample a random number to get the start_index of the horizon
        start_index = torch.randint(0, episode_length - horizon, (1,), device = self.device)

        # Step 04: We only want the horizon of the episode
        horizon_states = states[start_index:start_index + horizon + 1]
        horizon_actions = actions[start_index:start_index + horizon + 1]
        horizon_rewards = rewards[start_index:start_index + horizon + 1].unsqueeze(-1)
        horizon_next_states = next_states[start_index:start_index + horizon + 1]
        horizon_dones = dones[start_index:start_index + horizon + 1].unsqueeze(-1)
        horizon_infos = infos[start_index:start_index + horizon + 1]

        return horizon_states, horizon_actions, horizon_rewards, horizon_next_states, horizon_dones, horizon_infos



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
