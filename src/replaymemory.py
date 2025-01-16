from collections import deque

import logging
import random
import torch
from torch import Tensor
from typing import Any, List, Tuple


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

    def sample_horizon(self, batch_size: int, horizon: int = 0) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Samples a trajectory of a random episode from the memory.
        This trajectory starts at a random position in the episode and goes #horizon steps in the future
        returns the tuple of the horizon trajectory.

        CAUTION: if horizon = 0, the output shape of each object is (batch_size, object_size)
        :returns
            e.g. if horizon = 3,
            horizon_states tensor(batch_size, horizon + 1, state_size)
            horizon_actions tensor(batch_size, horizon + 1, action_size)
            horizon_rewards tensor(batch_size, horizon + 1, 1)
            horizon_dones tensor(batch_size, horizon + 1, 1)
            horizon_next_state tensor(batch_size, horizon + 1, state_size)
            horizon_infos list(batch_size, horizon + 1, info_length) -> not available yet
        """
        assert horizon >= 0, "Horizon must be >= 0!"

        # Step 01: We sample a random episode with replacement *for each batch*
        batches = random.choices(self.storage, k = batch_size)
        states, actions, rewards, next_states, dones, infos = zip(*batches)
        episode_lengths = torch.tensor([tensor.size(0) for tensor in states])


        # Step 04: We sample a random number to get the start_index of the horizon
        start_indices = [random.randint(0, episode_length.item() - horizon - 1) for episode_length in episode_lengths]
        slices = [slice(start_ind, start_ind + horizon + 1) for start_ind in start_indices]

        # Step 05: We only want the horizon of the episode
        horizon_states = torch.stack([_state[_slice] for _state, _slice in zip(states, slices)])
        horizon_actions = torch.stack([_action[_slice] for _action, _slice in zip(actions, slices)])
        horizon_rewards = torch.stack([_reward[_slice] for _reward, _slice in zip(rewards, slices)]).unsqueeze(-1)
        horizon_next_states = torch.stack([_next_state[_slice] for _next_state, _slice in zip(next_states, slices)])
        horizon_dones = torch.stack([done[_slice] for done, _slice in zip(dones, slices)]).unsqueeze(-1)
        # horizon_infos = infos[:, start_indices]

        # # Step 06 (Optional): If the horizon = 0, we can get rid of the second dim
        # # TODO: df: Can be implemented better, but it works for now
        # if horizon == 0:
        #     horizon_states = horizon_states.squeeze(1)
        #     horizon_actions = horizon_actions.squeeze(1)
        #     horizon_rewards = horizon_rewards.squeeze(1)
        #     horizon_next_states = horizon_next_states.squeeze(1)
        #     horizon_dones = horizon_dones.squeeze(1)


        return horizon_states, horizon_actions, horizon_rewards, horizon_next_states, horizon_dones  # , horizon_infos



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
