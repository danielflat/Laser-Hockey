from collections import deque
import random
import numpy as np
import torch

class ReplayMemory:
    def __init__(self, capacity, device: torch.device) -> None:
        """
        Initialize the replay memory with a given capacity.
        :param capacity: Maximum number of transitions to store.
        """
        self.storage = deque([], maxlen=capacity)
        self.device = device

    def push(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float,
        next_state: np.ndarray, 
        done: bool, 
        info: dict
    ) -> None:
        """
        Save a single-step transition in the replay memory as NumPy + scalars.
        
        state, action, next_state: typically shape (obs_dim,) or (action_dim,)
        reward: float
        done: bool
        info: dict
        """
        self.storage.append((state, action, reward, next_state, done, info))

    def push_batch(self, transitions_list):
        """
        Optionally push a batch of transitions at once.
        Each element in transitions_list is a tuple:
           (state, action, reward, next_state, done, info)
        This can reduce Python function-call overhead.
        """
        for item in transitions_list:
            self.storage.append(item)

    def sample(self, batch_size: int, randomly: bool = True):
        """
        Sample a batch of transitions from the memory and convert them to torch Tensors.
        
        Returns (states, actions, rewards, next_states, dones, infos)
        where states, actions, next_states = float32 Torch tensors
              rewards, dones = float32 Torch tensors (with shape [batch_size, 1])
              infos = list of dicts (unchanged)
        """
        if batch_size > len(self.storage):
            batch_size = len(self.storage)

            # Or you can raise an Exception if you prefer.
            # raise ValueError("Batch size is bigger than memory!")

        if randomly:
            batch = random.sample(self.storage, batch_size)
        else:
            batch = [self.storage[i] for i in range(batch_size)]

        # Unzip
        states, actions, rewards, next_states, dones, infos = zip(*batch)

        # Convert to NumPy
        # states is a tuple of shape (batch_size,) each with shape (obs_dim,)
        states_np     = np.stack(states).astype(np.float32)      # shape (batch_size, obs_dim)
        actions_np    = np.stack(actions).astype(np.float32)     # shape (batch_size, act_dim)
        rewards_np    = np.array(rewards, dtype=np.float32).reshape(-1, 1)
        next_states_np= np.stack(next_states).astype(np.float32)
        dones_np      = np.array(dones, dtype=np.float32).reshape(-1, 1)

        # Now convert to torch
        states_t      = torch.from_numpy(states_np).to(self.device)
        actions_t     = torch.from_numpy(actions_np).to(self.device)
        rewards_t     = torch.from_numpy(rewards_np).to(self.device)
        next_states_t = torch.from_numpy(next_states_np).to(self.device)
        dones_t       = torch.from_numpy(dones_np).to(self.device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t, list(infos)

    """
    # NOTE: The original sample_horizon() expected states to be entire episodes (T x state_dim).
    # Storing single-step transitions in self.storage breaks that approach.
    # If you truly need horizon-based sampling, you must adapt the data structure
    # to store entire episodes or partial trajectories rather than single steps.

    def sample_horizon(self, batch_size: int, horizon: int = 0):
        raise NotImplementedError(
            "sample_horizon() is not compatible with single-step transitions. "
            "Either store entire episodes in memory or remove horizon sampling."
        )
    """

    def __len__(self) -> int:
        return len(self.storage)

    def clear(self) -> None:
        self.storage.clear()