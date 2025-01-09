from typing import Tuple

import numpy as np
import torch
from torch import device

from src.util.constants import OU_NOISE, PINK_NOISE, SUPPORTED_NOISE_TYPES, WHITE_NOISE


def initNoise(action_shape: Tuple[int], noise_settings: dict, device: device):
    noise_type = noise_settings["NOISE_TYPE"]
    noise_params = noise_settings["NOISE_PARAMS"]

    if noise_type not in SUPPORTED_NOISE_TYPES:
        raise Exception(f"Unsupported noise type: {noise_type}!")

    if noise_type == WHITE_NOISE:
        return WhiteNoise(action_shape, noise_params["MEAN"], noise_params["STD"], device)
    elif noise_type == OU_NOISE:
        return OUNoise(action_shape, noise_params["THETA"], noise_params["DT"], device)
    elif noise_type == PINK_NOISE:
        return PinkNoise(action_shape, device)
    else:
        raise Exception("Unknown noise type: {}".format(noise_type))

class OUNoise:
    def __init__(self, shape, theta: float, dt: float, device: device):
        self._shape = shape
        self._theta = theta
        self._dt = dt
        self.device = device
        self.noise_prev = np.zeros(self._shape)
        self.reset()

    def sample(self) -> torch.Tensor:
        noise = (
                self.noise_prev
                + self._theta * (- self.noise_prev) * self._dt
                + np.sqrt(self._dt) * np.random.normal(size = self._shape)
        )
        self.noise_prev = noise
        return torch.from_numpy(noise).to(self.device)

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)


class WhiteNoise:
    def __init__(self, shape, mean: float, std: float, device: device):
        self._shape = shape
        self._mean = mean
        self._std = std
        self.device = device

    def sample(self) -> torch.Tensor:
        return torch.from_numpy(np.random.normal(loc = self._mean, scale = self._std, size = self._shape)).to(
            self.device)


class PinkNoise:
    def __init__(self, shape, device: device):
        self.shape = shape
        self.device = device

        # Is used to create the pink noise
        self.prev_values = np.zeros((16,) + self.shape)

    def sample(self) -> np.ndarray:
        """
        Generate one sample of pink noise using a recursive filter.

        Returns:
        - A single pink noise sample.
        """

        white_sample = np.random.normal(loc = 0, scale = 1, size = self.shape)
        self.prev_values[1:] = self.prev_values[:-1]
        self.prev_values[0] = white_sample

        pink_sample = (self.prev_values / np.expand_dims(1 + np.arange(16), axis = -1)).sum(axis = 0)
        return pink_sample / 4  # Normalization factor
