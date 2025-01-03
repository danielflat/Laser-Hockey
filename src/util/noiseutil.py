import numpy as np
import torch
from torch import device

from src.util.constants import OU_NOISE, PINK_NOISE, SUPPORTED_NOISE_TYPES, WHITE_NOISE


def initNoise(action_shape, noise_settings: dict, device: device):
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
        """
        Initialize the PinkNoise generator.

        Args:
            shape (tuple): Shape of the output noise (e.g., (num_channels, num_samples)).
            device (str): Device to generate the noise ('cpu' or 'cuda').
        """
        self.shape = shape
        self.device = device
        self.num_samples = shape
        self.num_channels = 1
        # self.num_samples = shape[-1]
        # self.num_channels = shape[0] if len(shape) > 1 else 1

        # Ensure number of samples is even for FFT symmetry
        if self.num_samples % 2 != 0:
            self.num_samples += 1

        # Frequency indices
        self.freqs = torch.fft.rfftfreq(self.num_samples, d = 1.0).to(self.device)

        # Scale by 1/f (avoid division by zero)
        self.scale = 1.0 / torch.sqrt(self.freqs + (self.freqs[1] if self.freqs[1] > 0 else 1e-10))

        # Initialize previous noise state
        self.noise_prev = torch.zeros(self.shape, device = self.device)

    def sample(self):
        """
        Generate a new pink noise sample.

        Returns:
            torch.Tensor: A tensor with the same shape as initialized containing pink noise.
        """
        # Random complex values for FFT coefficients
        real_part = torch.randn(self.num_channels, len(self.freqs), device = self.device)
        imag_part = torch.randn(self.num_channels, len(self.freqs), device = self.device)

        # Combine real and imaginary parts
        fft_coeffs = real_part + 1j * imag_part

        # Apply the 1/f scaling
        fft_coeffs *= self.scale

        # Perform inverse FFT
        pink_noise = torch.fft.irfft(fft_coeffs, n = self.num_samples)

        # Trim to original size if adjusted for evenness
        pink_noise = pink_noise[..., :self.num_samples - 1]

        self.noise_prev = pink_noise
        return pink_noise.squeeze(dim = -1)
