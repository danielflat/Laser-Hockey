import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# Define the Policy Network
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.fc(x)


# Custom HLGaussLoss (Already provided by you)
class HLGaussLoss(nn.Module):
    def __init__(self, min_value: float, max_value: float, num_bins: int, sigma: float):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.num_bins = num_bins
        self.sigma = sigma
        self.support = torch.linspace(min_value, max_value, num_bins + 1, dtype = torch.float32)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, self.transform_to_probs(target))

    def transform_to_probs(self, target: torch.Tensor) -> torch.Tensor:
        cdf_evals = torch.special.erf(
            (self.support - target.unsqueeze(-1)) / (torch.sqrt(torch.tensor(2.0)) * self.sigma)
        )
        z = cdf_evals[..., -1] - cdf_evals[..., 0]
        bin_probs = cdf_evals[..., 1:] - cdf_evals[..., :-1]
        return bin_probs / z.unsqueeze(-1)

    def transform_from_probs(self, probs: torch.Tensor) -> torch.Tensor:
        centers = (self.support[:-1] + self.support[1:]) / 2
        return torch.sum(probs * centers, dim = -1)


# Initialize Environment and Hyperparameters
env = gym.make('Pendulum-v1')
state_dim = env.observation_space.shape[0]
action_bins = 10  # Number of bins for action space
action_min = env.action_space.low[0]
action_max = env.action_space.high[0]

policy = PolicyNetwork(state_dim, action_bins)
hl_loss = HLGaussLoss(action_min, action_max, action_bins, sigma = 0.1)
optimizer = optim.Adam(policy.parameters(), lr = 1e-3)

# Training Loop
num_episodes = 500
for episode in range(num_episodes):
    state, _ = env.reset()
    state = torch.tensor(state, dtype = torch.float32)
    total_reward = 0
    total_loss = 0

    for t in range(200):  # Maximum timesteps in an episode
        logits = policy(state)  # Get logits for bins
        action_probs = F.softmax(logits, dim = -1)
        action_dist = Categorical(action_probs)
        action_bin = action_dist.sample()  # Sample an action bin
        action = hl_loss.transform_from_probs(action_probs)

        next_state, reward, done, _, _ = env.step([action.item()])
        target_action = torch.tensor([action.item()], dtype = torch.float32)

        total_reward += reward
        loss = hl_loss(logits.unsqueeze(0), target_action)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        state = torch.tensor(next_state, dtype = torch.float32)

        if done:
            break

    print(f"Episode {episode + 1}, Loss: {total_loss:.3f}, Total Reward: {total_reward:.3f}")

env.close()
