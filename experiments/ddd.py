import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO

# Create Gym environment
env = gym.make("CartPole-v1")


# Define self-play logic
class SelfPlayEnv(gym.Env):
    def __init__(self):
        super(SelfPlayEnv, self).__init__()
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space  # Agent 1 actions
        self.observation_space = self.env.observation_space

        # Disturbance action space for Agent 2
        self.disturbance_space = gym.spaces.Discrete(3)  # Left, None, Right

    # def reset(self):
    #     return self.env.reset()

    def step(self, action):
        # Agent 1 action
        obs, reward, done, info = self.env.step(action)

        # Agent 2 disturbance
        disturbance = np.random.choice([-1, 0, 1])  # Apply random force
        self.env.env.state[1] += disturbance * 0.01  # Modify velocity

        # Adjust reward for self-play scenario
        reward -= abs(disturbance * 0.1)  # Penalize disturbances
        return obs, reward, done, info

    def render(self, mode = "human"):
        return self.env.render()

    def close(self):
        self.env.close()


# Initialize self-play environment
self_play_env = SelfPlayEnv()

# Train PPO on the self-play environment
model = PPO("MlpPolicy", self_play_env, verbose = 1)
model.learn(total_timesteps = 50000, progress_bar = True)

# Evaluate and visualize the results
obs, info = self_play_env.reset()
rewards = []
for _ in range(200):
    action, _ = model.predict(obs)
    obs, reward, terminated, truncated, _ = self_play_env.step(action)
    done = terminated or truncated
    rewards.append(reward)
    if done:
        obs, info = self_play_env.reset()

plt.plot(rewards)
plt.title("Rewards Over Time")
plt.xlabel("Steps")
plt.ylabel("Reward")
plt.show()
