import gymnasium as gym
import numpy as np


class SelfPlayEnv(gym.Env):
    def __init__(self, ):
        super(SelfPlayEnv, self).__init__()
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space  # Agent 1 actions
        self.observation_space = self.env.observation_space

        # Disturbance action space for Agent 2
        self.disturbance_space = gym.spaces.Discrete(3)  # Left, None, Right

    def reset(self, seed = None, options = None):
        obs, _ = self.env.reset(seed = seed, options = options)
        return obs, {}

    def step(self, action):
        # Agent 1 action
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Agent 2 disturbance (random force)
        disturbance = np.random.choice([-1, 0, 1])  # Apply random force
        self.env.unwrapped.state[1] += disturbance * 0.01  # Modify velocity

        # Adjust reward for self-play scenario
        reward -= abs(disturbance * 0.1)  # Penalize disturbances

        return obs, reward, terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
