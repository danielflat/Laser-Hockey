from itertools import count

import numpy as np
import torch

from src.settings import AGENT_SETTINGS, DQN_SETTINGS, PPO_SETTINGS, MPO_SETTINGS
from src.util.constants import DQN_ALGO, MPO_ALGO, HUMAN, PENDULUM, HALFCHEETAH
from src.util.contract import initAgent, initEnv, initSeed
from src.util.directoryutil import get_path

# USEFUL CONSTANTS
TEST_CHECK_POINT_NAME = "25-01-15 19_04_37"  # which model do you want to test
TEST_ITERATION = "02000"  # Which iteration do you want to test
TEST_USE_ENV = HALFCHEETAH
TEST_USE_ALGO = MPO_ALGO
TEST_NUMBER_DISCRETE_ACTIONS = None
TEST_SEED = 5
TEST_RENDER_MODE = HUMAN  # None or HUMAN
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_ITERATIONS = 1000

if __name__ == '__main__':
    initSeed(seed = TEST_SEED, device = TEST_DEVICE)
    print(f"Test seed: {TEST_SEED}, Test Device: {TEST_DEVICE}")

    env = initEnv(TEST_USE_ENV, TEST_RENDER_MODE, TEST_NUMBER_DISCRETE_ACTIONS)

    agent = initAgent(use_algo = TEST_USE_ALGO, env = env, agent_settings = AGENT_SETTINGS, dqn_settings = DQN_SETTINGS,
                      ppo_settings = PPO_SETTINGS, device = TEST_DEVICE)
    check_point = get_path(f"output/checkpoints/{TEST_CHECK_POINT_NAME}/{TEST_CHECK_POINT_NAME}_{TEST_ITERATION}.pth")
    agent.setMode(eval = True)
    agent.loadModel(file_name = check_point)

    state, info = env.reset(seed = TEST_SEED)
    env.render()

    episode_steps = []
    episode_rewards = []
    for i_test in range(1, TEST_ITERATIONS + 1):
        total_steps = 0
        total_reward = 0

        for _ in count():
            state = torch.tensor(state, device = TEST_DEVICE, dtype = torch.float32)
            action = agent.act(state)

            next_step, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            total_reward += reward
            
            action = torch.tensor(action, device = TEST_DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = TEST_DEVICE, dtype = torch.float32)
            terminated = torch.tensor(terminated, device = TEST_DEVICE, dtype = torch.float32)
            truncated = torch.tensor(truncated, device = TEST_DEVICE, dtype = torch.float32)
            next_step = torch.from_numpy(next_step).to(dtype=torch.float32, device = TEST_DEVICE)

            state = next_step

            done = terminated or truncated

            if done:
                episode_steps.append(total_steps)
                episode_rewards.append(total_reward)
                print(f"Episode: {i_test} | Total steps: {total_steps} | Total reward: {total_reward}")

                state, info = env.reset(seed = TEST_SEED + i_test)  # for reproducibility
                break
    print(f"Tests done! "
          f"Durations average: {np.array(episode_steps).mean():.4f} | Durations std. dev: {np.array(episode_steps).std():.4f} | Durations variance: {np.array(episode_steps).var():.4f} | "
          f"Reward average: {np.array(episode_rewards).mean():.4f} | Reward std. dev: {np.array(episode_rewards).std():.4f} | Reward variance: {np.array(episode_rewards).var():.4f}")
