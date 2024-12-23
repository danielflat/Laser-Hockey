from itertools import count

import numpy as np
import torch

from src.config import DEVICE, HYPERPARAMS, OPTIMIZER, OPTIONS
from src.util.constants import DQN, HUMAN, PENDULUM
from src.util.contract import initAgent, initEnv, initSeed
from src.util.directoryutil import get_path

# USEFUL CONSTANTS
TEST_CHECK_POINT_NAME = "24-12-22 08_15_22"  # which model do you want to test
TEST_ITERATION = "00600"  # Which iteration do you want to test
TEST_USE_ENV = PENDULUM
TEST_USE_ALGO = DQN
TEST_NUMBER_DISCRETE_ACTIONS = 10
TEST_SEED = None
TEST_RENDER_MODE = HUMAN
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEST_ITERATIONS = 10

if __name__ == '__main__':
    initSeed(seed=TEST_SEED, device=DEVICE)
    print(f"Test seed: {TEST_SEED}, Test Device: {TEST_DEVICE}")

    env = initEnv(TEST_USE_ENV, TEST_RENDER_MODE, TEST_NUMBER_DISCRETE_ACTIONS)

    agent = initAgent(use_algo=TEST_USE_ALGO, env=env, options=OPTIONS, optim=OPTIMIZER, hyperparams=HYPERPARAMS,
                      device=DEVICE)
    check_point = get_path(f"output/checkpoints/{TEST_CHECK_POINT_NAME}/{TEST_CHECK_POINT_NAME}_{TEST_ITERATION}.pth")
    agent.setMode(eval=True)
    agent.loadModel(file_name=check_point)

    state, info = env.reset(seed=TEST_SEED)
    env.render()

    episode_steps = []
    episode_rewards = []

    for i_test in range(1, TEST_ITERATIONS + 1):
        total_steps = 0
        total_reward = 0

        for _ in count():
            state = torch.tensor(state, device=DEVICE)
            action = agent.act(state)

            next_step, reward, terminated, truncated, info = env.step(action)
            total_steps += 1
            total_reward += reward

            state = next_step

            done = terminated or truncated

            if done:
                episode_steps.append(total_steps)
                episode_rewards.append(total_reward)
                print(f"Episode: {i_test} | Total steps: {total_steps} | Total reward: {total_reward}")

                state, info = env.reset()
                break
    print(f"Tests done! "
          f"Durations average: {np.array(episode_steps).mean():.4f} | Durations std. dev: {np.array(episode_steps).std():.4f} | Durations variance: {np.array(episode_steps).var():.4f} | "
          f"Reward average: {np.array(episode_rewards).mean():.4f} | Reward std. dev: {np.array(episode_rewards).std():.4f} | Reward variance: {np.array(episode_rewards).var():.4f}")
