import copy

import time

from itertools import count

import numpy as np
import torch

from src.agent import Agent
from src.settings import AGENT_SETTINGS, DQN_SETTINGS, PPO_SETTINGS, MPO_SETTINGS, TD_MPC2_SETTINGS
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, HUMAN, PENDULUM, HALFCHEETAH, TDMPC2_ALGO
from src.util.contract import initAgent, initEnv, initSeed
from src.util.directoryutil import get_path

import hockey.hockey_env as h_env

"""
This is a test class with which you can sanity check your agent's performances.
The pendulum and the hockey environment are supported.
You can go and adjust the settings below to test your agent's performance.

Author: Daniel Flat
"""

# Example usage of these setting parameters:
# Example 01: Hockey
# TEST_CHECK_POINT_NAME = get_path("good_checkpoints/hockey_ddpg_25-01-22 17_36_56_05000.pth")
# TEST_USE_ENV = HOCKEY
# TEST_USE_ALGO = DDPG_ALGO
# TEST_MODE = "human" # "human", "strong" or "self"
# TEST_NUMBER_DISCRETE_ACTIONS = None
# TEST_SEED = 5
# TEST_RENDER_MODE = HUMAN  # None or HUMAN
# TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TEST_ITERATIONS = 10

# EXAMPLE 02: Pendulum
# TEST_CHECK_POINT_NAME = get_path("output/checkpoints/25-01-21 18_37_28/25-01-21 18_37_28_00100.pth")
# TEST_USE_ENV = PENDULUM
# TEST_USE_ALGO = DDPG_ALGO
# TEST_MODE = "human" # "human", "strong" or "self"
# TEST_NUMBER_DISCRETE_ACTIONS = None
# TEST_SEED = 5
# TEST_RENDER_MODE = HUMAN  # None or HUMAN
# TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TEST_ITERATIONS = 10

# output/checkpoints/25-01-21 18_37_28/25-01-21 18_37_28_00100.pth pendulum with DDPG
# output/checkpoints/25-01-22 12_04_21/25-01-22 12_04_21_000000280.pth pendulum with TDMPC2

# USEFUL CONSTANTS
TEST_CHECK_POINT_NAME = get_path(
    "output/checkpoints/25-01-22 12_04_21/25-01-22 12_04_21_000000280.pth")  # Which checkpoint do you want to test
TEST_USE_ENV = PENDULUM  # On which environment do you want to test?
TEST_USE_ALGO = TDMPC2_ALGO  # Which algorithm do you want to test?
TEST_MODE = "human"  # HOCKEY ONLY: Against which agent do you want to play? "human", "strong" or "self" are supported
TEST_NUMBER_DISCRETE_ACTIONS = None  # if you want to use discrete actions or continuous. If > 0, you use the DiscreteActionWrapper
TEST_SEED = 100000  # Set a test seed if you want to
TEST_RENDER_MODE = HUMAN  # For whom do you want to render? None or HUMAN
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # which device are you using?
TEST_ITERATIONS = 10  # The number of test iterations

if __name__ == '__main__':
    initSeed(seed = TEST_SEED, device = TEST_DEVICE)
    print(f"Test seed: {TEST_SEED}, Test Device: {TEST_DEVICE}")

    env = initEnv(TEST_USE_ENV, TEST_RENDER_MODE, TEST_NUMBER_DISCRETE_ACTIONS)

    agent = initAgent(use_algo = TEST_USE_ALGO, env = env, agent_settings = AGENT_SETTINGS, device = TEST_DEVICE,
                      checkpoint_name = TEST_CHECK_POINT_NAME)
    agent.setMode(eval = True)

    if TEST_USE_ENV == HOCKEY and TEST_MODE == "human":
        player1 = h_env.HumanOpponent(env = env, player = 1)
        player2 = agent
    elif TEST_USE_ENV == HOCKEY and TEST_MODE == "strong":
        player1 = agent
        player2 = h_env.BasicOpponent()
    elif TEST_USE_ENV == HOCKEY and TEST_MODE == "self":
        player1 = agent
        player2 = copy.deepcopy(agent)
    elif TEST_USE_ENV == PENDULUM:
        player1 = agent
    else:
        raise NotImplementedError

    episode_steps = []
    episode_rewards = []
    for i_test in range(1, TEST_ITERATIONS + 1):
        t_start = time.time()
        total_steps = 0
        total_reward = 0

        state, info = env.reset(seed = TEST_SEED + i_test)
        env.render()
        if TEST_USE_ENV == HOCKEY:
            state2 = env.obs_agent_two()

        for _ in count():
            if isinstance(player1, Agent):
                state = torch.tensor(state, device = TEST_DEVICE, dtype = torch.float32)

            if TEST_USE_ENV == HOCKEY and isinstance(player2, Agent):
                state2 = torch.tensor(state2, device = TEST_DEVICE, dtype = torch.float32)
            action1 = player1.act(state)
            if TEST_USE_ENV == HOCKEY:
                action2 = player2.act(state2)
                next_step, reward, terminated, truncated, info = env.step(np.hstack([action1, action2]))
            else:
                next_step, reward, terminated, truncated, info = env.step(action1)
            total_steps += 1
            total_reward += reward

            state = next_step
            if TEST_USE_ENV == HOCKEY:
                state2 = env.obs_agent_two()
            done = terminated or truncated

            if done:
                episode_steps.append(total_steps)
                episode_rewards.append(total_reward)
                t_end = time.time()
                t_required = t_end - t_start
                print(
                    f"Episode: {i_test} | Total steps: {total_steps} | Total reward: {total_reward} | Req. Time: {t_required:.4} sec.")
                break
    print(f"Tests done! "
          f"Durations average: {np.array(episode_steps).mean():.4f} | Durations std. dev: {np.array(episode_steps).std():.4f} | Durations variance: {np.array(episode_steps).var():.4f} | "
          f"Reward average: {np.array(episode_rewards).mean():.4f} | Reward std. dev: {np.array(episode_rewards).std():.4f} | Reward variance: {np.array(episode_rewards).var():.4f}")
