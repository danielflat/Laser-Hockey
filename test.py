import copy

import time

from itertools import count

import numpy as np
import torch

from src.agent import Agent
from src.settings import AGENT_SETTINGS, DQN_SETTINGS, PPO_SETTINGS, MPO_SETTINGS, TD_MPC2_SETTINGS
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, HUMAN, PENDULUM, HALFCHEETAH, RANDOM_ALGO, \
    STRONG_COMP_ALGO, \
    TDMPC2_ALGO
from src.util.contract import initAgent, initEnv, initSeed
from src.util.directoryutil import get_path

import hockey.hockey_env as h_env

"""
This is a test class with which you can sanity check your agent's checkpoints against other agents, e.g. against a human agent, a strong agent 
or against another algorithmic agent.
The pendulum and the hockey environment are supported.
You can go and adjust the settings below to test your agent's performance.

In the following, we will document the settings variables in order to use this test class.

Since the pendulum is a one player game, you only need to set the settings for player 1.
For Hockey, you have to set variables for both players.

With `TEST_CHECK_POINT_NAME_PLAYER_*`, you can set the path of the checkpoint '.pth' file in order to load the model for this agent.
An example for this can be:
TEST_CHECK_POINT_NAME_PLAYER_1 = get_path("good_checkpoints/hockey_tdmpc2_bad_25-01-22 15_05_13_000000060.pth")

The same can be done for player 2. It is also possible to set the same checkpoint for both players the same way.

`TEST_USE_ENV` is either HOCHEY or PENDULUM to set the test environment.

`TEST_USE_ALGO_PLAYER_*` sets the algorithm for this player. It can be either "human" or the name for the algorithm. For that, check out constants.py `SUPPORTED_ALGORITHMS`

TEST_NUMBER_DISCRETE_ACTIONS is deprecated now. But it can be used to discretize the continous action space into n bins.

TEST_RENDER_MODE = HUMAN  # For whom do you want to render? None or HUMAN

TEST_DEVICE, TEST_SEED and TEST_ITERATIONS should be clear to understand.

Author: Daniel Flat
"""

# Settings for this class
TEST_CHECK_POINT_NAME_PLAYER_1 = get_path(
    "good_checkpoints/hockey_mpo_disc_25-01-31 20_20_14_18000.pth")  # Which checkpoint do you want to test
TEST_CHECK_POINT_NAME_PLAYER_2 = get_path(
    "good_checkpoints/hockey_tdmpc2_bad_25-01-22 15_05_13_000000060.pth")  # Which checkpoint do you want to test
TEST_USE_ENV = HOCKEY  # On which environment do you want to test?
TEST_USE_ALGO_PLAYER_1 = MPO_ALGO  # Which algorithm do you want to test? Can be "human" or an algo constant
TEST_USE_ALGO_PLAYER_2 = STRONG_COMP_ALGO  # Only Hockey: Which algorithm do you want to test for player 2? Can be "human" or an algo constant
TEST_NUMBER_DISCRETE_ACTIONS = None  # if you want to use discrete actions or continuous. If > 0, you use the DiscreteActionWrapper
TEST_SEED = 100000  # Set a test seed if you want to
TEST_RENDER_MODE = HUMAN  # For whom do you want to render? None or HUMAN
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # which device are you using?
TEST_ITERATIONS = 10  # The number of test iterations


def test():
    initSeed(seed = TEST_SEED, device = TEST_DEVICE)
    print(f"Test seed: {TEST_SEED}, Test Device: {TEST_DEVICE}")

    env = initEnv(TEST_USE_ENV, TEST_RENDER_MODE, TEST_NUMBER_DISCRETE_ACTIONS)

    if TEST_USE_ALGO_PLAYER_1 == "human":
        player1 = h_env.HumanOpponent(env = env, player = 1)
    else:
        player1 = initAgent(use_algo = TEST_USE_ALGO_PLAYER_1, env = env, agent_settings = AGENT_SETTINGS,
                            device = TEST_DEVICE,
                            checkpoint_name = TEST_CHECK_POINT_NAME_PLAYER_1)
        player1.setMode(eval = True)

    if TEST_USE_ALGO_PLAYER_2 == "human":
        player2 = h_env.HumanOpponent(env = env, player = 2)
    else:
        player2 = initAgent(use_algo = TEST_USE_ALGO_PLAYER_2, env = env, agent_settings = AGENT_SETTINGS,
                            device = TEST_DEVICE,
                            checkpoint_name = TEST_CHECK_POINT_NAME_PLAYER_2)
        player2.setMode(eval = True)

    episode_steps = []
    episode_rewards = []
    for i_test in range(1, TEST_ITERATIONS + 1):
        t_start = time.time()
        total_steps = 0
        total_reward = 0

        if isinstance(player1, Agent):
            player1.reset()
        if isinstance(player2, Agent):
            player2.reset()

        state, info = env.reset(seed = TEST_SEED + i_test)
        env.render()
        if TEST_USE_ENV == HOCKEY:
            state2 = env.obs_agent_two()

        for _ in count():
            if TEST_USE_ENV == HOCKEY:
                env.render()

            if isinstance(player1, Agent):
                state = torch.tensor(state, device = TEST_DEVICE, dtype = torch.float32)

            action1 = player1.act(state)
            # Check if the action is discrete 
            if isinstance(action1, int) and TEST_USE_ENV == HOCKEY: 
                action1 = env.discrete_to_continous_action(action1)
            if TEST_USE_ENV == HOCKEY and isinstance(player2, Agent):
                state2 = torch.tensor(state2, device = TEST_DEVICE, dtype = torch.float32)

            if TEST_USE_ENV == HOCKEY:
                action2 = player2.act(state2)
                # Check if the action is discrete
                if isinstance(action2, int):
                    action2 = env.discrete_to_continous_action(action2)
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


if __name__ == '__main__':
    test()
