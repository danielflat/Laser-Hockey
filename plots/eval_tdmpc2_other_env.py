# Settings for this class
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from tueplots import bundles

from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, BUFFER_SIZE, DDPG_SETTINGS, MODEL_NAME, USE_TF32
from src.training_loops.tdmpc2_training import do_tdmpc2_hockey_training, do_tdmpc2agent_other_env_training
from src.util.constants import DDPG_ALGO, HOCKEY, HUMAN, LUNARLANDER_CONTINOUS, PENDULUM, RANDOM_ALGO, STRONG_COMP_ALGO, \
    TDMPC2_ALGO, \
    WEAK_COMP_ALGO
from src.util.contract import initAgent, initEnv, initSeed, setupLogging
from src.util.directoryutil import get_path

PENDULUM_TRAINING_STEPS = 500  # The number of training steps for the pendulum environment
LUNAR_LANDER_TRAINING_STEPS = 10_000  # The number of training steps for the pendulum environment

EVAL_USE_ALGO = TDMPC2_ALGO
EVAL_SEEDS = [1000000, 2000000, 3000000]  # Set a test seed if you want to
EVAL_RENDER_MODE = HUMAN  # For whom do you want to render? None or HUMAN
EVAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # which device are you using?
EVAL_NUM_ROUNDS = 3
COLOR_MAP = ["#008000", "#FFD700", "#800080", "#FF7F50", "#40E0D0", "#708090", "#FF8C00", "#FF00FF",
             "#008080", "#ADD8E6"]


def eval_tdmpc2_other_env():
    if USE_TF32:
        torch.set_float32_matmul_precision("high")

    # Setup Logging
    setupLogging(model_name=MODEL_NAME)

    plt.rcParams.update(bundles.neurips2024(nrows=1, ncols=2, usetex=False))

    all_pendulum_rewards, all_lunar_lander_rewards = [], []

    for i in range(EVAL_NUM_ROUNDS):
        seed = EVAL_SEEDS[i]
        initSeed(seed=seed, device=EVAL_DEVICE)
        logging.info(f"Round {i + 1} of {EVAL_NUM_ROUNDS} | Test seed: {seed}, Test Device: {EVAL_DEVICE}")
        # Initialize the pendulum environment
        env = initEnv(PENDULUM, EVAL_RENDER_MODE, None, False)

        agent = initAgent(EVAL_USE_ALGO, env=env, device=EVAL_DEVICE, agent_settings=AGENT_SETTINGS,
                          checkpoint_name=None)
        agent.setMode(eval=False)  # Set the agent in training mode
        memory = ReplayMemory(capacity=BUFFER_SIZE, device=EVAL_DEVICE)

        logging.info(f"Round {i + 1} of {EVAL_NUM_ROUNDS} | Start Pendulum!")
        pendulum_rewards = do_tdmpc2agent_other_env_training(
            env=env, agent=agent, memory=memory, num_training_episodes=PENDULUM_TRAINING_STEPS, seed=seed)

        all_pendulum_rewards.append(pendulum_rewards)

        # Initialize the lunarlander environment
        env = initEnv(LUNARLANDER_CONTINOUS, EVAL_RENDER_MODE, None, False)

        agent = initAgent(EVAL_USE_ALGO, env=env, device=EVAL_DEVICE, agent_settings=AGENT_SETTINGS,
                          checkpoint_name=None)
        agent.setMode(eval=False)  # Set the agent in training mode
        memory = ReplayMemory(capacity=BUFFER_SIZE, device=EVAL_DEVICE)

        logging.info(f"Round {i + 1} of {EVAL_NUM_ROUNDS} | Start Lunar Lander!")
        lunar_lander_rewards = do_tdmpc2agent_other_env_training(
            env=env, agent=agent, memory=memory, num_training_episodes=LUNAR_LANDER_TRAINING_STEPS, seed=seed)

        all_lunar_lander_rewards.append(lunar_lander_rewards)

    print(all_pendulum_rewards)
    pendulum_x_axis = np.arange(1, PENDULUM_TRAINING_STEPS + 1)
    pendulum_reward = np.array(all_pendulum_rewards)
    pendulum_reward_mean = pendulum_reward.mean(0)
    pendulum_reward_std = pendulum_reward.std(0, ddof=1)

    lunar_lander_x_axis = np.arange(1, LUNAR_LANDER_TRAINING_STEPS + 1)
    lunar_lander_reward = np.array(all_lunar_lander_rewards)
    lunar_lander_reward_mean = lunar_lander_reward.mean(0)
    lunar_lander_reward_std = lunar_lander_reward.std(0, ddof=1)

    np.save(get_path("plots/data/eval_tdmpc2_other_env_pendulum_reward.npy"), pendulum_reward)
    np.save(get_path("plots/data/eval_tdmpc2_other_env_lunar_lander_reward.npy"), lunar_lander_reward)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))

    axes[0].plot(pendulum_x_axis, pendulum_reward_mean, color=COLOR_MAP[0])
    axes[0].fill_between(pendulum_x_axis, pendulum_reward_mean - pendulum_reward_std,
                         pendulum_reward_mean + pendulum_reward_std, color=COLOR_MAP[0], alpha=0.2)
    axes[0].set_xticks(pendulum_x_axis)
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Pendulum Rewards over Episodes')

    axes[1].plot(lunar_lander_x_axis, lunar_lander_reward_mean, color=COLOR_MAP[0])
    axes[1].fill_between(lunar_lander_x_axis, lunar_lander_reward_mean - lunar_lander_reward_std,
                         lunar_lander_reward_mean + lunar_lander_reward_std, color=COLOR_MAP[0], alpha=0.2)
    axes[1].set_xticks(lunar_lander_x_axis)
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Reward')
    axes[1].set_title('Lunar Lander Rewards over Episodes')

    # Save individual plots
    fig.savefig("eval_tdmpc2_other_env.pdf")
    axes[0].figure.savefig("eval_tdmpc2_other_env_pendulum.pdf")
    axes[1].figure.savefig("eval_tdmpc2_other_env_lunarlander.pdf")

    plt.show()


if __name__ == '__main__':
    eval_tdmpc2_other_env()
