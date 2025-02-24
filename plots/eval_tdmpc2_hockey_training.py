# Settings for this class
import logging

import numpy as np
import torch
from matplotlib import pyplot as plt
from pytorch_lightning.utilities.types import EVAL_DATALOADERS

from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, BUFFER_SIZE, DDPG_SETTINGS, MODEL_NAME, USE_TF32
from src.training_loops.tdmpc2_training import do_tdmpc2_hockey_training
from src.util.constants import DDPG_ALGO, HOCKEY, HUMAN, RANDOM_ALGO, STRONG_COMP_ALGO, TDMPC2_ALGO, WEAK_COMP_ALGO
from src.util.contract import initAgent, initEnv, initSeed, setupLogging
from src.util.directoryutil import get_path

BEST_TDMPC2_CHECKPOINT = get_path(
    "final_checkpoints/tdmpc2-v2-all-i6 25-02-20 17_44_47_000061500.pth")  # Which checkpoint do you want to test
BEST_DDPG_CHECKPOINT = get_path(
    "final_checkpoints/hockey_ddpg_smoothl1_25-01-22 17_36_56_100000.pth")  # Which checkpoint do you want to test
TOURNAMENT_RESULTS_FILE_NAME = get_path("plots/eval_tdmpc2_hockey_results.tex")
NUM_TRAINING_EPISODES = 10  # The number of games each pair is playing against

EVAL_USE_ALGO = TDMPC2_ALGO
EVAL_TDMPC2_ENV = HOCKEY  # On which environment do you want to test?
EVAL_TDMPC2_PROXY_REWARDS = True  # On which environment do you want to test?
EVAL_NUMBER_DISCRETE_ACTIONS = None  # if you want to use discrete actions or continuous. If > 0, you use the DiscreteActionWrapper
EVAL_SEEDS = [1000000, 2000000, 3000000]  # Set a test seed if you want to
EVAL_RENDER_MODE = HUMAN  # For whom do you want to render? None or HUMAN
EVAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # which device are you using?
EVAL_SELF_PLAY = False
EVAL_NUM_ROUNDS = 3
COLOR_MAP = ["#ADD8E6", "#008000", "#FFD700", "#800080", "#FF7F50", "#40E0D0", "#708090", "#FF8C00", "#FF00FF",
             "#008080"]


def eval_tdmpc2_hockey_training():
    if USE_TF32:
        torch.set_float32_matmul_precision("high")

    # Setup Logging
    setupLogging(model_name=MODEL_NAME)

    # Initialize the environment
    env = initEnv(EVAL_TDMPC2_ENV, EVAL_RENDER_MODE, EVAL_NUMBER_DISCRETE_ACTIONS, EVAL_TDMPC2_PROXY_REWARDS)

    random_agent = initAgent(use_algo=RANDOM_ALGO, env=env, device=EVAL_DEVICE, checkpoint_name=None)
    weak_comp_agent = initAgent(use_algo=WEAK_COMP_ALGO, env=env, device=EVAL_DEVICE, checkpoint_name=None)
    strong_comp_agent = initAgent(use_algo=STRONG_COMP_ALGO, env=env, device=EVAL_DEVICE, checkpoint_name=None)
    ddpg_agent = initAgent(use_algo=DDPG_ALGO, env=env, device=EVAL_DEVICE,
                           checkpoint_name=DDPG_SETTINGS["CHECKPOINT_NAME"])

    # Currently, we do not allow the opponent networks to train as well. This might be an extra feature
    random_agent.setMode(eval=True)
    weak_comp_agent.setMode(eval=True)
    strong_comp_agent.setMode(eval=True)
    ddpg_agent.setMode(eval=True)

    opponent_pool = {
        RANDOM_ALGO: random_agent,
        WEAK_COMP_ALGO: weak_comp_agent,
        STRONG_COMP_ALGO: strong_comp_agent,
        f"{DDPG_ALGO}_Checkpoint": ddpg_agent,
    }

    all_episode_durations, all_episode_rewards, all_episode_training_statistics, all_opponent_statistics = [], [], [], []

    for i in range(EVAL_NUM_ROUNDS):
        seed = EVAL_SEEDS[i]
        initSeed(seed=seed, device=EVAL_DEVICE)
        logging.info(f"Round {i + 1} of {EVAL_NUM_ROUNDS} | Test seed: {seed}, Test Device: {EVAL_DEVICE}")
        agent = initAgent(EVAL_USE_ALGO, env=env, device=EVAL_DEVICE, agent_settings=AGENT_SETTINGS,
                          checkpoint_name=None)
        agent.setMode(eval=False)  # Set the agent in training mode
        memory = ReplayMemory(capacity=BUFFER_SIZE, device=EVAL_DEVICE)

        episode_durations, episode_rewards, episode_training_statistics, opponent_statistics = do_tdmpc2_hockey_training(
            env=env, agent=agent, memory=memory,
            opponent_pool=opponent_pool, self_opponent=None, num_training_episodes=NUM_TRAINING_EPISODES, seed=seed)
        all_episode_training_statistics.append(episode_training_statistics)
        all_episode_durations.append(episode_durations)
        all_episode_rewards.append(episode_rewards)
        all_opponent_statistics.append(opponent_statistics)

    print(all_episode_durations, all_episode_rewards, all_episode_training_statistics, all_opponent_statistics)
    x_axis = np.arange(1, NUM_TRAINING_EPISODES + 1)
    mean_episode_duration = np.array(all_episode_durations).mean(0)
    std_episode_duration = np.array(all_episode_durations).std(0, ddof=1)
    all_avg_total_losses = np.array(
        [[episode['Avg. Total Loss'] for episode in iter] for iter in all_episode_training_statistics])
    mean_total_losses = all_avg_total_losses.mean(0)
    std_total_losses = all_avg_total_losses.std(0, ddof=1)
    random_agent_list = [[episode["Random_Algo"] for episode in iter] for iter in all_opponent_statistics]
    random_agent_win_rate = np.array([[episode["WIN_RATE"] for episode in iter] for iter in random_agent_list])
    random_agent_win_rate_mean = random_agent_win_rate.mean(0)
    random_agent_win_rate_std = random_agent_win_rate.std(0, ddof=1)
    weak_agent_list = [[episode["Weak_Comp_Algo"] for episode in iter] for iter in all_opponent_statistics]
    weak_agent_win_rate = np.array([[episode["WIN_RATE"] for episode in iter] for iter in weak_agent_list])
    weak_agent_win_rate_mean = weak_agent_win_rate.mean(0)
    weak_agent_win_rate_std = weak_agent_win_rate.std(0, ddof=1)
    strong_agent_list = [[episode["Strong_Comp_Algo"] for episode in iter] for iter in all_opponent_statistics]
    strong_agent_win_rate = np.array([[episode["WIN_RATE"] for episode in iter] for iter in strong_agent_list])
    strong_agent_win_rate_mean = strong_agent_win_rate.mean(0)
    strong_agent_win_rate_std = strong_agent_win_rate.std(0, ddof=1)
    ddpg_agent_list = [[episode["DDPG_Algo_Checkpoint"] for episode in iter] for iter in all_opponent_statistics]
    ddpg_agent_win_rate = np.array([[episode["WIN_RATE"] for episode in iter] for iter in ddpg_agent_list])
    ddpg_agent_win_rate_mean = ddpg_agent_win_rate.mean(0)
    ddpg_agent_win_rate_std = ddpg_agent_win_rate.std(0, ddof=1)

    # Plot Mean with 95% Confidence Interval
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(x_axis, mean_episode_duration, color=COLOR_MAP[0])
    axes[0].fill_between(x_axis, mean_episode_duration - std_episode_duration,
                         mean_episode_duration + std_episode_duration, color=COLOR_MAP[0], alpha=0.2)
    axes[0].set_xticks(x_axis)
    axes[0].set_xlabel('Episodes')
    axes[0].set_ylabel('Duration')
    axes[0].set_title('Duration over Episodes')
    axes[0].grid(True)

    axes[1].plot(x_axis, mean_total_losses, color=COLOR_MAP[0])
    axes[1].fill_between(x_axis, mean_total_losses - std_total_losses,
                         mean_total_losses + std_total_losses, color=COLOR_MAP[0], alpha=0.2)
    axes[1].set_xticks(x_axis)
    axes[1].set_xlabel('Episodes')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Duration over Episodes')
    axes[1].grid(True)

    # Random
    axes[2].plot(x_axis, random_agent_win_rate_mean, color=COLOR_MAP[0], label='Random Agent')
    axes[2].fill_between(x_axis, random_agent_win_rate_mean - random_agent_win_rate_std,
                         random_agent_win_rate_mean + random_agent_win_rate_std, color=COLOR_MAP[0], alpha=0.2)
    # Weak
    axes[2].plot(x_axis, weak_agent_win_rate_mean, color=COLOR_MAP[1], label='Weak Agent')
    axes[2].fill_between(x_axis, weak_agent_win_rate_mean - weak_agent_win_rate_std,
                         weak_agent_win_rate_mean + weak_agent_win_rate_std, color=COLOR_MAP[1], alpha=0.2)
    # Strong
    axes[2].plot(x_axis, strong_agent_win_rate_mean, color=COLOR_MAP[2], label='Strong Agent')
    axes[2].fill_between(x_axis, strong_agent_win_rate_mean - strong_agent_win_rate_std,
                         strong_agent_win_rate_mean + strong_agent_win_rate_std, color=COLOR_MAP[2], alpha=0.2)
    # DDPG
    axes[2].plot(x_axis, ddpg_agent_win_rate_mean, color=COLOR_MAP[3], label='DDPG Agent')
    axes[2].fill_between(x_axis, ddpg_agent_win_rate_mean - ddpg_agent_win_rate_std,
                         ddpg_agent_win_rate_mean + ddpg_agent_win_rate_std, color=COLOR_MAP[3], alpha=0.2)
    axes[2].set_xticks(x_axis)
    axes[2].set_xlabel('Episodes')
    axes[2].set_ylabel('Win Rate')
    axes[2].set_title('Win Rate over Episodes')
    axes[2].grid(True)
    axes[2].legend()

    # Adjust layout
    plt.tight_layout()

    # Save individual plots
    fig.savefig("eval_tdmpc2_hockey_training.pdf")
    axes[0].figure.savefig("eval_tdmpc2_hockey_training_episode_duration.pdf")
    axes[1].figure.savefig("eval_tdmpc2_hockey_training_episode_duration.pdf")
    axes[2].figure.savefig("eval_tdmpc2_hockey_training_episode_win_rates.pdf")

    plt.show()


if __name__ == '__main__':
    eval_tdmpc2_hockey_training()
