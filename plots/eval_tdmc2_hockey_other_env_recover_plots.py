import pickle

import numpy as np
from matplotlib import pyplot as plt
from tueplots import bundles

from plots.eval_tdmpc2_other_env import LUNAR_LANDER_TRAINING_STEPS, PENDULUM_TRAINING_STEPS
from src.util.directoryutil import get_path

with open(get_path("plots/data/eval_tdmpc2_hockey_training_random_agent_list.pkl"), "rb") as file:
    random_agent_list = pickle.load(file)

print("Pickle done")

with open(get_path("plots/data/eval_tdmpc2_hockey_training_weak_agent_list.pkl"), "rb") as file:
    weak_agent_list = pickle.load(file)

print("Pickle done")

with open(get_path("plots/data/eval_tdmpc2_hockey_training_strong_agent_list.pkl"), "rb") as file:
    strong_agent_list = pickle.load(file)

print("Pickle done")

with open(get_path("plots/data/eval_tdmpc2_hockey_training_ddpg_agent_list.pkl"), "rb") as file:
    ddpg_agent_list = pickle.load(file)

print("Pickle done")

duration = np.load(get_path("plots/data/eval_tdmpc2_hockey_training_duration.npy"))
all_avg_total_losses = np.load(get_path("plots/data/eval_tdmpc2_hockey_training_all_avg_total_losses.npy"))
random_agent_win_rate = np.load(get_path("plots/data/eval_tdmpc2_hockey_training_all_random_agent_win_rate.npy"))
weak_agent_win_rate = np.load(get_path("plots/data/eval_tdmpc2_hockey_training_all_weak_agent_win_rate.npy"))
strong_agent_win_rate = np.load(get_path("plots/data/eval_tdmpc2_hockey_training_strong_agent_win_rate.npy"))
ddpg_agent_win_rate = np.load(get_path("plots/data/eval_tdmpc2_hockey_training_ddpg_agent_win_rate.npy"))
pendulum_reward = np.load(get_path("plots/data/eval_tdmpc2_other_env_pendulum_reward.npy"))
lunar_lander_reward = np.load(get_path("plots/data/eval_tdmpc2_other_env_lunar_lander_reward.npy"))

print("Numpy done")

# plt.rcParams.update(bundles.iclr2024(nrows=1, ncols=3, usetex=False))


COLOR_MAP = ["#008000", "#FFD700", "#800080", "#FF7F50", "#40E0D0", "#708090", "#FF8C00", "#FF00FF",
             "#008080", "#ADD8E6"]

x_axis = np.arange(1, 1000 + 1)
pendulum_x_axis = np.arange(1, PENDULUM_TRAINING_STEPS + 1)
pendulum_reward_mean = pendulum_reward.mean(0)
pendulum_reward_std = pendulum_reward.std(0, ddof=1)

lunar_lander_x_axis = np.arange(1, LUNAR_LANDER_TRAINING_STEPS + 1)
lunar_lander_reward_mean = lunar_lander_reward.mean(0)
lunar_lander_reward_std = lunar_lander_reward.std(0, ddof=1)
mean_episode_duration = duration.mean(0)
std_episode_duration = duration.std(0, ddof=1)
mean_total_losses = all_avg_total_losses.mean(0)
std_total_losses = all_avg_total_losses.std(0, ddof=1)
random_agent_win_rate_mean = random_agent_win_rate.mean(0)
random_agent_win_rate_std = random_agent_win_rate.std(0, ddof=1)
weak_agent_win_rate_mean = weak_agent_win_rate.mean(0)
weak_agent_win_rate_std = weak_agent_win_rate.std(0, ddof=1)
strong_agent_win_rate_mean = strong_agent_win_rate.mean(0)
strong_agent_win_rate_std = strong_agent_win_rate.std(0, ddof=1)
ddpg_agent_win_rate_mean = ddpg_agent_win_rate.mean(0)
ddpg_agent_win_rate_std = ddpg_agent_win_rate.std(0, ddof=1)

print("Prep done")

# Plot Mean with 95% Confidence Interval
fig, axes = plt.subplots(1, 4, figsize=(18, 5))

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

print("ax 00 & 01 done")

axes[2].plot(x_axis, mean_total_losses, color=COLOR_MAP[0])
axes[2].fill_between(x_axis, mean_total_losses - std_total_losses,
                     mean_total_losses + std_total_losses, color=COLOR_MAP[0], alpha=0.2)
axes[2].set_xlabel('Episodes')
axes[2].set_ylabel('Loss')
axes[2].set_title('Duration over Episodes')

print("ax 02 done")

# Random
axes[3].plot(x_axis, random_agent_win_rate_mean, color=COLOR_MAP[0], label='Random Agent')
axes[3].fill_between(x_axis, random_agent_win_rate_mean - random_agent_win_rate_std,
                     random_agent_win_rate_mean + random_agent_win_rate_std, color=COLOR_MAP[0], alpha=0.2)
# Weak
axes[3].plot(x_axis, weak_agent_win_rate_mean, color=COLOR_MAP[1], label='Weak Agent')
axes[3].fill_between(x_axis, weak_agent_win_rate_mean - weak_agent_win_rate_std,
                     weak_agent_win_rate_mean + weak_agent_win_rate_std, color=COLOR_MAP[1], alpha=0.2)
# Strong
axes[3].plot(x_axis, strong_agent_win_rate_mean, color=COLOR_MAP[2], label='Strong Agent')
axes[3].fill_between(x_axis, strong_agent_win_rate_mean - strong_agent_win_rate_std,
                     strong_agent_win_rate_mean + strong_agent_win_rate_std, color=COLOR_MAP[2], alpha=0.2)
# DDPG
axes[3].plot(x_axis, ddpg_agent_win_rate_mean, color=COLOR_MAP[3], label='DDPG Agent')
axes[3].fill_between(x_axis, ddpg_agent_win_rate_mean - ddpg_agent_win_rate_std,
                     ddpg_agent_win_rate_mean + ddpg_agent_win_rate_std, color=COLOR_MAP[3], alpha=0.2)
axes[3].set_xlabel('Episodes')
axes[3].set_ylabel('Win Rate')
axes[3].set_title('Win Rate over Episodes')
axes[3].set_ylim(0, 1)
axes[3].axhline(y=0.57, linestyle='--', color=COLOR_MAP[0], linewidth=2, label="Random Agent (after training)")
axes[3].axhline(y=0.55, linestyle='--', color=COLOR_MAP[1], linewidth=2, label="Weak Agent (after training)")
axes[3].axhline(y=0.47, linestyle='--', color=COLOR_MAP[2], linewidth=2, label="Strong Agent (after training)")
axes[3].axhline(y=0.64, linestyle='--', color=COLOR_MAP[3], linewidth=2, label="DDPG Agent (after training)")
axes[3].legend()

print("ax 03 done")

plt.tight_layout()

# Save individual plots
fig.savefig("eval_tdmpc2_hockey_other_env.pdf")
axes[0].figure.savefig("eval_tdmpc2_other_env_pendulum.pdf")
axes[1].figure.savefig("eval_tdmpc2_other_env_lunarlander.pdf")
axes[2].figure.savefig("eval_tdmpc2_hockey_training_episode_loss.pdf")
axes[3].figure.savefig("eval_tdmpc2_hockey_training_episode_win_rates.pdf")
plt.show()

print("All done")
