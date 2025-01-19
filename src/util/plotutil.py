from typing import List

import matplotlib.pyplot as plt
import numpy as np

# Initialize the interactive mode in matplotlib
plt.ion()


def plot_training_metrics(episode_durations: List[int], episode_rewards: List[float], episode_losses: list,
                          current_episode: int, episode_update_iter: int):
    """
    Dynamically updates the plots for training metrics.
    """

    # Step 01: Clear the current figure
    plt.clf()

    # Step 02: Plot rewards for each episode
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(1, len(episode_durations) + 1), episode_durations, marker = 'o', label = 'Duration')
    plt.title("Duration per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.legend()
    plt.grid()

    # Step 03: Plot rewards for each episode
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(1, len(episode_rewards) + 1), episode_rewards, marker = 'o', label = 'Reward')
    plt.title("Rewards per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid()

    # Step 03: Plot average loss for each episode
    average_losses = [np.array(losses).mean() for losses in episode_losses]
    plt.subplot(2, 2, 3)
    plt.plot(episode_update_iter * np.arange(1, len(episode_losses) + 1), average_losses, marker = 'o',
             label = 'Avg Loss', color = 'orange')
    plt.title("Average Loss per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid()

    # Step 04: Plot all losses for the current episode
    all_losses = episode_losses[-1]
    plt.subplot(2, 2, 4)
    plt.plot(np.arange(1, len(all_losses) + 1), all_losses, marker = 'x', label = f'Losses (Episode {current_episode})',
             color = 'red')
    plt.title(f"All Losses for Episode {current_episode}")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Step 05: Layouting
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.5, hspace = 0.5)

    # Step 06: Pause to refresh the plot
    plt.pause(1)

def rolling_avg_variable(data, window):
    """
    Returns a rolling-average-smoothed version of `data`
    with the same length as `data`. At the edges, the window
    size is adjusted so that only valid data points are used.
    """
    data = np.array(data)
    n = len(data)
    if n < window or window < 2:
        return data
    
    smoothed = np.empty(n)
    for i in range(n):
        # Determine the start and end of the window
        start = max(0, i - window//2)
        end = min(n, i + window//2 + 1)  # +1 because slicing is exclusive
        smoothed[i] = np.mean(data[start:end])
    return smoothed

def plot_sac_training_metrics(
    rewards: list,
    wins: list,
    critic_losses: list,
    actor_losses: list,
    alpha_losses: list,
    current_epoch: int,
    smoothing_window: int = 10,
    save = False
):
    plt.figure("SAC_Training_Metrics", figsize=(10, 8))
    plt.clf()

    r = np.array(rewards).reshape(-1, smoothing_window).mean(axis=1)

    # Choose one of the smoothing functions:
    w = np.array(wins).reshape(-1, smoothing_window).mean(axis=1)
    c_loss = critic_losses
    a_loss = actor_losses
    alpha_l = alpha_losses

    # Plot them with the same index range (1-based if desired)
    x_vals = np.arange(1, len(r) + 1)

    # Subplot 1: Episode Rewards
    plt.subplot(2, 2, 1)
    plt.plot(x_vals, r, label='Episode Rewards')
    plt.title("Rewards (Smoothed)")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Win Rate
    plt.subplot(2, 2, 2)
    plt.plot(x_vals, w, color='green', label='Win Indicator (smoothed)')
    plt.title("Win Rate (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Win (0/1)")
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True)

    x_vals = np.arange(1, len(c_loss) + 1)

    # Subplot 3: Critic vs Actor Loss
    plt.subplot(2, 2, 3)
    plt.plot(x_vals, c_loss, label='Critic Loss', color='red')
    plt.plot(x_vals, a_loss, label='Actor Loss', color='orange')
    plt.title("Critic and Actor Loss (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Subplot 4: Alpha Loss
    plt.subplot(2, 2, 4)
    plt.plot(x_vals, alpha_l, label='Alpha Loss', color='purple')
    plt.title("Alpha Loss (Smoothed)")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"SAC Training Metrics (Epoch {current_epoch})", fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(f"sac_training_metrics_epoch.png")
    plt.pause(0.1)