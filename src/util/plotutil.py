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

def plot_sac_training_metrics(
    rewards: list,
    wins: list,
    critic_losses: list,
    actor_losses: list,
    alpha_losses: list,
    episodes_per_epoch: int,
    save = False
):
    plt.figure("SAC_Training_Metrics", figsize=(10, 8))
    plt.clf()

    r = np.array(rewards).reshape(-1, episodes_per_epoch).mean(axis=1)

    # Choose one of the smoothing functions:
    w = np.array(wins).reshape(-1, episodes_per_epoch).mean(axis=1)
    c_loss = critic_losses
    a_loss = actor_losses
    alpha_l = alpha_losses

    # Plot them with the same index range (1-based if desired)
    x_vals = np.arange(1, len(r) + 1)

    # Subplot 1: Episode Rewards
    plt.subplot(2, 2, 1)
    plt.plot(x_vals, r, label='Episode Rewards')
    plt.title("Rewards")
    plt.xlabel("Epoch")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid(True)

    # Subplot 2: Win Rate
    plt.subplot(2, 2, 2)
    plt.plot(x_vals, w, color='green', label='Win Indicator (smoothed)')
    plt.title("Win Rate")
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
    plt.title("Critic and Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    # Subplot 4: Alpha Loss
    plt.subplot(2, 2, 4)
    plt.plot(x_vals, alpha_l, label='Alpha Loss', color='purple')
    plt.title("Alpha Loss")
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.suptitle(f"SAC Training Metrics", fontsize=14)
    plt.tight_layout()

    if save:
        plt.savefig(f"sac_training_metrics_epoch.png")

def plot_sac_validation_metrics(
    val_win_rates: list,
    val_avg_rewards: list,
    val_interval: int = 10,
    save: bool = False
):
    plt.figure("SAC_Validation_Metrics", figsize=(8, 4))
    plt.clf()
    
    # X-axis might be the epoch index at which validation was done
    x_vals = np.arange(1, len(val_win_rates) + 1) * val_interval
    
    # Subplot for Win Rates
    plt.subplot(1, 2, 1)
    plt.plot(x_vals, val_win_rates, marker='o', color='blue')
    plt.title("Validation Win Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Win Rate")
    plt.ylim([0, 1])
    plt.grid(True)

    # Subplot for Average Rewards
    plt.subplot(1, 2, 2)
    plt.plot(x_vals, val_avg_rewards, marker='o', color='green')
    plt.title("Validation Average Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Avg Reward")
    plt.grid(True)

    plt.suptitle("SAC Validation Metrics", fontsize=14)
    plt.tight_layout()
    
    if save:
        plt.savefig("sac_validation_metrics.png")
        
def plot_mpo_training_metrics(
    critic_losses: List[float],
    actor_losses: List[float],
    kl: List[float],# Discrete
    kl_µ: List[float],# Continuous
    kl_Σ: List[float],# Continuous
    opponent_metrics: List[dict],
    discrete: bool,
    save: bool 
):
    
    # Clear the current figure and set size
    fig = plt.figure(1, figsize=(6, 5))
    fig.clf()

    # Plot critic loss
    plt.subplot(2, 2, 1)
    plt.plot(np.arange(1, len(critic_losses) + 1), critic_losses, marker='o', label='Critic Loss', color='orange')
    plt.title("Critic Loss")
    plt.xlabel("Episode")
    plt.ylabel("Critic Loss")
    plt.grid()

    # Plot actor loss
    plt.subplot(2, 2, 2)
    plt.plot(np.arange(1, len(actor_losses) + 1), actor_losses, marker='o', label='Actor Loss', color='green')
    plt.title("Actor Loss")
    plt.xlabel("Episode")
    plt.ylabel("Actor Loss")
    plt.grid()

    # Plot KL Divergence
    plt.subplot(2, 2, 3)
    if discrete:
        plt.plot(np.arange(1, len(kl) + 1), kl, marker='o', label='KL', color='blue')
    else:
        plt.plot(np.arange(1, len(kl_µ) + 1), kl_µ, marker='o', label='KL µ', color='purple')
        plt.plot(np.arange(1, len(kl_Σ) + 1), kl_Σ, marker='o', label='KL Σ', color='red')
    plt.title("$KL_D(\pi||\pi_{{target}})$")
    plt.xlabel("Episode")
    plt.ylabel("$KL_D$")
    plt.grid()

    # Plot opponent metrics
    plt.subplot(2, 2, 4)
    opponent_names = list(opponent_metrics[0].keys())
    for opp in opponent_names:
        win_rate_values = [
            iteration.get(opp, {}).get("win_rate", 0)  # Extract win_rate, default to 0 if missing
            for iteration in opponent_metrics
        ]
        plt.plot(
            np.arange(1, len(win_rate_values) + 1),
            win_rate_values,
            marker="o",
            label=f"{opp}",
        )
        
    plt.title("Win Rate")
    plt.xlabel("Validation")
    plt.ylabel("Win Rate")
    plt.legend(prop={'size': 6})
    plt.grid()
    
    plt.tight_layout()

    # Pause to refresh the plot
    plt.pause(1)
    
    if save:
        plt.savefig("mpo_training_metrics.pdf")

def plot_mpo_intrinsic_rewards(
    rewards : List[float],
    intrinsic_rewards : List[float],
    save: bool
):
    # Clear the current figure and set size
    fig = plt.figure(2, figsize=(6, 3))
    fig.clf()
    
    # Plot the rewards
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, len(rewards) + 1), rewards, marker='o', label='Rewards', color='blue')
    plt.title("True Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(prop={'size': 6})
    plt.grid()
    
    # Plot the intrinsic rewards
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, len(intrinsic_rewards) + 1), intrinsic_rewards, marker='o', label='Intrinsic Rewards', color='green')
    plt.title("Intrinsic Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(prop={'size': 6})
    plt.grid()
    
    plt.tight_layout()
    
    plt.pause(1)
    
    if save:
        plt.savefig("intrinsic_rewards.pdf")
    
