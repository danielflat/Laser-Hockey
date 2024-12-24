import matplotlib.pyplot as plt
import numpy as np

# Initialize the interactive mode in matplotlib
plt.ion()


def plot_training_metrics(episode_durations: list[int], episode_rewards: list[float], episode_losses: list,
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
