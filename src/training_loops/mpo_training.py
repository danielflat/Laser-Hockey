import logging
import numpy as np

import random

import time
import torch

from src.agent import Agent
from src.settings import CHECKPOINT_ITER, CURIOSITY, DEVICE, EPISODE_UPDATE_ITER, MODEL_NAME, NUM_TRAINING_EPISODES, \
    SEED, \
    SELF_PLAY, USE_ALGO


def do_mpo_hockey_training(env, agent, memory, opponent_pool: dict, self_opponent: Agent):
    """
    TODO: Add support for discrete action spaces
    """
    episode_durations, episode_rewards, training_statistics = [], [], []

    # Initialize statistics
    self_statistics = {
        "WIN_RATE": 0.5, "DRAW_RATE": 0.0, "LOSE_RATE": 0.5,
        "NUM_GAMES": 0, "NUM_GAMES_WIN": 0, "NUM_GAMES_DRAW": 0, "NUM_GAMES_LOSE": 0,
    } if SELF_PLAY else None

    opponent_statistics = {
        opponent: {
            "WIN_RATE": 0.5, "DRAW_RATE": 0.0, "LOSE_RATE": 0.5,
            "NUM_GAMES": 0, "NUM_GAMES_WIN": 0, "NUM_GAMES_DRAW": 0, "NUM_GAMES_LOSE": 0,
        }
        for opponent in opponent_pool.keys()
    }

    # Training loop
    for episode in range(1, NUM_TRAINING_EPISODES + 1):
        # Start the episode
        start_time = time.time()
        total_reward, steps = 0, 0

        # Select self opponent in 1/2 of the cases
        if self_opponent and episode % 2 == 0:
            opponent, opponent_name = self_opponent, USE_ALGO
        # Select opponent based on win rates
        else:
            win_rates = {key: 1 - abs(0.5 - val["WIN_RATE"]) for key, val in opponent_statistics.items()}
            opponent_name = random.choices(list(opponent_pool.keys()), weights = win_rates.values())[0]
            opponent = opponent_pool[opponent_name]

        # Reset environment
        state, info = env.reset(seed = SEED + episode - 1)
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)
        state_opponent = torch.tensor(env.obs_agent_two(), device = DEVICE, dtype = torch.float32)

        while True:
            # Perform actions
            action = agent.act(state)
            action_opponent = opponent.act(state_opponent)
            next_state, reward, terminated, truncated, info = env.step(np.hstack([action, action_opponent]))
            next_state_opponent = env.obs_agent_two()
            done = terminated or truncated

            # Track rewards and push to memory
            total_reward += reward

            # Convert quantities into tensors
            action = torch.tensor(action, device = DEVICE, dtype = torch.float32)
            action_opponent = torch.tensor(action_opponent, device = DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = DEVICE, dtype = torch.float32)
            done = torch.tensor(terminated or truncated, device = DEVICE, dtype = torch.int)
            next_state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)
            next_state_opponent = torch.tensor(next_state_opponent, device = DEVICE, dtype = torch.float32)

            # Add intrinsic reward if curiosity is enabled
            if CURIOSITY is not None:
                reward = reward + CURIOSITY * agent.icm.compute_intrinsic_reward(state, next_state, action)

            # Store transitions
            memory.push(state, action, reward, next_state, done, info)

            # Update states
            state, state_opponent = (next_state, next_state_opponent)

            steps += 1
            if done:
                break

        # Update statistics
        stat_entry = self_statistics if SELF_PLAY and opponent_name == USE_ALGO else opponent_statistics[opponent_name]
        stat_entry["NUM_GAMES"] += 1
        if info["winner"] == 1:
            stat_entry["NUM_GAMES_WIN"] += 1
        elif info["winner"] == -1:
            stat_entry["NUM_GAMES_LOSE"] += 1
        else:
            stat_entry["NUM_GAMES_DRAW"] += 1
        stat_entry["WIN_RATE"] = stat_entry["NUM_GAMES_WIN"] / stat_entry["NUM_GAMES"]
        stat_entry["DRAW_RATE"] = stat_entry["NUM_GAMES_DRAW"] / stat_entry["NUM_GAMES"]
        stat_entry["LOSE_RATE"] = stat_entry["NUM_GAMES_LOSE"] / stat_entry["NUM_GAMES"]

        # Log episode statistics
        episode_durations.append(steps)
        episode_rewards.append(total_reward)
        if episode % EPISODE_UPDATE_ITER == 0:
            training_stats = agent.optimize(memory = memory, episode_i = episode)
            training_statistics.append(training_stats)
            duration = time.time() - start_time
            logging.info(
                f"Episode: {episode} | Time: {duration:.2f}s | Steps: {steps} | Reward: {total_reward:.2f} | Winner: {info['winner']} |"
                f"Opponent: {opponent_name} | Stats: {', '.join(f'{k}: {v:.4f}' for k, v in training_stats.items())}"
            )

        # Periodic updates of self-play opponent and logging of statistics
        if episode % 10 == 0:
            if SELF_PLAY:
                self_opponent.import_checkpoint(agent.export_checkpoint())
                logging.info(f"Episode {episode}: Updated self-play opponent.")
            all_stats = {USE_ALGO: self_statistics} if SELF_PLAY else {}
            all_stats.update(opponent_statistics)
            formatted_stats = "\n".join(
                f"{name}: " + ", ".join(f"{stat}: {val:.4f}" for stat, val in stats.items())
                for name, stats in all_stats.items()
            )
            logging.info(f"Statistics at Episode {episode}:\n{formatted_stats}")

        # Save checkpoints
        if episode % CHECKPOINT_ITER == 0:
            agent.saveModel(MODEL_NAME, episode)
