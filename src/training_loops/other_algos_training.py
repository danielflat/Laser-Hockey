import random

import time

from itertools import count

import logging
import numpy as np
import torch

from src.replaymemory import ReplayMemory
from src.settings import BATTLE_STATISTICS_FREQUENCY, BUFFER_SIZE, CHECKPOINT_ITER, DEVICE, EPISODE_UPDATE_ITER, \
    MODEL_NAME, NUM_TEST_EPISODES, \
    NUM_TRAINING_EPISODES, \
    RENDER_MODE, \
    SEED, SELF_PLAY, \
    SHOW_PLOTS, USE_ALGO
from src.util.plotutil import plot_training_metrics


def do_hockey_training(env, agent, memory, opponent_pool: dict):
    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_epsilon = []
    win_rates = {opponent: 0.5 for opponent in opponent_pool.keys()}  # Start with 50% win rate for each opponent

    memory_opponent = ReplayMemory(capacity = BUFFER_SIZE, device = DEVICE)

    for i_training in range(1, NUM_TRAINING_EPISODES + 1):
        # We track for each episode how high the reward was
        t_start = time.time()
        total_reward = 0

        # Select an opponent based on win rates. Currently, it prefers sampling winning rates ~0.5
        # example 1: win_rate = 0.8 -> 1 - abs(0.5 - 0.8) = 1 - 0.3 = 0.7
        # example 2: win_rate = 0.5 -> 1 - abs(0.5 - 0.5) = 1 - 0.0 = 1.0 -> preferred opponent
        # example 3: win_rate = 0.1 -> 1 - abs(0.5 - 0.1) = 1 - 0.4 = 0.6
        opponent_name = \
            random.choices(list(opponent_pool.keys()), weights = [1 - abs(0.5 - win_rates[o]) for o in opponent_pool])[
                0]
        opponent = opponent_pool[opponent_name]

        # For reproducibility of the training, we use predefined seeds
        state, info = env.reset(seed = SEED + i_training - 1)
        state_opponent = env.obs_agent_two()
        # Convert state to torch
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)
        state_opponent = torch.tensor(state_opponent, device = DEVICE, dtype = torch.float32)

        for step in count(start = 1):
            # Render the scene
            env.render(mode = RENDER_MODE)
            # env.render(mode = "human")  # for debugging

            # choose the action
            action = agent.act(state)
            action_opponent = opponent.act(state_opponent)

            # perform the action
            next_state, reward, terminated, truncated, info = env.step(np.hstack([action, action_opponent]))
            next_state_opponent = env.obs_agent_two()

            # df: Sometimes, the reward yields small numbers because in the env, they also consider the distance to the puck.
            # In my suggestion, this is not a good thing to do because it sets an unnecessary prior to the model.
            # Therefore, we normalize it to 0 for simplicity.
            if info["winner"] == 1:
                reward = 10
            elif info["winner"] == -1:
                reward = -10
            else:
                reward = 0

            # track the total reward
            total_reward += reward

            # Convert quantities into tensors
            action = torch.tensor(action, device = DEVICE, dtype = torch.float32)
            action_opponent = torch.tensor(action_opponent, device = DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = DEVICE, dtype = torch.float32)
            done = torch.tensor(terminated or truncated, device = DEVICE,
                                dtype = torch.int)  # to be able to do arithmetics with the done signal, we need an int
            next_state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)
            next_state_opponent = torch.tensor(next_state_opponent, device = DEVICE, dtype = torch.float32)

            # Store this transition in the memory
            memory.push(state, action, reward, next_state, done, info)
            memory_opponent.push(state_opponent, action_opponent, -reward, next_state_opponent, done, info)

            # Update the state
            state = next_state
            state_opponent = next_state_opponent

            # If this transition is the last, safe the number of done steps in the env. and end this episode
            if done:
                episode_durations.append(step)
                break

        # after each episode, we want to log some statistics
        episode_rewards.append(total_reward)

        # After some episodes and collecting some data, we optimize the agent
        if i_training % EPISODE_UPDATE_ITER == 0:
            # Train the main agent
            losses = agent.optimize(memory = memory, episode_i = i_training)
            # Train the opponent agents
            for o_name, opp in opponent_pool.items():
                if o_name != USE_ALGO:
                    # you update all the algorithms
                    _ = opp.optimize(memory = memory_opponent, episode_i = i_training)

            # After optimization, we can log some *more* statistics
            t_end = time.time()
            episode_time = t_end - t_start
            episode_losses.append(losses)
            episode_epsilon.append(agent.epsilon)
            logging.info(
                f"Training Iter: {i_training} | Req. Steps: {episode_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
                f" Opponent: {opponent_name} | Avg. Loss: {np.array(losses).mean():.4f} | Epsilon: {agent.epsilon:.4f} | Req. Time: {episode_time:.4f} sec.")

        # Every 100 episodes, you update the self opponent with the current weights
        if SELF_PLAY and i_training % 100 == 0:
            opponent_pool[USE_ALGO].import_checkpoint(agent.export_checkpoint())
            logging.info(f"Training Iter: {i_training} Update Self Opponent weights {opponent_pool[USE_ALGO]}")

        # Plot every 100 episodes
        if SHOW_PLOTS and i_training % 100 == 0:
            plot_training_metrics(episode_durations = episode_durations, episode_rewards = episode_rewards,
                                  episode_losses = episode_losses, current_episode = i_training,
                                  episode_update_iter = EPISODE_UPDATE_ITER)

        # after some time, we save a checkpoint of our model
        if (i_training % CHECKPOINT_ITER == 0):
            agent.saveModel(MODEL_NAME, i_training)


def do_other_env_training(env, agent, memory):
    episodes_durations = []
    episodes_rewards = []
    episodes_losses = []
    episodes_epsilon = []

    state, info = env.reset(seed = SEED)

    for i_training in range(1, NUM_TRAINING_EPISODES + 1):
        # We track for each episode how high the reward was
        t_start = time.time()
        total_reward = 0

        # Convert state to torch
        state = torch.from_numpy(state).to(DEVICE).to(dtype = torch.float32)

        for step in count(start = 1):
            # choose the action
            action = agent.act(state)

            # perform the action
            next_state, reward, terminated, truncated, info = env.step(action)

            # track the total reward
            total_reward += reward

            # Convert quantities into tensors
            action = torch.tensor(action, device = DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = DEVICE, dtype = torch.float32)
            done = torch.tensor(terminated or truncated, device = DEVICE,
                                dtype = torch.int)  # to be able to do arithmetics with the done signal, we need an int
            next_state = torch.from_numpy(next_state).to(device = DEVICE, dtype = torch.float32)

            # Store this transition in the memory
            memory.push(state, action, reward, next_state, done, info)

            # Update the state
            state = next_state
            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                episodes_durations.append(step)
                break

        # after each episode, we want to log some statistics
        episodes_rewards.append(total_reward)

        # After some episodes and collecting some data, we optimize the agent
        if i_training % EPISODE_UPDATE_ITER == 0:
            losses = agent.optimize(memory = memory, episode_i = i_training)

            # After optimization, we can log some *more* statistics
            t_end = time.time()
            episode_time = t_end - t_start
            episodes_losses.append(losses)
            episodes_epsilon.append(agent.epsilon)
            logging.info(
                f"Training Iter: {i_training} | Req. Steps: {episodes_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
                f" Avg. Loss: {np.array(losses).mean():.4f} | Epsilon: {agent.epsilon:.4f} | Req. Time: {episode_time:.4f} sec.")

        # Plot every 100 episodes
        if SHOW_PLOTS and i_training % 100 == 0:
            plot_training_metrics(episode_durations = episodes_durations, episode_rewards = episodes_rewards,
                                  episode_losses = episodes_losses, current_episode = i_training,
                                  episode_update_iter = EPISODE_UPDATE_ITER)

        # after some time, we save a checkpoint of our model
        if (i_training % CHECKPOINT_ITER == 0):
            agent.saveModel(MODEL_NAME, i_training)

        # reset the environment
        state, info = env.reset(
            seed = SEED + i_training)  # by resetting always a different but predetermined seed, we ensure the reproducibility of the results


def do_other_env_testing(env, agent):
    test_durations = []
    test_rewards = []
    state, info = env.reset()

    for i_test in range(1, NUM_TEST_EPISODES + 1):
        # We track for each episode how high the reward was
        total_reward = 0

        # Convert state to torch
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)

        for step in count(start = 1):
            # choose the action
            action = agent.act(state)

            # perform the action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # track the total reward
            total_reward += reward

            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                test_durations.append(step)
                test_rewards.append(total_reward)
                logging.info(f"Test Iter: {i_test} | Req. Steps: {step} | Total reward: {total_reward}")
                break
            else:
                # Update the state
                state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)

        # reset the environment
        state, info = env.reset()

    logging.info(f"Tests done! "
                 f"Durations average: {np.array(test_durations).mean():.4f} | Durations std. dev: {np.array(test_durations).std():.4f} | Durations variance: {np.array(test_durations).var():.4f} | "
                 f"Reward average: {np.array(test_rewards).mean():.4f} | Reward std. dev: {np.array(test_rewards).std():.4f} | Reward variance: {np.array(test_rewards).var():.4f}")


def do_hockey_testing(env, agent, opponent_pool: dict):
    episode_durations = []
    episode_rewards = []

    # create a dict for each opponent and save some statistics there
    opponent_statistics = {
        f"{opponent}": {
            # has to sum up to 1
            "WIN_RATE": 0.5,  # Start with 50% win rate for each opponent
            "DRAW_RATE": 0.0,
            "LOSE_RATE": 0.5,

            # Currently it is not used so much. But an interesting quantity to log with us
            "NUM_GAMES": 0,

            # has to sum up to "NUM_GAMES"
            "NUM_GAMES_WIN": 0,
            "NUM_GAMES_DRAW": 0,
            "NUM_GAMES_LOSE": 0,
        }
        for opponent in opponent_pool.keys()
    }

    for i_test in range(1, NUM_TEST_EPISODES + 1):
        # We track for each episode how high the reward was
        t_start = time.time()
        total_reward = 0
        episode_won = None  # "WON", "DRAW" or "LOST"

        # Select an opponent based on win rates. Currently, it prefers sampling winning rates ~0.5
        # example 1: win_rate = 0.8 -> 1 - abs(0.5 - 0.8) = 1 - 0.3 = 0.7
        # example 2: win_rate = 0.5 -> 1 - abs(0.5 - 0.5) = 1 - 0.0 = 1.0 -> preferred opponent
        # example 3: win_rate = 0.1 -> 1 - abs(0.5 - 0.1) = 1 - 0.4 = 0.6
        opponent_name = list(opponent_pool.keys())[i_test % len(opponent_pool)]
        opponent = opponent_pool[opponent_name]

        # For reproducibility of the training, we use predefined seeds
        state, info = env.reset(seed = SEED + i_test - 1)
        state_opponent = env.obs_agent_two()
        # Convert state to torch
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)
        state_opponent = torch.tensor(state_opponent, device = DEVICE, dtype = torch.float32)

        for step in count(start = 1):
            # Render the scene
            env.render(mode = RENDER_MODE)
            # env.render(mode = HUMAN if i_test % 10 == 0 else None)  # for debugging

            # choose the action
            action = agent.act(state)
            action_opponent = opponent.act(state_opponent)

            # perform the action
            next_state, reward, terminated, truncated, info = env.step(np.hstack([action, action_opponent]))
            next_state_opponent = env.obs_agent_two()

            # df: Sometimes, the reward yields small numbers because in the env, they also consider the distance to the puck.
            # In my suggestion, this is not a good thing to do because it sets an unnecessary target to the model.
            # Therefore, we normalize it to 0 for simplicity.
            # if info["winner"] == 1:
            #     reward = 10
            # elif info["winner"] == -1:
            #     reward = -10
            # else:
            #     reward = 0

            # track the total reward
            total_reward += reward

            # Convert quantities into tensors
            done = torch.tensor(terminated or truncated, device = DEVICE,
                                dtype = torch.int)  # to be able to do arithmetics with the done signal, we need an int
            next_state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)
            next_state_opponent = torch.tensor(next_state_opponent, device = DEVICE, dtype = torch.float32)

            # Update the state
            state = next_state
            state_opponent = next_state_opponent

            # When this episode is over...
            if done:
                # ... Step 01: we save some statistics
                episode_durations.append(step)
                episode_rewards.append(total_reward)

                # ... Step 02: update the statistics against the opponent
                opponent_stat_entry = opponent_statistics[opponent_name]
                opponent_stat_entry["NUM_GAMES"] += 1
                if info["winner"] == 1:
                    opponent_stat_entry["NUM_GAMES_WIN"] += 1
                elif info["winner"] == -1:
                    opponent_stat_entry["NUM_GAMES_LOSE"] += 1
                else:
                    opponent_stat_entry["NUM_GAMES_DRAW"] += 1
                opponent_stat_entry["WIN_RATE"] = opponent_stat_entry[
                                                      "NUM_GAMES_WIN"] / \
                                                  opponent_stat_entry["NUM_GAMES"]
                opponent_stat_entry["DRAW_RATE"] = opponent_stat_entry[
                                                       "NUM_GAMES_DRAW"] / \
                                                   opponent_stat_entry["NUM_GAMES"]
                opponent_stat_entry["LOSE_RATE"] = opponent_stat_entry[
                                                       "NUM_GAMES_LOSE"] / \
                                                   opponent_stat_entry["NUM_GAMES"]

                # ... Step 03: end this episode
                break

        t_end = time.time()
        episode_time = t_end - t_start
        logging.info(
            f"Test Iter: {i_test} | Req. Time: {episode_time:.4f} sec. | Req. Steps: {episode_durations[i_test - 1]} | Total reward: {total_reward:.4f} |"
            f" Opponent: {opponent_name}")

        if i_test % BATTLE_STATISTICS_FREQUENCY == 0:
            # ... You log the battle statistics of each opponent
            battle_statistics = " |\n".join([
                f'"{key}": {{' + ", ".join(
                    f'"{stat_name}": {stat_value:.4f}' if isinstance(stat_value,
                                                                     float) else f'"{stat_name}": {stat_value}'
                    for stat_name, stat_value in values.items()
                ) + "}"
                for key, values in opponent_statistics.items()])

            logging.info(f"Test Iter: {i_test} | {battle_statistics}")
