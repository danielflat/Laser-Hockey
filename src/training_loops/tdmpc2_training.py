import logging
import numpy as np
import random
import time
import torch
import yaml
import copy
from itertools import count

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, BATTLE_STATISTICS_FREQUENCY, CHECKPOINT_ITER, DDPG_SETTINGS, DEVICE, \
    DQN_SETTINGS, EPISODE_UPDATE_ITER, \
    MAIN_SETTINGS, \
    MODEL_NAME, MPO_SETTINGS, \
    NUM_TRAINING_EPISODES, PLOT_FREQUENCY, PPO_SETTINGS, \
    RENDER_MODE, SAC_SETTINGS, \
    SEED, SELF_PLAY, SELF_PLAY_FREQUENCY, SELF_PLAY_KEEP_AGENT_FREQUENCY, SELF_PLAY_UPDATE_FREQUENCY, SETTINGS, \
    SHOW_PLOTS, TD3_SETTINGS, TD_MPC2_SETTINGS, USE_ALGO
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, PPO_ALGO, RANDOM_ALGO, SAC_ALGO, STRONG_COMP_ALGO, \
    TD3_ALGO, TDMPC2_ALGO, WEAK_COMP_ALGO, MPO_ALGO
from src.util.contract import initAgent, initEnv, initValEnv, initSeed, setupLogging
from src.util.plotutil import plot_training_metrics, plot_sac_training_metrics, plot_sac_validation_metrics


def do_tdmpc2agent_other_env_training(env, agent, memory):
    episodes_durations = []
    episodes_rewards = []
    episode_training_statistics = []
    episodes_losses = []  # TODO: CURRENTLY NOT USED ANYMORE

    state, info = env.reset(seed = SEED)

    for i_training in range(1, NUM_TRAINING_EPISODES + 1):
        # We track for each episode how high the reward was
        t_start = time.time()
        total_reward = 0
        all_states = []
        all_actions = []
        all_rewards = []
        all_next_states = []
        all_dones = []
        all_infos = []

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
            next_state = torch.from_numpy(next_state).to(device = DEVICE)

            # We keep track of the episode
            all_states.append(state)
            all_actions.append(action)
            all_rewards.append(reward)
            all_dones.append(done)
            all_next_states.append(next_state)
            all_infos.append(info)

            # Update the state
            state = next_state
            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                episodes_durations.append(step)
                episodes_rewards.append(total_reward)
                break

        all_states = torch.stack(all_states, dim = 0)
        all_actions = torch.stack(all_actions, dim = 0)
        all_rewards = torch.stack(all_rewards, dim = 0)
        all_dones = torch.stack(all_dones, dim = 0)
        all_next_states = torch.stack(all_next_states, dim = 0)

        memory.push(all_states, all_actions, all_rewards, all_next_states, all_dones, all_infos)

        # After some episodes and collecting some data, we optimize the agent
        if i_training % EPISODE_UPDATE_ITER == 0:
            training_statistics = agent.optimize(memory = memory, episode_i = i_training)

            # After optimization, we can log some *more* statistics
            t_end = time.time()
            episode_time = t_end - t_start
            episode_training_statistics.append(training_statistics)
            other_statistics = " | ".join([f"{key}: {value:.4f}" for key, value in training_statistics.items()])
            logging.info(
                f"Training Iter: {i_training} | Req. Time: {episode_time:.4f} sec. | Req. Steps: {episodes_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
                f" {other_statistics}")

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


def do_tdmpc2_hockey_training(env, agent, memory, opponent_pool: dict, self_opponent: Agent):
    episode_durations = []
    episode_rewards = []
    episode_training_statistics = []

    if SELF_PLAY:
        # create a dict for the self component for logging statistics
        self_statistics = {
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

    # df: We currently do not support opponent training.
    # memory_opponent = ReplayMemory(capacity = BUFFER_SIZE, device = DEVICE)

    for i_training in range(1, NUM_TRAINING_EPISODES + 1):
        # We track for each episode how high the reward was
        t_start = time.time()

        # temporal variables to track the episode
        total_reward = 0
        episode_result = None  # Can be won, draw or lose
        all_states = []
        all_states_opponent = []
        all_actions = []
        all_actions_opponent = []
        all_rewards = []
        all_next_states = []
        all_next_states_opponent = []
        all_dones = []
        all_infos = []

        # if we use self play, we want to play \frac{SELF_PLAY_FREQUENCY-1}{SELF_PLAY_FREQUENCY} times against the
        # own agent
        if self_opponent is not None and i_training % SELF_PLAY_FREQUENCY != 0:
            opponent = self_opponent
            opponent_name = USE_ALGO
        else:
            # Select an opponent based on win rates. Currently, it prefers sampling winning rates ~0.5
            # example 1: win_rate = 0.8 -> 1 - abs(0.5 - 0.8) = 1 - 0.3 = 0.7
            # example 2: win_rate = 0.5 -> 1 - abs(0.5 - 0.5) = 1 - 0.0 = 1.0 -> preferred opponent
            # example 3: win_rate = 0.1 -> 1 - abs(0.5 - 0.1) = 1 - 0.4 = 0.6
            _win_rates = {key: value["WIN_RATE"] for key, value in opponent_statistics.items()}
            opponent_name = \
                random.choices(list(opponent_pool.keys()),
                               weights = [1 - abs(0.5 - _win_rates[o]) for o in opponent_pool])[0]
            opponent = opponent_pool[opponent_name]

        # For reproducibility of the training, we use predefined seeds
        state, info = env.reset(seed = SEED + i_training - 1)

        # we reset the agents for the new episode
        agent.reset()
        opponent.reset()

        state_opponent = env.obs_agent_two()
        # Convert state to torch
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)
        state_opponent = torch.tensor(state_opponent, device = DEVICE, dtype = torch.float32)

        for step in count(start = 1):
            # Render the scene
            env.render(mode = RENDER_MODE)
            # env.render(mode = HUMAN if i_training % 10 == 0 else None)  # for debugging

            # choose the action
            action = agent.act(state)
            action_opponent = opponent.act(state_opponent)

            # perform the action
            next_state, reward, terminated, truncated, info = env.step(np.hstack([action, action_opponent]))
            next_state_opponent = env.obs_agent_two()
            done = terminated or truncated

            # If the agent wins, we want to give a high reward
            if info["winner"] == 1:
                reward = 10
            # If the agent loses, we want to give a high penalty
            elif info["winner"] == -1:
                reward = -10
            # If the agent draws in the end, we want to give a medium penalty
            elif info["winner"] == 0 and done:
                reward = -5
            # We want to penalize over the time to go and win *fast*, because the simulator only goes up to 251 steps.
            else:
                reward = -0.01

            # track the total reward
            total_reward += reward

            # Convert quantities into tensors
            action = torch.tensor(action, device = DEVICE, dtype = torch.float32)
            action_opponent = torch.tensor(action_opponent, device = DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = DEVICE, dtype = torch.float32)
            done = torch.tensor(done, device = DEVICE,
                                dtype = torch.int)  # to be able to do arithmetics with the done signal, we need an int
            next_state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)
            next_state_opponent = torch.tensor(next_state_opponent, device = DEVICE, dtype = torch.float32)

            # We keep track of the episode
            all_states.append(state)
            all_states_opponent.append(state_opponent)
            all_actions.append(action)
            all_actions_opponent.append(action_opponent)
            all_rewards.append(reward)
            all_next_states.append(next_state)
            all_next_states_opponent.append(next_state_opponent)
            all_dones.append(done)
            all_infos.append(info)

            # Update the state
            state = next_state
            state_opponent = next_state_opponent

            # When this episode is over...
            if done:
                # ... Step 01: we save some statistics
                episode_durations.append(step)
                episode_rewards.append(total_reward)

                # ... Step 02: update the statistics against the opponent
                if SELF_PLAY and opponent_name == USE_ALGO:
                    opponent_stat_entry = self_statistics
                else:
                    opponent_stat_entry = opponent_statistics[opponent_name]
                opponent_stat_entry["NUM_GAMES"] += 1
                if info["winner"] == 1:
                    opponent_stat_entry["NUM_GAMES_WIN"] += 1
                    episode_result = "WON"
                elif info["winner"] == -1:
                    opponent_stat_entry["NUM_GAMES_LOSE"] += 1
                    episode_result = "LOST"
                else:
                    opponent_stat_entry["NUM_GAMES_DRAW"] += 1
                    episode_result = "DRAW"
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

        # We push the episode into the buffer
        all_states = torch.stack(all_states, dim = 0)
        all_actions = torch.stack(all_actions, dim = 0)
        all_rewards = torch.stack(all_rewards, dim = 0)
        all_dones = torch.stack(all_dones, dim = 0)
        all_next_states = torch.stack(all_next_states, dim = 0)
        memory.push(all_states, all_actions, all_rewards, all_next_states, all_dones, all_infos)

        # After some episodes and collecting some data, we optimize the agent
        if i_training % EPISODE_UPDATE_ITER == 0:
            # Train the main agent
            training_statistics = agent.optimize(memory = memory, episode_i = i_training)

            # After optimization, we can log some *more* statistics
            t_end = time.time()
            episode_time = t_end - t_start
            episode_training_statistics.append(training_statistics)
            other_statistics = " | ".join([f"{key}: {value:.4f}" for key, value in training_statistics.items()])
            logging.info(
                f"Training Iter: {i_training} | Req. Time: {episode_time:.4f} sec. | Req. Steps: {episode_durations[i_training - 1]}"
                f" | Episode result: {episode_result} | Total reward: {total_reward:.4f}"
                f" | Opponent: {opponent_name} | {other_statistics}")

        # (Optional): If self play is activated, we want to update the opponent pool ...
        if SELF_PLAY and i_training % SELF_PLAY_UPDATE_FREQUENCY == 0:
            # if we decide to keep the old agent in our pool, we save the statistics and create a new statistics object
            if i_training % SELF_PLAY_KEEP_AGENT_FREQUENCY == 0:
                opponent_pool[f"{USE_ALGO}_{i_training}"] = copy.deepcopy(self_opponent)
                opponent_statistics[f"{USE_ALGO}_{i_training}"] = copy.deepcopy(self_statistics)
                self_statistics = {
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
                added_opponent_extra_info = f" and added {USE_ALGO}_{i_training - SELF_PLAY_UPDATE_FREQUENCY} to the training"
            else:
                added_opponent_extra_info = ""

            # we update the self_opponent
            self_opponent = copy.deepcopy(agent)
            logging.info(
                f"Training Iter: {i_training} | Updated self-opponent to {USE_ALGO}_{i_training}{added_opponent_extra_info}!")

        if i_training % BATTLE_STATISTICS_FREQUENCY == 0:
            # ... You log the battle statistics of each opponent
            battle_statistics = " |\n".join([
                f'"{key}": {{' + ", ".join(
                    f'"{stat_name}": {stat_value:.4f}' if isinstance(stat_value,
                                                                     float) else f'"{stat_name}": {stat_value}'
                    for stat_name, stat_value in values.items()
                ) + "}"
                for key, values in opponent_statistics.items()])

            # if we have self-play, we also log the battle statistics against itself
            if SELF_PLAY:
                battle_statistics = f'"{USE_ALGO}": {{' + ", ".join(
                    f'"{stat_name}": {stat_value:.4f}' if isinstance(stat_value,
                                                                     float) else f'"{stat_name}": {stat_value}'
                    for stat_name, stat_value in self_statistics.items()
                ) + "} |\n" + battle_statistics

            logging.info(f"Training Iter: {i_training} | {battle_statistics}")

        # Plot statistics every #PLOT_FREQUENCY episodes
        if SHOW_PLOTS and i_training % PLOT_FREQUENCY == 0:
            # TODO: df: Does not work yet
            plot_training_metrics(episode_durations = episode_durations, episode_rewards = episode_rewards,
                                  episode_losses = episode_training_statistics, current_episode = i_training,
                                  episode_update_iter = EPISODE_UPDATE_ITER)

        # Frequently after some episodes, we save a checkpoint of our model
        if (i_training % CHECKPOINT_ITER == 0):
            agent.saveModel(MODEL_NAME, i_training)
