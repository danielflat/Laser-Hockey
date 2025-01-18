import logging
import numpy as np
import random
import time
import torch
import yaml
from itertools import count

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, MAIN_SETTINGS, SETTINGS
from src.util.constants import HOCKEY, RANDOM_ALGO, SAC_ALGO, STRONG_COMP_ALGO, \
    TDMPC2_ALGO, WEAK_COMP_ALGO
from src.util.contract import initAgent, initEnv, initSeed, setupLogging
from src.util.plotutil import plot_training_metrics

"""
This is the main file of this project.
Here, you can find the main training loop.
In order to set the parameters for training, you can change the values in the settings.py file.

Author: Daniel Flat
"""

# Some helpful constants to use here
SEED = MAIN_SETTINGS["SEED"]
DEVICE = MAIN_SETTINGS["DEVICE"]
USE_TF32 = MAIN_SETTINGS["USE_TF32"]
USE_ENV = MAIN_SETTINGS["USE_ENV"]
RENDER_MODE = MAIN_SETTINGS["RENDER_MODE"]
NUMBER_DISCRETE_ACTIONS = MAIN_SETTINGS["NUMBER_DISCRETE_ACTIONS"]
USE_ALGO = MAIN_SETTINGS["USE_ALGO"]
SELF_PLAY = MAIN_SETTINGS["SELF_PLAY"]
MODEL_NAME = MAIN_SETTINGS["MODEL_NAME"]
BUFFER_SIZE = MAIN_SETTINGS["BUFFER_SIZE"]
NUM_TRAINING_EPISODES = MAIN_SETTINGS["NUM_TRAINING_EPISODES"]
NUM_TEST_EPISODES = MAIN_SETTINGS["NUM_TEST_EPISODES"]
EPISODE_UPDATE_ITER = MAIN_SETTINGS["EPISODE_UPDATE_ITER"]
SHOW_PLOTS = MAIN_SETTINGS["SHOW_PLOTS"]
CHECKPOINT_ITER = MAIN_SETTINGS["CHECKPOINT_ITER"]
CHECKPOINT_NAME = MAIN_SETTINGS["CHECKPOINT_NAME"]

BATCH_SIZE = AGENT_SETTINGS["BATCH_SIZE"]

#
# def evaluate_main_agent(main_agent, opponent, env, num_episodes = 10):
#     """Evaluate the main agent against a specific opponent."""
#     total_reward = 0
#     win_count = 0
#
#     for _ in range(num_episodes):
#         state = env.reset()
#         done = False
#         episode_reward = 0
#
#         while not done:
#             # Main agent action
#             main_action = main_agent.select_action(state)
#
#             # Opponent action
#             opponent_action = opponent.select_action(state)
#
#             # Step environment
#             next_state, reward, done, _ = env.step(main_action, opponent_action)
#             state = next_state
#
#             # Accumulate reward
#             episode_reward += reward
#
#         # Track rewards and wins
#         total_reward += episode_reward
#         if episode_reward > 0:  # Define win criteria (e.g., positive reward = win)
#             win_count += 1
#
#     # Calculate metrics
#     average_reward = total_reward / num_episodes
#     win_rate = win_count / num_episodes
#
#     print(f"Evaluation against {opponent}: Avg Reward = {average_reward}, Win Rate = {win_rate}")
#
#     return average_reward, win_rate


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
            next_state = torch.from_numpy(next_state).to(device = DEVICE)

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


def do_tdmpc2agent_other_env_training(env, agent, memory):
    episodes_durations = []
    episodes_rewards = []
    episode_training_statistics = []

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

def do_sac_hockey_training(env, agent, memory, opponent_pool: dict):
    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_epsilon = []

    # Initialize win rates for each opponent in the pool at 0.5
    pool_win_rates = {name: 0.5 for name in opponent_pool.keys()}

    memory_opponent = ReplayMemory(capacity=BUFFER_SIZE, device=DEVICE)
    i_training = 0

    print("[INFO] Collecting Experiences for SAC until Buffer is full enough to sample well...")

    done_sampling = False

    while i_training < NUM_TRAINING_EPISODES:
        t_start = time.time()
        total_reward = 0

        # Sample an opponent from the pool based on win rate weights:
        # (The closer the win rate to 0.5, the higher its weight)
        opponent_name = random.choices(
            list(opponent_pool.keys()),
            weights=[1 - abs(0.5 - pool_win_rates[o]) for o in opponent_pool]
        )[0]
        opponent = opponent_pool[opponent_name]

        # For reproducibility of the training, we use predefined seeds
        state, info = env.reset(seed=SEED + i_training - 1)
        state_opponent = env.obs_agent_two()

        # Convert state to torch tensors
        state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
        state_opponent = torch.tensor(state_opponent, device=DEVICE, dtype=torch.float32)

        losses = []

        for step in count(start=1):
            env.render(mode=RENDER_MODE)
            # Choose actions for agent and opponent
            action = agent.act(state)
            action_opponent = opponent.act(state_opponent)

            next_state, reward, terminated, truncated, info = env.step(np.hstack([action, action_opponent]))
            next_state_opponent = env.obs_agent_two()

            total_reward += reward

            # Convert quantities into tensors
            action = torch.tensor(action, device=DEVICE, dtype=torch.float32)
            action_opponent = torch.tensor(action_opponent, device=DEVICE, dtype=torch.float32)
            reward_tensor = torch.tensor(reward, device=DEVICE, dtype=torch.float32)
            done = torch.tensor(terminated or truncated, device=DEVICE, dtype=torch.int)
            next_state = torch.tensor(next_state, device=DEVICE, dtype=torch.float32)
            next_state_opponent = torch.tensor(next_state_opponent, device=DEVICE, dtype=torch.float32)

            # Store transitions for agent and opponent (note: opponent uses negative reward)
            memory.push(state, action, reward_tensor, next_state, done, info)
            memory_opponent.push(state_opponent, action_opponent, -reward_tensor, next_state_opponent, done, info)

            if len(memory) >= (100 * BATCH_SIZE):
                step_losses = agent.optimize(memory=memory, episode_i=i_training)
                losses = np.concatenate((losses, step_losses), axis=0)

            # Update states
            state = next_state
            state_opponent = next_state_opponent

            # End the episode when done
            if done:
                episode_durations.append(step)
                episode_rewards.append(total_reward)
                break

        if len(memory) >= 100 * BATCH_SIZE:
            done_sampling = True
            t_end = time.time()
            episode_time = t_end - t_start
            episode_losses.append(losses)
            episode_epsilon.append(agent.epsilon)
            logging.info(
                f"Training Iter: {i_training} | Steps: {episode_durations[-1]} | Total reward: {total_reward:.4f} |"
                f" Opponent: {opponent_name} | Avg. Loss: {np.array(losses).mean():.4f} | "
                f"Epsilon: {agent.epsilon:.4f} | Episode Time: {episode_time:.4f} sec."
            )

            # Running Mean of Opponent's Win Rates
            current_win = 1 if info["winner"] == 1 else 0
            pool_win_rates[opponent_name] = 0.9 * pool_win_rates[opponent_name] + 0.1 * current_win

            # Every 100 episodes, add the current agent to the pool
            if SELF_PLAY and i_training % 100 == 0:
                checkpoint_name = f"self_{i_training}"
                new_opponent = initAgent(use_algo=USE_ALGO, env=env, device=DEVICE)
                new_opponent.setMode(eval=True)
                new_opponent.import_checkpoint(agent.export_checkpoint())
                opponent_pool[checkpoint_name] = new_opponent
                pool_win_rates[checkpoint_name] = 0.5  # Start new opponent with 50% win rate

                logging.info(f"Added new self-play opponent '{checkpoint_name}' to the pool. Pool size: {len(opponent_pool)}")

            # Plot training metrics every 100 episodes
            if SHOW_PLOTS and i_training % 100 == 0:
                plot_training_metrics(
                    episode_durations=episode_durations,
                    episode_rewards=episode_rewards,
                    episode_losses=episode_losses,
                    current_episode=i_training,
                    episode_update_iter=EPISODE_UPDATE_ITER
                )

            # Save a checkpoint of the agent every CHECKPOINT_ITER episodes
            if (i_training % CHECKPOINT_ITER == 0):
                agent.saveModel(MODEL_NAME, i_training)

            i_training += 1

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
        total_reward = 0
        all_states = []
        all_states_opponent = []
        all_actions = []
        all_actions_opponent = []
        all_rewards = []
        all_next_states = []
        all_next_states_opponent = []
        all_dones = []
        all_infos = []

        # if we use self play, we want to play 1/2 times against the own agent
        if self_opponent is not None and i_training % 2 == 0:
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
            action = torch.tensor(action, device = DEVICE, dtype = torch.float32)
            action_opponent = torch.tensor(action_opponent, device = DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = DEVICE, dtype = torch.float32)
            done = torch.tensor(terminated or truncated, device = DEVICE,
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
            # # Train the opponent agents NOT SUPPORTED YET
            # for o_name, opp in opponent_pool.items():
            #     if o_name != USE_ALGO:
            #         # you update all the algorithms
            #         _ = opp.optimize(memory = memory_opponent, episode_i = i_training)

            # After optimization, we can log some *more* statistics
            t_end = time.time()
            episode_time = t_end - t_start
            episode_training_statistics.append(training_statistics)
            other_statistics = " | ".join([f"{key}: {value:.4f}" for key, value in training_statistics.items()])
            logging.info(
                f"Training Iter: {i_training} | Req. Time: {episode_time:.4f} sec. | Req. Steps: {episode_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
                f" Opponent: {opponent_name} | {other_statistics}")

        # (Optional): After every 100 episodes ...
        if i_training % 5 == 0:
            # ... (Optional): If self play is activated, you update the self opponent with the current weights
            if SELF_PLAY:
                self_opponent.import_checkpoint(agent.export_checkpoint())
                logging.info(f"Training Iter: {i_training} | self updated weights for the agent {agent}")

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

        # Plot statistics every 100 episodes
        if SHOW_PLOTS and i_training % 100 == 0:
            plot_training_metrics(episode_durations = episode_durations, episode_rewards = episode_rewards,
                                  episode_losses = episode_training_statistics, current_episode = i_training,
                                  episode_update_iter = EPISODE_UPDATE_ITER)

        # Frequently after some episodes, we save a checkpoint of our model
        if (i_training % CHECKPOINT_ITER == 0):
            agent.saveModel(MODEL_NAME, i_training)


def do_multiple_testing(env, agent, opponent_pool):
    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_epsilon = []
    win_rates = {opponent: 0.5 for opponent in opponent_pool}  # Start with 50% win rate for each opponent

    for i_test in range(1, NUM_TRAINING_EPISODES + 1):
        # We track for each episode how high the reward was
        total_reward = 0

        # Select an opponent based on win rates (e.g., preference for challenging opponents)
        opponent = random.choices(opponent_pool, weights = [1 - abs(0.5 - win_rates[o]) for o in opponent_pool])[0]

        # We reset the environment
        state, info = env.reset()
        state_opponent = env.obs_agent_two()
        # Convert state to torch
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)
        state_opponent = torch.tensor(state_opponent, device = DEVICE, dtype = torch.float32)

        for step in count(start = 1):
            # Render the scene every 10th episode
            # env.render(mode = "human" if i_test % 10 == 0 else None)

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

            # Update the state
            state = next_state
            state_opponent = next_state_opponent

            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                episode_durations.append(step)
                episode_rewards.append(total_reward)
                logging.info(f"Test Iter: {i_test} | Req. Steps: {step} | Total reward: {total_reward}")
                break

    return episode_durations, episode_rewards


def main():
    # Let's first set the seed
    initSeed(seed = SEED, device = DEVICE)

    # If you want to use TF32 instead of Float 32, you can activate it here. Might not be available for old GPUs
    if USE_TF32:
        torch.set_float32_matmul_precision("high")

    # Initialize the environment
    env = initEnv(USE_ENV, RENDER_MODE, NUMBER_DISCRETE_ACTIONS)

    # Choose which algorithm to pick to initialize the agent
    agent = initAgent(USE_ALGO, env = env, device = DEVICE, agent_settings=AGENT_SETTINGS)

    # Init the memory
    memory = ReplayMemory(capacity = BUFFER_SIZE, device = DEVICE)

    # Setup Logging
    setupLogging(model_name = MODEL_NAME)

    # Log the settings.py such that we can save the settings under which we did the training
    logging.info(yaml.dump(SETTINGS, default_flow_style = False, sort_keys = False, allow_unicode = True))

    # Setup agent for training
    agent.setMode(eval = False)  # Set the agent in training mode
    if CHECKPOINT_NAME is not None:
        agent.loadModel(CHECKPOINT_NAME)
    logging.info(f"The configuration was valid! Start training 💪")

    # Training loop
    if USE_ENV == HOCKEY:
        # Only in the Hockey env, we need some opponent_pool to play against
        random_agent = initAgent(use_algo = RANDOM_ALGO, env = env, device = DEVICE)
        weak_comp_agent = initAgent(use_algo = WEAK_COMP_ALGO, env = env, device = DEVICE)
        strong_comp_agent = initAgent(use_algo = STRONG_COMP_ALGO, env = env, device = DEVICE)
        # dqn_agent = initAgent(use_algo = DQN_ALGO, env = env, device = DEVICE)
        # ppo_agent = initAgent(use_algo = PPO_ALGO, env = env, device = DEVICE)
        # ddpg_agent = initAgent(use_algo = DDPG_ALGO, env = env, device = DEVICE)
        # td3_agent = initAgent(use_algo = TD3_ALGO, env = env, device = DEVICE)
        # sac_agent = initAgent(use_algo = SAC_ALGO, env = env, device = DEVICE)
        # mpo_agent = initAgent(use_algo = MPO_ALGO, env = env, device = DEVICE)
        # tdmpc2_agent = initAgent(use_algo = TDMPC2_ALGO, env = env, device = DEVICE)

        # Currently, we do not allow the opponent networks to train as well. This might be an extra feature
        random_agent.setMode(eval = True)
        weak_comp_agent.setMode(eval = True)
        strong_comp_agent.setMode(eval = True)
        # dqn_agent.setMode(eval = True)
        # ppo_agent.setMode(eval = True)
        # ddpg_agent.setMode(eval = True)
        # td3_agent.setMode(eval = True)
        # sac_agent.setMode(eval = True)
        # mpo_agent.setMode(eval = True)

        opponent_pool = {
            RANDOM_ALGO: random_agent,
            WEAK_COMP_ALGO: weak_comp_agent,
            STRONG_COMP_ALGO: strong_comp_agent,
            # DQN_ALGO: dqn_agent,
            # PPO_ALGO: ppo_agent,
            # DDPG_ALGO: ddpg_agent,
            # TD3_ALGO: td3_agent,
            # SAC_ALGO: sac_agent,
            # MPO_ALGO: mpo_agent,
        }

        if SELF_PLAY:
            self_opponent = initAgent(use_algo = USE_ALGO, env = env, device = DEVICE)
            self_opponent.setMode(eval = True)
            self_opponent.import_checkpoint(agent.export_checkpoint())
        else:
            self_opponent = None

        if USE_ALGO == TDMPC2_ALGO:
            do_tdmpc2_hockey_training(env = env, agent = agent, memory = memory, opponent_pool = opponent_pool,
                                      self_opponent = self_opponent)
        elif USE_ALGO == SAC_ALGO:
            do_sac_hockey_training(env = env, agent = agent, memory = memory, opponent_pool = opponent_pool)
        else:
            do_hockey_training(env = env, agent = agent, memory = memory, opponent_pool = opponent_pool)

    # If you use another env (e.g. Pendulum), train normally
    else:
        if USE_ALGO == TDMPC2_ALGO:
            do_tdmpc2agent_other_env_training(env = env, agent = agent, memory = memory)
        else:
            do_other_env_training(env = env, agent = agent, memory = memory)

    # Testing loop
    logging.info("Training is done! Now we will do some testing!")
    agent.setMode(eval = True)  # Set the agent in eval mode

    if USE_ENV == HOCKEY:
        ...
        # test_durations, test_rewards = do_multiple_testing(env = env, agent = agent, opponent_pool = opponent_pool)
    else:
        do_other_env_testing(env = env, agent = agent)
    logging.info(f"Finished! 🚀")


if __name__ == '__main__':
    main()
