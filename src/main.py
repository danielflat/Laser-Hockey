import logging
import time
from itertools import count

import numpy as np
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_  # Map bool8 to bool_
import torch
import yaml

from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, DQN_SETTINGS, MAIN_SETTINGS, PPO_SETTINGS, TD3_SETTINGS, SAC_SETTINGS, MPO_SETTINGS, SETTINGS
from src.util.contract import initAgent, initEnv, initSeed, setupLogging
from src.util.plotutil import plot_training_metrics

from src.util.constants import SAC_ALGO

SEED = MAIN_SETTINGS["SEED"]
DEVICE = MAIN_SETTINGS["DEVICE"]
USE_TF32 = MAIN_SETTINGS["USE_TF32"]
USE_ENV = MAIN_SETTINGS["USE_ENV"]
RENDER_MODE = MAIN_SETTINGS["RENDER_MODE"]
NUMBER_DISCRETE_ACTIONS = MAIN_SETTINGS["NUMBER_DISCRETE_ACTIONS"]
USE_ALGO = MAIN_SETTINGS["USE_ALGO"]
MODEL_NAME = MAIN_SETTINGS["MODEL_NAME"]
BUFFER_SIZE = MAIN_SETTINGS["BUFFER_SIZE"]
NUM_TRAINING_EPISODES = MAIN_SETTINGS["NUM_TRAINING_EPISODES"]
NUM_TEST_EPISODES = MAIN_SETTINGS["NUM_TEST_EPISODES"]
EPISODE_UPDATE_ITER = MAIN_SETTINGS["EPISODE_UPDATE_ITER"]
SHOW_PLOTS = MAIN_SETTINGS["SHOW_PLOTS"]
CHECKPOINT_ITER = MAIN_SETTINGS["CHECKPOINT_ITER"]
BATCH_SIZE = AGENT_SETTINGS["BATCH_SIZE"]

def main():
    # Let's first set the seed
    initSeed(seed=SEED, device=DEVICE)

    # If you want to use TF32 instead of Float 32, you can activate it here. Might not be available for old GPUs
    if USE_TF32:
        torch.set_float32_matmul_precision("high")

    # Initialize the environment
    env = initEnv(USE_ENV, RENDER_MODE, NUMBER_DISCRETE_ACTIONS)

    # Get some priors regarding the environment
    # state_space_shape: tuple[int, ...] = env.observation_space.shape

    # Choose which algorithm to pick to initialize the agent
    agent = initAgent(USE_ALGO, env = env, agent_settings = AGENT_SETTINGS, dqn_settings = DQN_SETTINGS,
                      ppo_settings = PPO_SETTINGS, td3_settings = TD3_SETTINGS, sac_settings = SAC_SETTINGS, mpo_settings = MPO_SETTINGS, device = DEVICE)

    # Init the memory
    memory = ReplayMemory(capacity = BUFFER_SIZE)

    # Setup Logging
    setupLogging(model_name = MODEL_NAME)

    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_epsilon = []

    # Log the Config.py
    logging.info(yaml.dump(SETTINGS, default_flow_style = False, sort_keys = False, allow_unicode = True))

    # Training loop
    logging.info(f"The configuration was valid! Start training 💪")
    agent.setMode(eval=False)  # Set the agent in training mode
    state, info = env.reset(seed=SEED)

    for i_training in range(1, NUM_TRAINING_EPISODES + 1):
        # We track for each episode how high the reward was
        t_start = time.time()
        total_reward = 0

        # Convert state to torch
        state = torch.from_numpy(state).to(DEVICE)
        losses = np.array([])

        for step in count(start=1):
            # choose the action
            action = agent.act(state)
            # perform the action
            next_state, reward, terminated, truncated, info = env.step(action)

            # track the total reward
            total_reward += reward

            # Convert quantities into tensors
            if USE_ALGO == SAC_ALGO:
                action = torch.tensor(action, device=DEVICE, dtype=torch.float32)
            else:
                action = torch.tensor(action, device=DEVICE, dtype=torch.int64)
            reward = torch.tensor(reward, device=DEVICE, dtype=torch.float32)
            done = torch.tensor(terminated or truncated, device=DEVICE,
                                dtype=torch.int)  # to be able to do arithmetics with the done signal, we need an int
            next_state = torch.tensor(next_state, device=DEVICE)


            # Store this transition in the memory
            memory.push(state, action, reward, next_state, done, info)

            if USE_ALGO == SAC_ALGO:
                if len(memory) >= (100 * BATCH_SIZE):
                    step_losses = agent.optimize(memory = memory, episode_i = i_training)
                    losses = np.concatenate((losses, step_losses), axis=0)

            # Update the state
            state = next_state
            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                episode_durations.append(step)
                break

        # after each episode, we want to log some statistics
        episode_rewards.append(total_reward)

        if USE_ALGO == SAC_ALGO and len(memory) >= (100 * BATCH_SIZE):
            t_end = time.time()
            episode_time = t_end - t_start
            episode_losses.append(losses)
            episode_epsilon.append(agent.epsilon)
            logging.info(
                f"Training Iter: {i_training} | Req. Steps: {episode_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
                f" Avg. Loss: {np.array(losses).mean():.4f} | Epsilon: {agent.epsilon:.4f} | Req. Time: {episode_time:.4f} sec.")

        # After some episodes and collecting some data, we optimize the agent
        if not USE_ALGO == SAC_ALGO and i_training % EPISODE_UPDATE_ITER == 0:
            losses = agent.optimize(memory = memory, episode_i = i_training)

            # After optimization, we can log some *more* statistics
            t_end = time.time()
            episode_time = t_end - t_start
            episode_losses.append(losses)
            episode_epsilon.append(agent.epsilon)
            logging.info(
                f"Training Iter: {i_training} | Req. Steps: {episode_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
                f" Avg. Loss: {np.array(losses).mean():.4f} | Epsilon: {agent.epsilon:.4f} | Req. Time: {episode_time:.4f} sec.")

        # Plot every 1 episodes
        if SHOW_PLOTS and i_training % 1 == 0 and len(memory) >= (100 * BATCH_SIZE):
            plot_training_metrics(episode_durations=episode_durations, episode_rewards=episode_rewards,
                                  episode_losses = episode_losses, current_episode = i_training,
                                  episode_update_iter = EPISODE_UPDATE_ITER)

        # after some time, we save a checkpoint of our model
        if (i_training % CHECKPOINT_ITER == 0):
            agent.saveModel(MODEL_NAME, i_training)

        # reset the environment
        state, info = env.reset(
            seed = SEED + i_training)  # by resetting always a different but predetermined seed, we ensure the reproducibility of the results
        

    # Now, we do some testing
    logging.info("Training is done! Now we will do some testing!")
    agent.setMode(eval=True)
    test_durations = []
    test_rewards = []

    for i_test in range(1, NUM_TEST_EPISODES + 1):
        # We track for each episode how high the reward was
        total_reward = 0

        # Convert state to torch
        state = torch.from_numpy(state).to(DEVICE)

        for step in count(start=1):
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
                state = torch.from_numpy(next_state).to(DEVICE)

        # clear the memory
        memory.clear()

        # reset the environment
        state, info = env.reset()
    logging.info(f"Tests done! "
                 f"Durations average: {np.array(test_durations).mean():.4f} | Durations std. dev: {np.array(test_durations).std():.4f} | Durations variance: {np.array(test_durations).var():.4f} | "
                 f"Reward average: {np.array(test_rewards).mean():.4f} | Reward std. dev: {np.array(test_rewards).std():.4f} | Reward variance: {np.array(test_rewards).var():.4f}")
    logging.info(f"Finished! 🚀")


if __name__ == '__main__':
    main()
