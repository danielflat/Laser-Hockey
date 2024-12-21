import logging
import time
from itertools import count

import numpy as np
import torch
import yaml

from src.config import CONFIG, DEVICE, HYPERPARAMS, MODEL_NAME, OPTIMIZER, OPTIONS, SEED, USE_ALGO, USE_ENV
from src.replaymemory import ReplayMemory
from src.util.contract import initAgent, initEnv, setupLogging
from src.util.directoryutil import get_path
from src.util.plotutil import plot_training_metrics


def main():
    # Let's first set the seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    if OPTIONS["USE_TF32"]: # activate TF32. Might not be available for old GPUs
        torch.set_float32_matmul_precision("high")

    # Initialize the environment
    env = initEnv(USE_ENV, HYPERPARAMS["NUMBER_DISCRETE_ACTIONS"])

    # Get some priors regarding the environment
    #state_space_shape: tuple[int, ...] = env.observation_space.shape

    # Choose which algorithm to pick to initialize the agent
    agent = initAgent(USE_ALGO, env=env, options=OPTIONS, optim=OPTIMIZER, hyperparams=HYPERPARAMS, device=DEVICE)

    # Init the memory
    memory = ReplayMemory(capacity=HYPERPARAMS["BUFFER_SIZE"])

    # Setup Logging
    setupLogging()

    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_epsilon = []

    # Log the Config
    logging.info(yaml.dump(CONFIG, default_flow_style=False, sort_keys=False, allow_unicode=True))

    # Training loop
    logging.info(f"The configuration was valid! Start training 💪")
    agent.setMode(eval=False)  # Set the agent in training mode
    state, info = env.reset(seed=SEED)

    for i_training in range(1, HYPERPARAMS["NUM_EPISODES"] + 1):
        # We track for each episode how high the reward was
        t_start = time.time()
        total_reward = 0

        # Convert state to torch
        state = torch.from_numpy(state).to(DEVICE)

        for step in count(start=1):
            # choose the action
            action = agent.act(state)

            # perform the action
            next_state, reward, terminated, truncated, info = env.step(action)

            # track the total reward
            total_reward += reward

            # Convert quantities into tensors
            action = torch.tensor(action, device=DEVICE, dtype=torch.int64)
            reward = torch.tensor(reward, device=DEVICE, dtype=torch.float32)
            done = torch.tensor(terminated or truncated, device=DEVICE,
                                dtype=torch.int)  # to be able to do arithmetics with the done signal, we need an int
            next_state = torch.tensor(next_state, device=DEVICE)

            # Store this transition in the memory
            memory.push(state, action, reward, next_state, done, info)

            # Update the state
            state = next_state
            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                episode_durations.append(step)
                break

        # after an episode is done, we want to optimize our agent
        losses = agent.optimize(memory=memory, episode_i=i_training)
        # clear the memory
        # memory.clear()

        # after some time, we save a checkpoint of our model
        if (i_training % HYPERPARAMS["CHECKPOINT_ITER"] == 0):
            agent.saveModel(get_path(f"output/checkpoints/{MODEL_NAME}_{i_training:05}.pth"))

        # some statistic magic
        t_end = time.time()
        episode_time = t_end - t_start
        episode_rewards.append(total_reward)
        episode_losses.append(losses)
        episode_epsilon.append(agent.epsilon)
        logging.info(
            f"Training Iter: {i_training} | Req. Steps: {episode_durations[i_training - 1]} | Total reward: {total_reward:.4f} |"
            f" Avg. Loss: {np.array(losses).mean():.4f} | Epsilon: {agent.epsilon:.4f} | Req. Time: {episode_time:.4f} sec.")
        if OPTIONS["SHOW_PLOTS"]:
            plot_training_metrics(episode_durations=episode_durations, episode_rewards=episode_rewards,
                                  episode_losses=episode_losses, current_episode=i_training)

        # reset the environment
        state, info = env.reset()

    # Now, we do some testing
    logging.info("Training is done! Now we will do some testing!")
    agent.setMode(eval=True)
    test_durations = []
    test_rewards = []

    for i_test in range(1, HYPERPARAMS["NUM_TEST_EPISODES"] + 1):
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
