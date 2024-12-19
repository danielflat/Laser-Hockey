from itertools import count

import numpy as np
import gymnasium

import hockey.hockey_env as h_env
import torch

from src.agents.dqnagent import DQNAgent
from src.replaymemory import ReplayMemory
from src.util.constants import ADAMW, DQN, HOCKEY, L1, PENDULUM, SUPPORTED_ENVIRONMENTS
from src.util.discreteactionmapper import DiscreteActionWrapper
from src.util.plotutil import plot_training_metrics

# Settings
SEED = 24  # The seed that we want to use
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # On which machine is it running?
USE_ENV = PENDULUM  # The used environment
USE_ALGO = DQN  # The used algorithm
OPTIONS = {
    "SELF_TRAINING": False,  # If the agent should play against itself like in AlphaGo  # If a target net is used
    "USE_TARGET_NET": True,  # If a target net is used
    "TARGET_NET_UPDATE_ITER": 1, # if "USE_TARGET_NET" == TRUE: int: Gives the number of iterations for update the target net, otherwise not relevant. If == 1, you update at every step.
    "USE_SOFT_UPDATES": True,  # If the target network is updated. True = softly, False = hardly
    "NUMBER_DISCRETE_ACTIONS": 10,  # If None, you use a continuous action space, else you use a discrete action set
    "LOSS_FUNCTION": L1,  # Which optimizer to use?
    "USE_GRADIENT_CLIPPING": True,  # If the gradients should be clipped
    "GRADIENT_CLIPPING_VALUE": 100,  # The gradient clipping value
    "SHOW_PLOTS": True, # If you want to plot statistics after each episode
}
HYPERPARAMS = {
    "NUM_EPISODES": 1000,  # How many training episodes should be run?
    "NUM_TEST_EPISODES": 100,  # How many test episodes should be run?
    "OPT_ITER": 100,  # How many iterations should be done for gradient descent after each episode?
    "BATCH_SIZE": 128,  # The batch size for doing gradient descent
    "BUFFER_SIZE": 100000,  # How many items can be stored in the replay buffer?
    "DISCOUNT": 1,  # The discount factor for the TD error
    "EPSILON": 1,  # The initial exploration rate for the epsilon greedy algo
    "EPSILON_MIN": 0.01,  # Minimum exploration rate
    "EPSILON_DECAY": 0.999,  # Decay rate per episode, if =1, no epsilon decay
    "TAU": 0.005,  # Soft update parameter
}
OPTIMIZER = {
    "OPTIM_NAME": ADAMW,  # Which optimizer to use
    "LEARNING_RATE": 3e-4,  # The learning rate for the agent
    "BETAS": (0.9, 0.99),  # The beta1, beta2 parameters of Adam
    "EPS": 1e-8,  # eps Adam param
    "WEIGHT_DECAY": 1e-2,  # The weight decay rate
    "USE_FUSION": True if torch.cuda.is_available() else False
    # if we have CUDA, we can use the fusion implementation of Adam -> Faster
}


def main():
    # Let's first set the seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if DEVICE == "cuda":
        torch.cuda.manual_seed(SEED)

    # Initialize the environment
    if USE_ENV == HOCKEY:
        env = h_env.HockeyEnv()
    elif USE_ENV != HOCKEY and USE_ENV in SUPPORTED_ENVIRONMENTS:
        env = gymnasium.make(USE_ENV)
    else:
        raise Exception(f"The environment '{USE_ENV}' is not supported! Please choose another one!")

    # Get some priors regarding the environment to initialize our objects
    state_space_shape: tuple[int, ...] = env.observation_space.shape
    action_space_shape: tuple[int, ...] = env.action_space.shape

    # Choose which algorithm to pick to initialize the agent
    if USE_ALGO == DQN:
        number_discrete_actions_ = OPTIONS["NUMBER_DISCRETE_ACTIONS"]
        if number_discrete_actions_ is not None and number_discrete_actions_ > 0:
            # We need to discretize the action space of the environment by using the DiscreteActionWrapper from the Lecture
            env = DiscreteActionWrapper(env, bins=number_discrete_actions_)
            action_size: int = env.action_space.n
            agent = DQNAgent(state_shape=state_space_shape, action_size=action_size,
                             options=OPTIONS, optim=OPTIMIZER, hyperparams=HYPERPARAMS, device=DEVICE)
        else:
            raise Exception(
                f"The environment '{USE_ALGO}' cannot work with a continuous action space. Please set the number of discrete actions by setting the variable '{OPTIONS['NUMBER_DISCRETE_ACTIONS']}' to a number!")
    else:
        raise Exception(f"The algorithm '{USE_ALGO}' is not supported! Please choose another one!")
    print(f"Seed: {SEED}, Device: {DEVICE}, Env: {USE_ENV}, Algo: {USE_ALGO}")

    # Init the memory
    memory = ReplayMemory(capacity=HYPERPARAMS["BUFFER_SIZE"])

    episode_durations = []
    episode_rewards = []
    episode_losses = []
    episode_epsilon = []

    # Training loop
    state, info = env.reset(seed=SEED)
    agent.setMode(eval=False)  # Set the agent in training mode
    print(f"The configuration was valid! Start training 💪")

    for i_test in range(1, HYPERPARAMS["NUM_EPISODES"] + 1):
        # We track for each episode how high the reward was
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
            action = torch.tensor(action, device=DEVICE, dtype=torch.long)
            reward = torch.tensor(reward, device=DEVICE)
            done = torch.tensor(terminated or truncated, device=DEVICE,
                                dtype=torch.int)  # to be able to do arithmetics with the done signal, we need an int
            if done:
                # Set next state to a fake 'nan' tensor if this is the last transition
                next_state = torch.full(state_space_shape, fill_value=torch.nan, device=DEVICE)

            else:
                next_state = torch.tensor(next_state, device=DEVICE)

            # Store this transition in the memory
            memory.push(state, action, reward, next_state, done, info)

            if done:
                # If this transition is the last, safe the number of done steps in the env. and end this episode
                episode_durations.append(step)
                break
            else:
                # Update the state
                state = next_state

        # after an episode is done, we want to optimize our agent
        losses = agent.optimize(memory=memory)

        # some statistic magic
        episode_rewards.append(total_reward)
        episode_losses.append(losses)
        episode_epsilon.append(agent.epsilon)
        print(
            f"Training Iter: {i_test} | Req. Steps: {episode_durations[i_test - 1]} | Total reward: {total_reward:.4f} |"
            f" Avg. Loss: {np.array(losses).mean():.4f} | Epsilon: {agent.epsilon:.4f}")
        if OPTIONS["SHOW_PLOTS"]:
            plot_training_metrics(episode_durations=episode_durations, episode_rewards=episode_rewards,
                                  episode_losses=episode_losses, current_episode=i_test)

        # clear the memory
        memory.clear()

        # reset the environment
        state, info = env.reset()

    # Now, we do some testing
    print("Training is done! Now we will do some testing!")
    agent.setMode(eval=True)
    test_durations = []

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
                print(f"Test Iter: {i_test} | Req. Steps: {step} | Total reward: {total_reward}")
                break
            else:
                # Update the state
                state = torch.from_numpy(next_state).to(DEVICE)

        # clear the memory
        memory.clear()

        # reset the environment
        state, info = env.reset()
    print(f"Finished! 🚀")


if __name__ == '__main__':
    main()
