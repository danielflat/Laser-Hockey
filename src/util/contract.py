"""
In this file, we check our settings setup.
Furthermore, we here check, if the settings are valid and if not, we throw an error.
In addition, this file provides helper functions in order to initialize our commonly required objects,
e.g. optimizers to prevent code duplicates.

Author: Daniel
"""
import logging

import gymnasium
import numpy as np
import torch
from torch import device

import hockey.hockey_env as h_env
from src.agents.dqnagent import DQNAgent
from src.agents.ppoagent import PPOAgent
from src.agents.td3agent import TD3Agent
from src.util.constants import DQN_ALGO, HOCKEY, PPO_ALGO, TD3_ALGO, SUPPORTED_ENVIRONMENTS, SUPPORTED_RENDER_MODES
from src.util.directoryutil import get_path
from src.util.discreteactionmapper import DiscreteActionWrapper


def initSeed(seed: int | None, device: device):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
    else:
        logging.warning("No seed was set!")


def initEnv(use_env: str, render_mode: str | None, number_discrete_actions: int):
    if use_env not in SUPPORTED_ENVIRONMENTS:
        raise Exception(f"The environment '{use_env}' is not supported! Please choose another one!")
    if render_mode not in SUPPORTED_RENDER_MODES:
        raise Exception(f"The render mode '{render_mode}' is not supported! Please choose another one!")

    if use_env == HOCKEY:
        env =  h_env.HockeyEnv()
    else:
        env = gymnasium.make(use_env, render_mode=render_mode)

    # if we use a discrete action space, we have to discrete the env before
    if number_discrete_actions is not None and number_discrete_actions > 0:
        env = DiscreteActionWrapper(env, bins=number_discrete_actions)

    return env


def initAgent(use_algo: str, env,
              agent_settings: dict, dqn_settings: dict, ppo_settings: dict, td3_settings: dict, device: device):
    """
    Initialize the agent based on the config
    """

    if use_algo == DQN_ALGO:
        state_space_shape: tuple[int, ...] = env.observation_space.shape
        action_size: int = env.action_space.n
        return DQNAgent(state_shape = state_space_shape, action_size = action_size, agent_settings = agent_settings,
                        dqn_settings = dqn_settings, device = device)
    elif use_algo == PPO_ALGO:
        state_space_shape: tuple[int, ...] = env.observation_space.shape
        action_size: int = env.action_space.n
        return PPOAgent(observation_size = state_space_shape[0], action_size = action_size,
                        agent_settings = agent_settings, ppo_settings = ppo_settings, device = device)
    elif use_algo == TD3_ALGO:
        state_space_shape: tuple[int, ...] = env.observation_space.shape
        action_space: tuple[int, ...] = env.action_space
        return TD3Agent(observation_size = state_space_shape[0], action_space = action_space,
                        agent_settings = agent_settings, td3_settings = td3_settings, device = device)
    else:
        raise Exception(f"The algorithm '{use_algo}' is not supported! Please choose another one!")


def setupLogging(model_name: str):
    """
    Configure logging to output to a file
    """
    logging.basicConfig(
        filename = get_path(f"output/logging/{model_name}.txt"),  # Log file name
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # (optional) Add a console handler such that you output also the logging to the console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)






