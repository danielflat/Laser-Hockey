"""
In this file, we check our settings setup.
Furthermore, we here check, if the settings are valid and if not, we throw an error.
In addition, this file provides helper functions in order to initialize our commonly required objects,
e.g. optimizers to prevent code duplicates.

Author: Daniel
"""
import logging
import time
from time import strftime

import gymnasium
from torch import device

import hockey.hockey_env as h_env
from src.agents.dqnagent import DQNAgent
from src.util.constants import DQN, HOCKEY, SUPPORTED_ENVIRONMENTS
from src.util.directoryutil import get_path
from src.util.discreteactionmapper import DiscreteActionWrapper


def initEnv(use_env: str, number_discrete_actions: int):
    if use_env not in SUPPORTED_ENVIRONMENTS:
        Exception(f"The environment '{use_env}' is not supported! Please choose another one!")

    if use_env == HOCKEY:
        env =  h_env.HockeyEnv()
    else:
        env =  gymnasium.make(use_env)

    # if we use a discrete action space, we have to discrete the env before
    if number_discrete_actions is not None and number_discrete_actions > 0:
        env = DiscreteActionWrapper(env, bins=number_discrete_actions)

    return env


def initAgent(use_algo: str, env,
              options: dict, optim:dict, hyperparams:dict, device: device):
    """
    Initialize the agent based on the config
    """
    if use_algo == DQN:
        state_space_shape: tuple[int, ...] = env.observation_space.shape
        action_size: int = env.action_space.n
        return DQNAgent(state_shape=state_space_shape, action_size=action_size, options=options, optim=optim, hyperparams=hyperparams, device=device)
    else:
        raise Exception(f"The algorithm '{use_algo}' is not supported! Please choose another one!")


def setupLogging():
    """
    Configure logging to output to a file
    """
    logging.basicConfig(
        filename=get_path(f"output/logging/{strftime('%Y-%M-%d %H_%M_%S', time.localtime())}.txt"),  # Log file name
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # (optional) Add a console handler such that you output also the logging to the console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)






