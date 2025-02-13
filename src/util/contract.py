"""
In this file, we check our settings setup.
Furthermore, we here check, if the settings are valid and if not, we throw an error.
In addition, this file provides helper functions in order to initialize our commonly required objects,
e.g. optimizers to prevent code duplicates.

Author: Daniel
"""
from __future__ import annotations

import gymnasium
import logging
import numpy as np
import os
import torch
from torch import device

import hockey.hockey_env as h_env
from src.agent import Agent
from src.agents.compagent import CompAgent
from src.agents.ddpgagent import DDPGAgent
from src.agents.dqnagent import DQNAgent
from src.agents.mpoagent import MPOAgent
from src.agents.ppoagent import PPOAgent
from src.agents.randomagent import RandomAgent
from src.agents.sac import SoftActorCritic
from src.agents.td3agent import TD3Agent
from src.agents.tdmpc2agent import TDMPC2Agent
from src.settings import AGENT_SETTINGS, DDPG_SETTINGS, DQN_SETTINGS, MPO_SETTINGS, PPO_SETTINGS, SAC_SETTINGS, \
    TD3_SETTINGS, TD_MPC2_SETTINGS
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, PPO_ALGO, RANDOM_ALGO, SAC_ALGO, \
    STRONG_COMP_ALGO, SUPPORTED_ALGORITHMS, \
    SUPPORTED_ENVIRONMENTS, \
    SUPPORTED_RENDER_MODES, \
    TD3_ALGO, TDMPC2_ALGO, WEAK_COMP_ALGO
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


def initEnv(use_env: str, render_mode: str | None, number_discrete_actions: None | int, proxy_rewards: bool = False):
    if use_env not in SUPPORTED_ENVIRONMENTS:
        raise Exception(f"The environment '{use_env}' is not supported! Please choose another one!")
    if render_mode not in SUPPORTED_RENDER_MODES:
        raise Exception(f"The render mode '{render_mode}' is not supported! Please choose another one!")

    if use_env == HOCKEY:
        env = h_env.HockeyEnv(proxy_rewards=proxy_rewards)
    else:
        env = gymnasium.make(use_env, render_mode = render_mode)

    # if we use a discrete action space, we have to discrete the env before
    if number_discrete_actions is not None and number_discrete_actions > 0:
        env = DiscreteActionWrapper(env, bins = number_discrete_actions)

    return env

def initValEnv():
    env = h_env.HockeyEnv_BasicOpponent(weak_opponent=True)

    return env

def initAgent(use_algo: str, env, device: device,
              checkpoint_name: str | None,
              agent_settings: dict = AGENT_SETTINGS, dqn_settings: dict = DQN_SETTINGS,
              ppo_settings: dict = PPO_SETTINGS,
              ddpg_settings: dict = DDPG_SETTINGS, td3_settings: dict = TD3_SETTINGS, sac_settings: dict = SAC_SETTINGS,
              mpo_settings: dict = MPO_SETTINGS,
              td_mpc2_settings: dict = TD_MPC2_SETTINGS) -> Agent:
    """
    Initialize the agent based on the config
    """

    state_space = env.observation_space
    action_space = env.action_space

    agent = None

    if use_algo in SUPPORTED_ALGORITHMS:
        if use_algo == DQN_ALGO:
            agent = DQNAgent(state_space = state_space, action_space = action_space, agent_settings = agent_settings,
                            dqn_settings = dqn_settings, device = device)
        elif use_algo == PPO_ALGO:
            agent = PPOAgent(state_space = state_space, action_space = action_space,
                            agent_settings = agent_settings, ppo_settings = ppo_settings, device = device)
        elif use_algo == DDPG_ALGO:
            agent = DDPGAgent(observation_space = env.observation_space, action_space = env.action_space,
                             agent_settings = agent_settings, ddpg_settings = ddpg_settings, device = device)
        elif use_algo == TD3_ALGO:
            agent = TD3Agent(state_space = state_space, action_space = action_space,
                            agent_settings = agent_settings, td3_settings = td3_settings, device = device)
        elif use_algo == SAC_ALGO:
            agent = SoftActorCritic(
                state_space = state_space,
                action_space = action_space,
                agent_settings = agent_settings,
                device = device,
                sac_settings = sac_settings
            )
        elif use_algo == MPO_ALGO:
            agent = MPOAgent(
                state_space = state_space,
                action_space = action_space,
                agent_settings = agent_settings,
                device = device,
                mpo_settings=mpo_settings,
                env=env
            )
        elif use_algo == RANDOM_ALGO:
            agent = RandomAgent(env = env, agent_settings = agent_settings, device = device)
        elif use_algo == WEAK_COMP_ALGO:
            agent = CompAgent(is_Weak = True, agent_settings = agent_settings, device = device)
        elif use_algo == STRONG_COMP_ALGO:
            agent = CompAgent(is_Weak = False, agent_settings = agent_settings, device = device)
        elif use_algo == TDMPC2_ALGO:
            agent = TDMPC2Agent(
                state_space = state_space,
                action_space = action_space,
                agent_settings = agent_settings,
                td_mpc2_settings = td_mpc2_settings,
                device = device,
            )
    else:
        raise Exception(f"The algorithm '{use_algo}' is not supported! Please choose another one!")

    if checkpoint_name is not None:
        agent.loadModel(checkpoint_name)
    return agent


def setupLogging(model_name: str):
    """
    Configure logging to output to a file
    """
    # First, build the log file path
    log_path = get_path(f"output/logging/{model_name}.txt")
    
    # Make sure the directory exists; create it if necessary
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        filename = log_path,  # Log file name
        level=logging.INFO,  # Set the logging level
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # (optional) Add a console handler such that you output also the logging to the console as well
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
