import gymnasium
import logging
import numpy as np
import os
import torch
from torch import device, nn
from typing import Any, Dict

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path
from src.util.noiseutil import initNoise

"""
Author: Daniel Flat
TODOS:
    - torch.compile does not work properly here.
    - when deactivate bfloat16, the agent is currently faster.
"""


class Actor(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()

        self.actor_net = nn.Sequential(
            nn.Linear(observation_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, action_size),
            nn.Tanh())

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor_net(state)
        return action


class Critic(nn.Module):
    def __init__(self, observation_size: int, action_size: int):
        super().__init__()

        self.critic_net = nn.Sequential(
            nn.Linear(observation_size + action_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        q_value = self.critic_net(input)
        return q_value


class ActorCritic:
    """
    A top level class that combines the actor and critic network.
    """

    def __init__(self, state_size: int, action_size: int, use_compile: bool, device: device):
        super().__init__()

        self.actor = Actor(state_size, action_size)
        self.actor.to(device)

        self.critic = Critic(state_size, action_size)
        self.critic.to(device)

        # if use_compile:
        #     self.actor = torch.compile(self.actor)
        #     self.critic = torch.compile(self.critic)
        # else:
        #     self.actor = actor
        #     self.critic = critic

    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor.forward(state)
        return action

    def QValue(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.hstack((state, action))
        q_value = self.critic.forward(input)
        return q_value

    def to(self, device: device):
        self.actor.actor_net.to(device)
        self.critic.critic_net.to(device)

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()


class DDPGAgent(Agent):
    def __init__(self, observation_space: gymnasium.spaces.box.Box, action_space: gymnasium.spaces.box.Box,
                 agent_settings: dict, ddpg_settings: dict,
                 device: device):
        super().__init__(agent_settings = agent_settings, device = device)

        self.isEval = None

        self.observation_space = observation_space
        self.action_space = action_space
        observation_size = observation_space.shape[0]
        self.action_size = self.get_num_actions(action_space)
        self.min_action = action_space.low[:self.action_size]
        self.min_action_torch = torch.tensor(self.min_action, device = self.device)
        self.max_action = action_space.high[:self.action_size]
        self.max_action_torch = torch.tensor(self.max_action, device = self.device)

        self.noise = initNoise(action_shape = (self.action_size,), noise_settings = ddpg_settings["NOISE"],
                               device = self.device)
        self.noise_factor = ddpg_settings["NOISE"]["NOISE_FACTOR"]

        # Define the Q-Network
        self.origin_net = ActorCritic(state_size = observation_size, action_size = self.action_size,
                                      use_compile = self.USE_COMPILE, device = self.device)

        # If you want to use a target network, it is defined here
        if self.use_target_net:
            self.target_net = ActorCritic(state_size = observation_size, action_size = self.action_size,
                                          use_compile = self.USE_COMPILE, device = self.device)
            # Copy the networks
            self._copyNets()

        # # Define the Optimizer
        self.actor_optim = self.initOptim(optim = ddpg_settings["ACTOR"]["OPTIMIZER"],
                                          parameters = self.origin_net.actor.parameters())
        self.critic_optim = self.initOptim(optim = ddpg_settings["CRITIC"]["OPTIMIZER"],
                                           parameters = self.origin_net.critic.parameters())

        # Define Loss function
        self.criterion = self.initLossFunction(loss_name = ddpg_settings["CRITIC"]["LOSS_FUNCTION"])

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"DDPGAgent"

    @torch.no_grad()
    def act(self, state: torch.Tensor) -> np.ndarray:
        """
        The Agent chooses an action.
        In Evaluation mode, we always exploit the best action.
        In Training mode, we sample an action based on epsilon greedy with the given epsilon hyperparam.
        :param state: The state
        :return: The action
        """

        # In evaluation mode, we always exploit
        if self.isEval:
            greedy_action = self.origin_net.greedyAction(state)
            return greedy_action.cpu().numpy()

        # In training mode, use epsilon greedy action sampling
        elif not self.isEval:
            proposed_action = self.origin_net.greedyAction(state)
            noise = self.noise_factor * self.noise.sample()
            noisy_action = proposed_action.cpu().numpy() + noise
            normalized_action = np.clip(noisy_action, self.min_action, self.max_action)
            return normalized_action

    def optimize(self, memory: ReplayMemory, episode_i: int) -> Dict[str, Any]:
        statistics_episode = []

        for i in range(1, self.opt_iter + 1):
            # Step 01: Sample a batch from the memory
            state, action, reward, next_state, done = memory.sample_horizon(batch_size = self.batch_size)

            # Step 02: Update the agent.
            statistics_iter = self._update(state, action, reward, next_state, done)

            # Step 03: Keep track of the loss
            statistics_episode.append(statistics_iter)

        # Step 04: After some time, update the agent
        # NOTE: HERE, we the update frequency is w.t.r. the total number of episodes
        if episode_i % self.target_net_update_freq == 0:
            self._copyNets()

        # We sum up the statistics from each optimization iteration by accumulating statistical quantities
        # such that we can log it
        sum_up_statistics_training_iter = {
            "Avg. Total Loss": np.array([episode["total_loss"] for episode in statistics_episode]).mean(),
            "Avg. Actor Loss": np.array([episode["actor_loss"] for episode in statistics_episode]).mean(),
            "Avg. Critic Loss": np.array(
                [episode["critic_loss"] for episode in statistics_episode]).mean(),
            "Avg. Actor Grad Norm": np.array([episode["actor_grad_norm"] for episode in statistics_episode]).mean(),
            "Avg. Critic Grad Norm": np.array(
                [episode["critic_grad_norm"] for episode in statistics_episode]).mean(),
        }

        return sum_up_statistics_training_iter

    def _update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor,
                done: torch.Tensor):

        # critic update
        if self.USE_BF_16:
            with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                critic_loss = self.critic_forward(action, done, next_state, reward, state)
        else:
            critic_loss = self.critic_forward(action, done, next_state, reward, state)

        # critic backward step
        self.critic_optim.zero_grad()
        critic_loss.backward()
        if self.use_norm_clipping:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(parameters = self.origin_net.critic.parameters(),
                                                              max_norm = self.norm_clipping_value,
                                            foreach = self.use_clip_foreach)
        else:
            critic_grad_norm = None
        self.critic_optim.step()

        if self.USE_BF_16:
            with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                actor_loss = self.actor_forward(state)
        else:
            actor_loss = self.actor_forward(state)

        # Step 05: Optimize the actor net
        self.actor_optim.zero_grad()
        # actor backward step
        actor_loss.backward()
        if self.use_norm_clipping:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(parameters = self.origin_net.actor.parameters(),
                                                             max_norm = self.norm_clipping_value,
                                            foreach = self.use_clip_foreach)
        else:
            actor_grad_norm = None
        self.actor_optim.step()

        total_loss = actor_loss + critic_loss

        statistics = {
            "total_loss": total_loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }

        return statistics

    def actor_forward(self, state):
        # actor forward step: Maximize the actor network by go and maximize the critic network
        q_greedy = self.origin_net.QValue(state = state, action = self.origin_net.greedyAction(state))
        actor_loss = -torch.mean(q_greedy)
        return actor_loss

    def critic_forward(self, action, done, next_state, reward, state):
        # Step 01: Compute the td target
        with torch.no_grad():
            if self.use_target_net:
                q_target = self.target_net.QValue(state = next_state, action = self.target_net.greedyAction(next_state))
            else:
                q_target = self.origin_net.QValue(state = next_state, action = self.origin_net.greedyAction(next_state))
            td_target = reward + (1 - done) * self.discount * q_target
        # Step 02: Compute the prediction
        q_origin = self.origin_net.QValue(state, action)
        # Step 03: critic loss
        critic_loss = self.criterion(q_origin, td_target)
        return critic_loss

    def setMode(self, eval = False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.origin_net.eval()
        else:
            self.origin_net.train()

    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """

        checkpoint = self.export_checkpoint()

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok = True)

        torch.save(checkpoint, file_path)
        logging.info(f"Iteration: {iteration} DDPG checkpoint saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        try:
            checkpoint = torch.load(file_name, map_location = self.device)
            self.import_checkpoint(checkpoint)
            logging.info(f"Model loaded successfully from {file_name}")
        except FileNotFoundError:
            logging.error(f"Error: File {file_name} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {str(e)}")

    def import_checkpoint(self, checkpoint: dict) -> None:
        self.origin_net.actor.load_state_dict(checkpoint["origin_actor"])
        self.origin_net.critic.load_state_dict(checkpoint["origin_critic"])
        self.target_net.actor.load_state_dict(checkpoint["target_actor"])
        self.target_net.critic.load_state_dict(checkpoint["target_critic"])

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "origin_actor": self.origin_net.actor.state_dict(),
            "origin_critic": self.origin_net.critic.state_dict(),
            "target_actor": self.target_net.actor.state_dict(),
            "target_critic": self.target_net.critic.state_dict(),
        }
        return checkpoint



    def _copyNets(self):
        # Step 01: Copy the actor net
        self.updateTargetNet(soft_update = False, source = self.origin_net.actor,
                             target = self.target_net.actor)

        # Step 02: Copy the critic net
        self.updateTargetNet(soft_update = False, source = self.origin_net.critic,
                             target = self.target_net.critic)
