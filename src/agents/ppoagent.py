import logging
import os
from typing import List

import numpy as np
import torch
from torch import device, nn
from torch.distributions import Categorical

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path


class ActorCritic(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        # It yields the prob. distribution of the action space given the state the given state
        self.actor = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_size),
            nn.Softmax(dim=-1))

        # It yields the state function for the given state
        self.critic = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1))

    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action_probs = self.actor(state)
            greedyAction = torch.argmax(action_probs, dim = -1)
            return greedyAction


class PPOAgent(Agent):
    def __init__(self, state_space, action_space, agent_settings: dict, ppo_settings: dict, device: device):
        super().__init__(agent_settings = agent_settings, device = device)

        self.observation_size = state_space
        self.action_size = action_space
        state_size = state_space.shape[0]
        action_size = self.get_num_actions(action_space)

        self.eps_clip = ppo_settings["EPS_CLIP"]

        self.policy_net = ActorCritic(state_size = state_size, action_size = action_size)
        self.policy_net.to(device)

        self.policy_old = ActorCritic(state_size = state_size, action_size = action_size)
        self.policy_old.to(device)
        self.policy_old.eval()  # set old policy net always to eval mode
        self.policy_old.load_state_dict(self.policy_net.state_dict())  # copy the network

        self.optimizer = self.initOptim(optim = ppo_settings["OPTIMIZER"], parameters = self.policy_net.parameters())

        self.criterion = self.initLossFunction(loss_name = ppo_settings["LOSS_FUNCTION"])

        # Activate torch.compile if wanted
        # if self.USE_COMPILE:
        #     self.policy_net = torch.compile(self.policy_net)
        #     self.policy_old = torch.compile(self.policy_old)

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"PPOAgent"

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        This function is used to train and optimize the Q Network with the help of the replay memory.
        :return: A list of all losses during optimization
        """
        assert self.isEval == False, "Make sure to put the agent in training mode before calling the opt. routine"

        losses = []

        # Since we do Monte Carlo Estimation, we sample the whole trajectory of the episode
        batch_size = len(memory)
        state, action, reward, _, done, _ = memory.sample(batch_size, randomly = False)

        # We have to discount the reward w.r.t. the discount factors
        discounted_reward = self.discount_reward(reward, done)

        # Now, lets squeeze the action tensor
        action = action.squeeze(1)

        # First, we need the logprobs of the old policy
        old_log_probs = self.logprobs(state, action)

        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            self.optimizer.zero_grad()

            # Forward step
            loss = self.forward_pass(state, action, discounted_reward, old_log_probs)

            # Track the loss
            losses.append(loss.item())

            # Backward step
            loss.backward()
            # if we want to clip our gradients
            if self.use_gradient_clipping:
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(parameters=self.policy_net.parameters(),
                                                clip_value=self.gradient_clipping_value,
                                                foreach=self.use_clip_foreach)
            self.optimizer.step()


        # in PPO, we have to clear the memory after each optimization loop, since
        memory.clear()

        # adjust epsilon after each optimization
        self.adjust_epsilon(episode_i)

        # Update the old policy net
        self.updateOldPolicyNet(soft_update = self.use_soft_updates)

        return losses

    def act(self, state: torch.Tensor) -> int:
        with torch.no_grad():
            if self.isEval:
                # if you are in eval mode, get the greedy Action
                greedy_action = self.policy_net.greedyAction(state)
                return greedy_action.item()
            else:
                # In training mode, use epsilon greedy action sampling
                rdn = np.random.random()
                if rdn <= self.epsilon:
                    # Exploration. Take a random action
                    return np.random.randint(low = 0, high = self.action_size)
                else:
                    # Exploitation. take the actions w.r.t. the old policy
                    action_probs = self.policy_old.actor(state)
                    categorical = Categorical(action_probs)
                    action = categorical.sample()

                    return action.item()

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.policy_net.eval()
        else:
            self.policy_net.train()

    def forward_pass(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                     old_log_probs: torch.Tensor) -> torch.Tensor:
        # Step 01: Evaluate the actions
        logprobs, state_values, dist_entropy = self.evaluate(state, action)

        # since we have to be careful with softmax to sum up to 1, we cannot have these steps in the bfloat16 mode
        if self.USE_BF_16:
            with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                # Step 02: Compute the ratio
                ratios = torch.exp(logprobs - old_log_probs.detach())

                # Step 03: Compute the advantages
                advantages = reward - state_values.detach()

                # Step 04: Compute the surrogate loss
                objective = ratios * advantages
                objective_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Step 05: Finally the loss
                loss = -torch.min(objective, objective_clipped) + 0.5 * self.criterion(state_values,
                                                                                       reward) - 0.01 * dist_entropy

                return loss.mean()
        else:
            # Step 02: Compute the ratio
            ratios = torch.exp(logprobs - old_log_probs.detach())

            # Step 03: Compute the advantages
            advantages = reward - state_values.detach()

            # Step 04: Compute the surrogate loss
            objective = ratios * advantages
            objective_clipped = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # Step 05: Finally the loss
            loss = -torch.min(objective, objective_clipped) + 0.5 * self.criterion(state_values,
                                                                                   reward) - 0.01 * dist_entropy
            return loss.mean()

    def logprobs(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Here we calculate the log probabilities of the given action and the state. Can be performed on a single tensor or on a *batch*!
        :param state: The state
        :param action: the action
        :return: the log prob as a torch.Tensor
        """

        # Step 01: Get the action probabilities
        action_probs = self.policy_old.actor(state)

        # Step 02: Create a Categorical distribution
        dist = Categorical(action_probs)

        # Step 03: Return the log prob of the action given the state
        return dist.log_prob(action)

    def evaluate(self, state, action):
        action_probs = self.policy_net.actor(state)
        dist = Categorical(action_probs)

        # Step 01: First, get the log prob of the action given the state
        action_logprobs = dist.log_prob(action)

        # Step 02: Next, we get the entropy of the action space
        dist_entropy = dist.entropy()

        # Step 03: Finally, we acquire the value of the state.
        # We squeeze it to have (batch_size,) shape
        state_value = self.policy_net.critic(state).squeeze(1)

        return action_logprobs, state_value, dist_entropy


    def updateOldPolicyNet(self, soft_update: bool):
        """
        Updates the target network with the weights of the original one
        """
        assert self.use_target_net == True, "You must use have 'self.use_target == True' to call 'updateTargetNet()'"

        if soft_update:
            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′ where θ′ are the target net weights
            for target_param, param in zip(self.policy_old.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            # Do a hard parameter update. Copy all values from the origin to the target network
            self.policy_old.load_state_dict(self.policy_net.state_dict())

    def discount_reward(self, rewards, dones):
        discounted_rewards = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.discount * discounted_reward)
            discounted_rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        discounted_rewards = torch.tensor(discounted_rewards, dtype = torch.float32).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-5)
        return discounted_rewards

    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok = True)
        torch.save(self.policy_net.state_dict(), file_path)
        logging.info(f"Q network weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        self.policy_net.load_state_dict(torch.load(file_name))
        logging.info(f"Q network weights loaded successfully!")

    def import_checkpoint(self, checkpoint: dict) -> None:
        raise NotImplementedError

    def export_checkpoint(self) -> dict:
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
