import os
from typing import List

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path


class Critic(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: List[int], action_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Q value for the given state and action
        """
        # Action and State concatenated as input into the q network
        concat_input = torch.cat((state, action), dim = 1)
        return self.network(concat_input)

class Actor(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: List[int], action_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[2], action_size),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Actor values over all actions for the given state
        """
        action = self.network(state)
        return action

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Calculates the detached Actor Value as a numpy array 
        """
        with torch.no_grad():
            action = self.forward(x).squeeze().cpu().numpy()
            action = np.expand_dims(action, axis = 0)
            return action


class TD3Agent(Agent):
    def __init__(self, state_space, action_space, agent_settings: dict, td3_settings: dict, device: device):
        super().__init__(agent_settings = agent_settings, device = device)

        self.isEval = None

        self.state_space = state_space
        self.action_space = action_space
        self.action_low = torch.tensor(action_space.low, dtype = torch.float32, device = self.device)
        self.action_high = torch.tensor(action_space.high, dtype = torch.float32, device = self.device)
        self.action_scale = (self.action_high - self.action_low) / 2.0
        self.action_bias = (self.action_high + self.action_low) / 2.0
        state_size = state_space.shape[0]
        action_size = self.get_num_actions(action_space)

        self.policy_delay = td3_settings["POLICY_DELAY"]
        self.noise_clip = td3_settings["NOISE_CLIP"]

        # Here we have 2 Q Networks
        self.Critic1 = Critic(state_size=state_size,
                            hidden_sizes=[128, 128],
                            action_size=action_size).to(device)
        self.Critic2 = Critic(state_size=state_size,
                            hidden_sizes=[128, 128],
                            action_size=action_size).to(device)
        self.Actor = Actor(state_size=state_size,
                                    hidden_sizes=[128, 128, 64],
                                    action_size=action_size).to(device)
        self.Critic1_target = Critic(state_size=state_size,
                                    hidden_sizes=[128, 128],
                                    action_size=action_size).to(device)
        self.Critic2_target = Critic(state_size=state_size,
                                    hidden_sizes=[128, 128],
                                    action_size=action_size).to(device)
        self.Actor_target = Actor(state_size=state_size,
                                            hidden_sizes=[128, 128, 64],
                                            action_size=action_size).to(device)
        # Set the target nets in eval
        self.Critic1_target.eval()
        self.Critic2_target.eval()
        self.Actor_target.eval()

        # Copying the weights of the Q and Policy networks to the target networks
        self._copy_nets(soft_update = False)

        # Initializing the optimizers, TO DO: Use different learning rates for Q and Policy networks
        self.optimizer_critic = torch.optim.Adam(list(self.Critic1.parameters()) + list(self.Critic2.parameters()),
                                            lr = 0.0001,
                                            eps = 0.000001)
        self.optimizer_actor = torch.optim.Adam(self.Actor.parameters(),
                                            lr = 0.00001,
                                            eps = 0.000001)
        #self.critic = self.initOptim(optim=agent_settings["OPTIMIZER"], parameters=list(self.Critic1.parameters()) + list(self.Critic2.parameters()))
        #self.optimizer_actor = self.initOptim(optim=agent_settings["OPTIMIZER"], parameters=self.Actor.parameters())

        #Define Loss function
        self.criterion = torch.nn.SmoothL1Loss()

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        The Agent chooses an action.
        In Evaluation mode, we set the noise eps = 0
        In Training mode, we sample an action using actor network and exploration noise
        :param state: The state
        """
        if self.isEval:
            self.epsilon = 0

        with torch.no_grad():
            # action squeezed in -1 to 1 via tanh (+ noise)
            action_deterministic = self.Actor.forward(state)
            action = action_deterministic + torch.randn_like(action_deterministic) * self.epsilon
            # rescale the action to the action space
            action = action * self.action_scale + self.action_bias
            action = torch.clamp(action, self.action_low, self.action_high)

        action = action.squeeze().cpu().numpy()

        action = np.expand_dims(action, axis = 0)
        return action

    def calc_td_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the TD Target
        1. Sample the next action from the target Actor network using gaussian exploration noise
        2. Calculate the Q values for the next state and next action using both target Q networks
        3. Calculate the TD Target by bootstrapping the minimum of the two Q networks
        """

        with torch.no_grad():
            # Exploration noise
            noise = torch.clamp(
                torch.randn(self.action_space.shape[0], device = self.device) * self.epsilon,
                min = -self.noise_clip,
                max = self.noise_clip)

            if self.use_target_net:
                # 1. Next action via the target Actor network
                next_action = self.Actor_target.forward(next_state) * self.action_scale + self.action_bias
                next_action = torch.clamp(
                    next_action + noise,
                    min = self.action_low,  # Minimum action value
                    max = self.action_high)  # Maximum action value

                # 2. Forward pass for both Q networks
                q_prime1 = self.Critic1_target.forward(next_state, next_action)
                q_prime2 = self.Critic2_target.forward(next_state, next_action)

            else:
                #1. Next action via the target Actor network
                next_action = self.Actor.forward(next_state) * self.action_scale + self.action_bias
                next_action = torch.clamp(
                    next_action + noise,
                    min = self.action_low,  # Minimum action value
                    max = self.action_high)  # Maximum action value

                # 2. Forward pass for both Q networks
                q_prime1 = self.Critic1.forward(next_state, next_action)
                q_prime2 = self.Critic2.forward(next_state, next_action)

            # 3. Bootstrapping the minimum of the two Q networks
            td_target = reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2)

        return td_target

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        Compute forward and backward pass for the Q and Policy networks
        """
        assert self.isEval == False
        # Storing losses in a list for logging as we run several optimization steps
        losses = []
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            # Sample from the replay memory
            state, action, reward, next_state, done, info = memory.sample(self.batch_size, randomly = True)

            # Forward pass for Q networks
            td_target = self.calc_td_target(reward, done, next_state)
            Critic1_loss = self.criterion(self.Critic1.forward(state, action), td_target)
            Critic2_loss = self.criterion(self.Critic2.forward(state, action), td_target)
            Critic_loss = Critic1_loss + Critic2_loss
            
            #Backward step for Q networks
            self.optimizer_critic.zero_grad()
            Critic_loss.backward()
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_value_(parameters=list(self.Q1.parameters()) + list(self.Q2.parameters()), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
            self.optimizer_critic.step()
            
            # Get the target for Policy network
            q_1 = self.Critic1.forward(state, self.policy.forward(state))
            policy_loss = -torch.mean(q_1)
            
            # Backward step for Policy network
            if i % self.policy_delay == 0:
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_value_(parameters=self.policy.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                self.optimizer_actor.step()

            #Logging the losses, here only the critic loss
            losses.append(Critic_loss.item())
            
        #after each optimization, decay epsilon
        self.adjust_epsilon(episode_i)

        #after each optimization, update actor and critic target networks
        if episode_i % self.target_net_update_freq == 0 and self.use_target_net:
            self._copy_nets(soft_update = self.use_soft_updates)

        return losses

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.Critic1.eval()
            self.Critic2.eval()
            self.Actor.eval()
            
        else:
            self.Critic1.train()
            self.Critic2.train()
            self.Actor.train()
            
    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """
        checkpoint = {
            "actor": self.Actor.state_dict(),
            "critic1": self.Critic1.state_dict(),
            "critic2": self.Critic2.state_dict(),
            "iteration": iteration
        }

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")
        os.makedirs(directory, exist_ok = True)
        torch.save(checkpoint, file_path)
        print(f"Actor and Critic weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        try:
            checkpoint = torch.load(file_name, map_location=self.device)
            self.Actor.load_state_dict(checkpoint["actor"])
            self.Critic1.load_state_dict(checkpoint["critic1"])
            self.Critic2.load_state_dict(checkpoint["critic2"])
            print(f"Model loaded successfully from {file_name}")
        except FileNotFoundError:
            print(f"Error: File {file_name} not found.")
        except Exception as e:
            print(f"An error occurred while loading the model: {str(e)}")

    def _copy_nets(self, soft_update: bool) -> None:
        """
        Updates the target network with the weights of the original one
        If soft_update is True, we perform a soft update via \tau \cdot \theta + (1 - \tau) \cdot \theta'
        """
        assert self.use_target_net == True
        with torch.no_grad():
            if soft_update:
                for target_param, param in zip(self.targetQ1.parameters(), self.Q1.parameters()):
                    target_param.data.copy_(
                        param.data * self.tau + target_param.data * (1 - self.tau) if soft_update  # Soft update
                        else param.data  # Hard update
                    )
                for target_param, param in zip(self.targetQ2.parameters(), self.Q2.parameters()):
                    target_param.data.copy_(
                        param.data * self.tau + target_param.data * (1 - self.tau) if soft_update
                        else param.data
                    )
                for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                    target_param.data.copy_(
                        param.data * self.tau + target_param.data * (1 - self.tau) if soft_update
                        else param.data
                    )
            else:
                #Copying the weights of the Q and Policy networks to the target networks
                self.Critic1_target.load_state_dict(self.Critic1.state_dict())
                self.Critic2_target.load_state_dict(self.Critic2.state_dict())
                self.Actor_target.load_state_dict(self.Actor.state_dict())

    def import_checkpoint(self, checkpoint: dict) -> None:
        raise NotImplementedError

    def export_checkpoint(self) -> dict:
        raise NotImplementedError
