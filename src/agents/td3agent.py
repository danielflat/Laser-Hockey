import os
import logging 

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path
from src.util.noiseutil import initNoise
"""
Author: Andre Pfrommer

TODOS:
- ADD OU NOISE/WHITE NOISE/PINK NOISE
- ADD BATCH NORMALIZATION
"""
    
####################################################################################################
# Critic and Actor Networks
####################################################################################################

class Critic(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: list[int], action_size: int):
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
    def __init__(self, state_size: int, hidden_sizes: list[int], action_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_size),
            nn.Tanh()
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Actor values over all actions for the given state
        """
        action = self.network(state)
        return action

####################################################################################################
# TD3 Agent
####################################################################################################

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
                                    hidden_sizes=[128, 128],
                                    action_size=action_size).to(device)
        if self.use_target_net:
            self.Critic1_target = Critic(state_size=state_size,
                                        hidden_sizes=[128, 128],
                                        action_size=action_size).to(device)
            self.Critic2_target = Critic(state_size=state_size,
                                        hidden_sizes=[128, 128],
                                        action_size=action_size).to(device)
            self.Actor_target = Actor(state_size=state_size,
                                                hidden_sizes=[128, 128],
                                                action_size=action_size).to(device)
            # Set the target nets in eval
            self.Critic1_target.eval()
            self.Critic2_target.eval()
            self.Actor_target.eval()

            # Copying the weights of the Q and Policy networks to the target networks
            self._copy_nets()

        # Initializing the optimizers
        self.optimizer_actor = self.initOptim(optim = td3_settings["ACTOR"]["OPTIMIZER"],
                                           parameters = self.Actor.parameters())
        self.optimizer_critic = self.initOptim(optim = td3_settings["CRITIC"]["OPTIMIZER"],
                                           parameters = list(self.Critic1.parameters()) + list(self.Critic2.parameters()))

        #Define Loss function
        self.criterion = self.initLossFunction(loss_name = td3_settings["CRITIC"]["LOSS_FUNCTION"])

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
            #Exploration noise
            noise = torch.clamp(
                torch.randn(self.action_space.shape[0], device=self.device) * self.epsilon,
                min=-self.noise_clip,
                max=self.noise_clip)
            
            #Forward pass for the Actor network and rescaling
            det_action = self.Actor.forward(state) * self.action_scale + self.action_bias
            action = det_action 
            action = torch.clamp(
                det_action + noise, 
                min=self.action_low, #Minimum action value
                max=self.action_high) #Maximum action value
            
        action = action.cpu().numpy()
        return action

    def critic_forward(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the TD Target and the loss for the Q networks
        1. Sample the next action from the target Actor network using gaussian exploration noise
        2. Calculate the Q values for the next state and next action using both target Q networks
        3. Calculate the TD Target by bootstrapping the minimum of the two Q networks
        4. Calculate the loss for both Q networks using the TD Target
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
                    min=self.action_low, #Minimum action value
                    max=self.action_high) #Maximum action value

                #2. Forward pass for both Q networks
                q_prime1 = self.Critic1.forward(next_state, next_action)
                q_prime2 = self.Critic2.forward(next_state, next_action)
                
            #3. Bootstrapping the minimum of the two Q networks
            td_target = reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2)
            
        #4. Logging the Q values for the next state and next action
        Critic1_loss = self.criterion(self.Critic1.forward(state, action), td_target)
        Critic2_loss = self.criterion(self.Critic2.forward(state, action), td_target)
        Critic_loss = Critic1_loss + Critic2_loss
            
        return Critic_loss

    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        """
        Compute forward and backward pass for the Q and Policy networks
        """
        assert self.isEval == False
        # Storing losses in a list for logging as we run several optimization steps
        losses = []
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            #Sample from the replay memory
            state, action, reward, next_state, done, info = memory.sample(self.batch_size, randomly=True)
            state = state.float()
            action = action.float()
            reward = reward.float()
            next_state = next_state.float()
            #Forward pass for Q networks
            if self.USE_BF_16:
                with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                        Critic_loss = self.critic_forward(state, action, reward, done, next_state)
            else:
                Critic_loss = self.critic_forward(state, action, reward, done, next_state)
            
            #Backward step for Q networks
            self.optimizer_critic.zero_grad()
            Critic_loss.backward()
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_value_(parameters=list(self.Q1.parameters()) + list(self.Q2.parameters()), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
            self.optimizer_critic.step()
            
            #Get the target for Policy network
            if self.USE_BF_16:
                with torch.autocast(device_type = self.device.type, dtype = torch.bfloat16):
                    q_1 = self.Critic1.forward(state, self.Actor.forward(state))
                    policy_loss = -torch.mean(q_1)
            else:
                q_1 = self.Critic1.forward(state, self.Actor.forward(state))
                policy_loss = -torch.mean(q_1)
            
            # Backward step for Policy network
            if i % self.policy_delay == 0:
                self.optimizer_actor.zero_grad()
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
            self._copy_nets()
        
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
        checkpoint = self.export_checkpoint()

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok = True)

        torch.save(checkpoint, file_path)
        logging.info(f"Iteration: {iteration} TD3 checkpoint saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        try:
            checkpoint = torch.load(file_name, map_location=self.device)
            self.import_checkpoint(checkpoint)
            logging.info(f"Model loaded successfully from {file_name}")
        except FileNotFoundError:
            logging.error(f"Error: File {file_name} not found.")
        except Exception as e:
            logging.error(f"An error occurred while loading the model: {str(e)}")

    def _copy_nets(self) -> None:
        assert self.use_target_net == True
        # Step 01: Copy the actor net
        self.updateTargetNet(soft_update = self.use_soft_updates, source = self.Actor,
                             target = self.Actor_target)

        # Step 02: Copy the critic nets
        self.updateTargetNet(soft_update = self.use_soft_updates, source = self.Critic1,
                             target = self.Critic1_target)
        self.updateTargetNet(soft_update = self.use_soft_updates, source = self.Critic2,
                             target = self.Critic2_target)

    def import_checkpoint(self, checkpoint: dict) -> None:
        self.Actor.load_state_dict(checkpoint["Actor"])
        self.Critic1.load_state_dict(checkpoint["Critic1"])
        self.Critic2.load_state_dict(checkpoint["Critic2"])

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "Actor": self.Actor.state_dict(),
            "Critic1": self.Critic1.state_dict(),
            "Critic2": self.Critic2.state_dict(),
        }
        return checkpoint
