import logging
import os
from typing import List

import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path
from src.util.noiseutil import initNoise
from src.util.icmutil import ICM


"""
Author: Andre Pfrommer
"""

####################################################################################################
# Critic and Actor Networks
####################################################################################################

class Critic(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, action_size: int, num_layers: int,
                 cross_q: bool, bn_momentum: float=0.9):
        super().__init__()
        
        #If cross_q method, we use 1D Batchnorm after every Linear layer
        if cross_q:
            layers = []
            
            layers.append(nn.BatchNorm1d(state_size + action_size, momentum=bn_momentum)) 
            layers.append(nn.Linear(state_size + action_size, hidden_size))  
            layers.append(nn.ReLU()) 
            
            #We can add more hidden layers as we do batchnorm now
            for _ in range(num_layers - 1):  
                layers.append(nn.BatchNorm1d(hidden_size, momentum=bn_momentum)) 
                layers.append(nn.Linear(hidden_size, hidden_size))  
                layers.append(nn.ReLU())  
                
            #Output layer
            layers.append(nn.Linear(hidden_size, 1))
            
            self.network = nn.Sequential(*layers)
            
        #Otherwise, the standard 2 layer Q network
        else:
            self.network = nn.Sequential(
                nn.Linear(state_size + action_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Q value for the given state and action
        """
        # Action and State concatenated as input into the q network
        
        concat_input = torch.hstack((state, action))
        return self.network(concat_input)

class Actor(nn.Module):
    def __init__(self, state_size: int, hidden_size: int, action_size: int):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
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
        action_size = action_space.shape[0]

        self.policy_delay = td3_settings["POLICY_DELAY"]
        self.noise_clip = td3_settings["NOISE_CLIP"]
        self.hidden_size = td3_settings["HIDDEN_DIM"]
        self.num_layers = td3_settings["NUM_LAYERS"]
        self.bn_momentum = td3_settings["BATCHNORM_MOMENTUM"]
        #Whether or not to use Batchnorm
        self.cross_q = True
        if self.use_target_net:
            self.cross_q = False
        
        #Initialize the noise 
        self.noise = initNoise(action_shape = (action_size,), noise_settings = td3_settings["NOISE"],
                               device = self.device)
        self.noise_factor = td3_settings["NOISE"]["NOISE_FACTOR"]

        # Here we have 2 Q Networks
        self.Critic1 = Critic(state_size=state_size,
                            hidden_size=self.hidden_size,
                            action_size=action_size,
                            num_layers=self.num_layers,
                            cross_q=self.cross_q,
                            bn_momentum=self.bn_momentum).to(device)
        self.Critic2 = Critic(state_size=state_size,
                            hidden_size=self.hidden_size,
                            action_size=action_size,
                            num_layers=self.num_layers,
                            cross_q=self.cross_q,
                            bn_momentum=self.bn_momentum).to(device)
        self.Actor = Actor(state_size=state_size,
                        hidden_size=self.hidden_size,
                        action_size=action_size).to(device)
        if self.use_target_net:
            self.Critic1_target = Critic(state_size=state_size,
                                        hidden_size=self.hidden_size,
                                        action_size=action_size,
                                        num_layers=self.num_layers,
                                        cross_q=self.cross_q).to(device)
            self.Critic2_target = Critic(state_size=state_size,
                                        hidden_size=self.hidden_size,
                                        action_size=action_size,
                                        num_layers=self.num_layers,
                                        cross_q=self.cross_q).to(device)
            self.Actor_target = Actor(state_size=state_size,
                                    hidden_size=self.hidden_size,
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
        
        #Initialize intrinsic curiosity module 
        self.icm = ICM(state_size, action_size, discrete = False)

    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        The Agent chooses an action.
        In Evaluation mode, we set the noise eps = 0
        In Training mode, we sample an action using actor network and exploration noise
        :param state: The state
        """
        
        if self.isEval:
            self.noise_factor = 0
            
        with torch.no_grad():
            #Exploration noise
            noise = torch.from_numpy(self.noise.sample() * self.noise_factor)
            
            #Forward pass for the Actor network and rescaling
            self.Actor.eval() #eval mode for batchnorm to compute running statistics
            det_action = self.Actor(state) * self.action_scale + self.action_bias
            action = det_action + noise
            action = torch.clamp(
                det_action + noise, 
                min=self.action_low, #Minimum action value
                max=self.action_high) #Maximum action value
            
        return action.numpy()

    def critic_forward(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the TD Target and the loss for the Q networks
        1. Sample the next action from the target Actor network using gaussian exploration noise  
        2. Calculate the Q values for the next state and next action using both Q networks
            NOTE: When using Cross-Q, we forward both the state and next state through the Q networks
        3. Calculate the TD Target by bootstrapping the minimum of the two Q networks
        4. Calculate the loss for both Q networks using the TD Target
        
        """

        # Exploration noise
        noise = torch.clamp(
            torch.from_numpy(self.noise.sample() * self.noise_factor),
            min = -self.noise_clip,
            max = self.noise_clip)
        
        if self.cross_q:
            # 1. Next action via the Actor network
            next_action = self.Actor.forward(next_state) * self.action_scale + self.action_bias
            #next_action = torch.clamp(
            #    next_action + noise,
            #    min = self.action_low,  # Minimum action value
            #    max = self.action_high).float()  # Maximum action value
            
            # Concat both states and actions for joint forward pass
            cat_states = torch.cat([state, next_state], 0) # (batch_size x 2, state_size)
            cat_actions = torch.cat([action, next_action], 0) # (batch_size x 2, action_size)

            #2. Forward pass for Q networks
            self.Critic1.train() # switch to training - to update BN statistics if any
            self.Critic2.train()
            q_full1 = self.Critic1(cat_states, cat_actions) # (batch_size x 2, 1)
            q_full2 = self.Critic2(cat_states, cat_actions) # (batch_size x 2, 1)
            self.Critic1.eval() # switch back to eval mode
            self.Critic2.eval()
            
            # Separating Q outputs
            q1, q_prime1 = torch.chunk(q_full1, chunks=2, dim=0)
            q2, q_prime2 = torch.chunk(q_full2, chunks=2, dim=0)
            
            #3. Bootstrapping the minimum of the two Q networks
            td_target = (reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2)).detach()
                
            #4. Find the MSE loss for both Q networks
            Critic1_loss = self.criterion(q1, td_target)
            Critic2_loss = self.criterion(q2, td_target)
            Critic_loss = Critic1_loss + Critic2_loss
        
        else:
            # 1. Next action via the target Actor network
            next_action = self.Actor_target(next_state) * self.action_scale + self.action_bias
            #next_action = torch.clamp(
            #    next_action + noise,
            #    min = self.action_low,          # Minimum action value
            #    max = self.action_high).float()  # Maximum action value
            
            # 2. Forward pass for both Q networks
            q_prime1 = self.Critic1_target(next_state, next_action)
            q_prime2 = self.Critic2_target(next_state, next_action)
            
            # 3. Bootstrapping the minimum of the two Q networks
            td_target = reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2).detach()
                
            # 4. Find the MSE loss for both Q networks
            Critic1_loss = self.criterion(self.Critic1(state, action), td_target)
            Critic2_loss = self.criterion(self.Critic2(state, action), td_target)
            Critic_loss = Critic1_loss + Critic2_loss
            
        return Critic_loss

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
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
            
            # Train the curiosity module 
            # self.icm.train(state, next_state, action)
            
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
                torch.nn.utils.clip_grad_value_(
                    parameters = list(self.Critic1.parameters()) + list(self.Critic2.parameters()),
                    clip_value = self.gradient_clipping_value, foreach = self.use_clip_foreach)
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
                    torch.nn.utils.clip_grad_value_(parameters = self.Actor.parameters(),
                                                    clip_value = self.gradient_clipping_value,
                                                    foreach = self.use_clip_foreach)
                self.optimizer_actor.step()

            #Logging the losses, here only the critic loss
            losses.append([Critic_loss.item(), policy_loss.item()])

        #after each optimization, update actor and critic target networks
        if episode_i % self.target_net_update_freq == 0 and self.use_target_net:
            self._copy_nets()
        
        sum_up_stats = {
            "Critic_Loss": sum([l[0] for l in losses]) / len(losses),
            "Policy_Loss": sum([l[1] for l in losses]) / len(losses)
        }
        
        return sum_up_stats

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

    def reset(self):
        raise NotImplementedError
