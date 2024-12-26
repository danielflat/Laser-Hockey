import logging
import os

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.constants import EXPONENTIAL, LINEAR
from src.util.directoryutil import get_path

to_torch = lambda x: torch.from_numpy(x.astype(np.float32))

class QFunction(nn.Module):
    def __init__(self, state_size, hidden_sizes, action_size):
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
        #Action and State concatenated as input into the q network
        concat_input = torch.cat((state, action), dim=1)
        return self.network(concat_input)

class PolicyFunction(nn.Module):
    def __init__(self, state_size, hidden_sizes, action_size):
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
        Calculates the Policy values over all actions for the given state
        """
        return self.network(state)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Calculates the detached Policy Value as a numpy array 
        """
        with torch.no_grad():
            return self.forward(x).numpy()


class TD3Agent(Agent):
    def __init__(self, observation_size, action_space, agent_settings: dict, td3_settings: dict, device: device):
        super().__init__(agent_settings = agent_settings, device = device)
        self.state_shape = observation_size
        self.action_space = action_space
        
        self.policy_delay = td3_settings["POLICY_DELAY"]


        # Here we have 2 Q Networks
        self.Q1 = QFunction(state_size=self.state_shape,
                            hidden_sizes=[128, 128],
                            action_size=self.action_space.shape[0])
        self.Q2 = QFunction(state_size=self.state_shape,
                            hidden_sizes=[128, 128],
                            action_size=self.action_space.shape[0])
        
        self.policy = PolicyFunction(state_size=self.state_shape,
                                    hidden_sizes=[128, 128, 64],
                                    action_size=self.action_space.shape[0])
        
        if self.use_target_net:
            self.targetQ1 = QFunction(state_size=self.state_shape,
                                        hidden_sizes=[128, 128],
                                        action_size=self.action_space.shape[0])
            self.targetQ2 = QFunction(state_size=self.state_shape,
                                        hidden_sizes=[128, 128],
                                        action_size=self.action_space.shape[0])
            self.policy_target = PolicyFunction(state_size=self.state_shape,
                                                hidden_sizes=[128, 128, 64],
                                                action_size=self.action_space.shape[0])
        
            self.targetQ1.to(self.device)
            self.targetQ2.to(self.device)
            self.policy_target.to(self.device)
            
            self.updateTargetNet(soft_update=self.use_soft_updates) # Copy the Networks
            
        self.optimizer_q1 = self.initOptim(optim=agent_settings["OPTIMIZER"], parameters=self.Q1.parameters()) #TO DO: Use different learning rates for Q and Policy
        self.optimizer_q2 = self.initOptim(optim=agent_settings["OPTIMIZER"], parameters=self.Q2.parameters())
        self.optimizer_policy = self.initOptim(optim=agent_settings["OPTIMIZER"], parameters=self.policy.parameters())
        
        #Define Loss function
        self.criterion = self.initLossFunction(loss_name = agent_settings["LOSS_FUNCTION"])
    
    
    def updateTargetNet(self, soft_update: bool) -> None:
        """
        Updates the target network with the weights of the original one
        """
        assert self.use_target_net == True, "You must use have 'self.use_target == True' to call 'updateTargetNet()'"

        for target_param, param in zip(self.targetQ1.parameters(), self.Q1.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau) if soft_update else param.data)
        for target_param, param in zip(self.targetQ2.parameters(), self.Q2.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau) if soft_update else param.data)
        for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau) if soft_update else param.data)
            

    def calc_td_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        
        #Calculates the TD Target for clipping the Q Network outputs
        if self.use_target_net:
            #Exploration noise
            noise = np.clip(
                np.random.normal(size=self.action_space.shape[0]) * self.epsilon,
                a_min=self.action_space.low,
                a_max=self.action_space.high)
            
            #Next action via the target policy network
            next_action = torch.clamp(
                self.policy_target.forward(next_state) + to_torch(noise), 
                min=to_torch(self.action_space.low), #Minimum action value
                max=to_torch(self.action_space.high)) #Maximum action value
            
            q_prime1 = self.targetQ1.forward(next_state, next_action)
            q_prime2 = self.targetQ2.forward(next_state, next_action)
            
            #Bootstrapping the minimum of the two Q networks
            return reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2) 
        
        else:
            noise = torch.clamp(
                np.random.normal(size=self.action_space.shape[0]) * self.epsilon,
                min=self.action_space.low,
                max=self.action_space.high
            ).to(self.device)
            
            next_action = torch.clamp(
                self.policy.forward(next_state) + to_torch(noise), 
                min=to_torch(self.action_space.low),
                max=to_torch(self.action_space.high))
            
            q_prime1 = self.Q1.forward(next_state, next_action)
            q_prime2 = self.Q2.forward(next_state, next_action)
            
            return reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2)
        

    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        """
        Compute forward and backward pass for the Q and Policy networks
        """
        assert self.isEval == False, "Make sure to put the model in training mode before calling the opt. routine"
        losses = []
        
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            
            state, action, reward, next_state, done, info = memory.sample(self.batch_size, randomly = True)

            #Forward pass for Q networks
            td_target = self.calc_td_target(reward, done, next_state)
            q1_loss = self.criterion(self.Q1.forward(state, action), td_target.detach())
            q2_loss = self.criterion(self.Q2.forward(state, action), td_target.detach())
            
            #Backward step for Q networks
            self.optimizer_q1.zero_grad()
            q1_loss.backward()
            self.optimizer_q1.step()
            
            self.optimizer_q2.zero_grad()
            q2_loss.backward()
            self.optimizer_q2.step()
            
            
            #Backward step for Policy network
            if i % self.policy_delay == 0:
                #Actor target, which is here the Q1 newtork output using the Policy Network
                q_1 = self.Q1.forward(state, self.policy.forward(state))
                self.optimizer_policy.zero_grad()
                policy_loss = -torch.mean(q_1)
                policy_loss.backward()
                self.optimizer_policy.step()
            
            
            #Clip gradients
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_value_(parameters=self.Policy.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                torch.nn.utils.clip_grad_value_(parameters=self.Q1.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                torch.nn.utils.clip_grad_value_(parameters=self.Q2.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)

            losses.append((q1_loss.item(), q2_loss.item(), policy_loss.item() if i % self.policy_delay == 0 else None))
            
        # after each optimization, we want to decay epsilon
        self.adjust_epsilon(episode_i)
        
        #after each optimization, update target network
        self.updateTargetNet(soft_update=self.use_soft_updates)

        return losses
    
    def act(self, state: torch.Tensor) -> torch.Tensor:
        """
        The Agent chooses an action.
        In Evaluation mode, we set the noise eps = 0
        In Training mode, we sample an action using actor network and exploration noise
        :param state: The state
        """
        # In evaluation mode, we always exploit
        if self.isEval:
            self.epsilon = 0
        
        action_deterministic = self.policy.predict(state)
        action = action_deterministic + self.epsilon * np.random.normal(size=self.action_space.shape[0]) # action in -1 to 1 (+ noise)
        
        action = self.action_space.low + (action + 1.0) / 2.0 * (self.action_space.high - self.action_space.low) # resacling into the action space
        return action

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.Q1.eval()
            self.Q2.eval()
            self.policy.eval()
        else:
            self.Q1.train()
            self.Q2.train()
            self.policy.train()
            
    
    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")

        # Ensure the directory exists
        os.makedirs(directory, exist_ok = True)
        torch.save((self.Q1.state_dict(), self.Q2.state_dict, self.policy.state_dict()), file_path)
        logging.info(f"Q  and Policy network weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        self.policy_net.load_state_dict(torch.load(file_name))
        logging.info(f"Q and Policy network weights loaded successfully!")
        

