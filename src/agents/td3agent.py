import logging

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.constants import EXPONENTIAL, LINEAR


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
    
    def QValue(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        
    def PolicyValue(self, state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Policy values over all actions for the given state
        """
        return self.network(state)
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return self.PolicyValue(x).numpy()

class OUNoise():
        def __init__(self, shape, theta: float = 0.15, dt: float = 1e-2):
            self._shape = shape
            self._theta = theta
            self._dt = dt
            self.noise_prev = np.zeros(self._shape)
            self.reset()

        def __call__(self) -> np.ndarray:
            noise = (
                self.noise_prev
                + self._theta * ( - self.noise_prev) * self._dt
                + np.sqrt(self._dt) * np.random.normal(size=self._shape)
            )
            self.noise_prev = noise
            return noise
        
        def reset(self) -> None:
            self.noise_prev = np.zeros(self._shape)


class TD3Agent(Agent):
    def __init__(self, state_shape, action_space, options: dict, optim:dict, hyperparams:dict,
                 device: device):
        super().__init__()

        self.isEval = None

        self.state_shape = state_shape
        self.action_space = action_space

        # Hyperparams
        self.opt_iter = hyperparams["OPT_ITER"]
        self.batch_size = hyperparams["BATCH_SIZE"]
        self.discount = hyperparams["DISCOUNT"]
        self.epsilon = hyperparams["EPSILON"]
        self.epsilon_start = hyperparams["EPSILON"]
        self.epsilon_min = hyperparams["EPSILON_MIN"]
        self.epsilon_decay = hyperparams["EPSILON_DECAY"]
        self.gradient_clipping_value = hyperparams["GRADIENT_CLIPPING_VALUE"]
        self.target_net_update_freq = hyperparams["TARGET_NET_UPDATE_FREQ"]
        self.tau = hyperparams["TAU"]
        self.policy_delay = 1

        # Options
        self.use_target_net = options["USE_TARGET_NET"]
        self.use_soft_updates = options["USE_SOFT_UPDATES"]
        self.use_gradient_clipping = options["USE_GRADIENT_CLIPPING"]
        self.epsilon_decay_strategy = options["EPSILON_DECAY_STRATEGY"]
        self.device: device = device
        self.use_clip_foreach = options["USE_CLIP_FOREACH"]
        self.USE_BF_16 = options["USE_BF16"]
        
        self.action_noise = OUNoise((self.action_space.shape[0],)) 

        # Here we have 2 Q Networks
        self.Q1 = QFunction(state_size=state_shape,
                            hidden_sizes=[128, 128],
                            action_size=action_space.shape[0])
        self.Q2 = QFunction(state_size=state_shape,
                            hidden_sizes=[128, 128],
                            action_size=action_space.shape[0])
        
        self.policy = PolicyFunction(state_size=state_shape,
                                    hidden_sizes=[128, 128, 64],
                                    action_size=action_space.shape[0])
        
        if self.use_target_net:
            self.targetQ1 = QFunction(state_size=state_shape,
                                        hidden_sizes=[128, 128],
                                        action_size=action_space.shape[0])
            self.targetQ2 = QFunction(state_size=state_shape,
                                        hidden_sizes=[128, 128],
                                        action_size=action_space.shape[0])
            self.policy_target = PolicyFunction(state_size=state_shape,
                                                hidden_sizes=[128, 128, 64],
                                                action_size=action_space.shape[0])
        
            self.targetQ1.to(self.device)
            self.targetQ2.to(self.device)
            self.policy_target.to(self.device)
            
            self.updateTargetNet(soft_update=self.use_soft_updates) # Copy the Networks
        
        
        self.train_iter = 0
        self.Policyoptimizer = self.initOptim(optim=optim, parameters=self.policy.parameters())
        self.Q1optimizer = self.initOptim(optim=optim, parameters=self.Q1.parameters())
        self.Q2optimizer = self.initOptim(optim=optim, parameters=self.Q2.parameters())
        
        # Define Loss function
        self.criterion = self.initLossFunction(loss_name = options["LOSS_FUNCTION"])
    
    
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
            noise = self.action_noise() * self.epsilon
            #calculate the next action via the target policy network
            next_action = torch.clamp(
                self.policy_target.PolicyValue(next_state) + to_torch(noise), 
                min=to_torch(self.action_space.low), #Minimum action value
                max=to_torch(self.action_space.high)) #Maximum action value
            
            q_prime1 = self.targetQ1.QValue(next_state, next_action)
            q_prime2 = self.targetQ2.QValue(next_state, next_action)
            
            #bootstrapping the minimum of the two Q networks
            return reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2) 
        
        else:
            noise = self.action_noise() * self.epsilon
            #calculate the next action via the policy network
            next_action = torch.clamp(
                self.policy.PolicyValue(next_state) + to_torch(noise), 
                min=to_torch(self.action_space.low), #Minimum action value
                max=to_torch(self.action_space.high)) #Maximum action value
            
            q_prime1 = self.Q1.QValue(next_state, next_action)
            q_prime2 = self.Q2.QValue(next_state, next_action)
            
            #bootstrapping the minimum of the two Q networks
            return reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2)
    
    def forward_pass(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        # Step 01: Calculate the predicted q value given the state and the action from the buffer
        predicted_q1_value = self.Q1.QValue(state, action)
        predicted_q2_value = self.Q2.QValue(state, action)

        # Step 02: Calculate the td target
        td_target = self.calc_td_target(reward, done, next_state)
        
        # Step 04: Calculate the actor target, which is here the Q1 newtork output using the Policy Network
        q_1 = self.Q1.QValue(state, self.policy.PolicyValue(state))
        
        # Step 03: Finally, we calculate all the losses
        loss_q1 = self.criterion(predicted_q1_value, td_target.detach())
        loss_q2 = self.criterion(predicted_q2_value, td_target.detach())
        loss_policy = -torch.mean(q_1)
        
        return loss_q1, loss_q2, loss_policy

    def reset(self) -> None:
        self.noise_prev = np.zeros(self._shape)
        
    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        assert self.isEval == False, "Make sure to put the model in training mode before calling the opt. routine"
        losses = []
        self.train_iter+=1
        
        if self.use_target_net and self.train_iter % self.target_net_update_freq == 0:
                self.updateTargetNet(soft_update=self.use_soft_updates)
        
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            
            state, action, reward, next_state, done, info = memory.sample(self.batch_size)

            # Forward step for Q and Policy networks
            if self.USE_BF_16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    q1_loss, q2_loss, policy_loss = self.forward_pass(state, action, reward, next_state, done)
            else:
                q1_loss, q2_loss, policy_loss = self.forward_pass(state, action, reward, next_state, done)
            
            #Compute backward passes for all networks
            self.Policyoptimizer.zero_grad()
            policy_loss.backward()
            self.Policyoptimizer.step()
            
            self.Q1optimizer.zero_grad()
            q1_loss.backward()
            self.Q1optimizer.step()
            
            self.Q2optimizer.zero_grad()
            q2_loss.backward()
            self.Q2optimizer.step()
            
            losses.append((q1_loss.item(), q1_loss.item(), policy_loss.item()))
        
            # if we want to clip our gradients
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_value_(parameters=self.Policy.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                torch.nn.utils.clip_grad_value_(parameters=self.Q1.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                torch.nn.utils.clip_grad_value_(parameters=self.Q2.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)

        # after each optimization, we want to decay epsilon
        self.adjust_epsilon(episode_i)

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
        action = action_deterministic + self.epsilon*self.action_noise()# action in -1 to 1 (+ noise)
        action = action_deterministic + self.epsilon*np.random.normal(size=self.action_space.shape[0]) # action in -1 to 1 (+ noise)
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
            
    
    def saveModel(self, fileName: str) -> None:
        """
        Saves the model parameters of the agent.
        """
        
        torch.save(self.Q1.state_dict(), 
                   self.Q2.state_dict(), 
                   self.policy.state_dict(), fileName)
        logging.info(f"Q and Policy network weights saved successfully!")
    
    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        self.Q1.load_state_dict(torch.load(file_name))
        self.Q2.load_state_dict(torch.load(file_name))
        self.policy.load_state_dict(torch.load(file_name))
        logging.info(f"Q and Policy network weights loaded successfully!") 
    
    def adjust_epsilon(self, episode_i: int) -> None:
        """
        Here, we decay epsilon w.r.t. the number of run episodes
        :param episode_i: The ith episode
        """
        if self.epsilon_decay_strategy == LINEAR:
            new_candidate = np.maximum(self.epsilon_min, self.epsilon_start - (episode_i/self.epsilon_decay) * (self.epsilon_start - self.epsilon_min))
            self.epsilon = new_candidate.item()
        elif self.epsilon_decay_strategy == EXPONENTIAL:
            new_candidate = np.maximum(self.epsilon_min, self.epsilon_start * (self.epsilon_decay ** episode_i))
            self.epsilon = new_candidate.item()
        else:
            raise NotImplementedError(f"The epsilon decay strategy '{self.epsilon_decay_strategy}' is not supported! Please choose another one!")
        

