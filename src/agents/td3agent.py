import logging

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.constants import EXPONENTIAL, LINEAR



class QFunction(nn.Module):
    def __init__(self, state_size, hidden_sizes, action_size):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_size + action_size, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )
    
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
            nn.Linear(state_size, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, action_size),
            nn.Tanh()
        )
        
    def PolicyValue(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Policy values over all actions for the given state
        """
        self.network(state)


class TD3Agent(Agent):
    def __init__(self, state_shape: tuple[int, ...], action_size: int, options: dict, optim:dict, hyperparams:dict,
                 device: device):
        super().__init__()

        self.isEval = None

        self.state_shape = state_shape
        self.action_size = action_size

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

        # Options
        self.use_target_net = options["USE_TARGET_NET"]
        self.use_soft_updates = options["USE_SOFT_UPDATES"]
        self.use_gradient_clipping = options["USE_GRADIENT_CLIPPING"]
        self.epsilon_decay_strategy = options["EPSILON_DECAY_STRATEGY"]
        self.device: device = device
        self.use_clip_foreach = options["USE_CLIP_FOREACH"]
        self.USE_BF_16 = options["USE_BF16"]
        

        # Here we have 2 Q Networks
        self.Q1 = QFunction(state_size=state_shape,
                            hidden_sizes=128,
                            action_size=action_size.shape[0])
        self.Q2 = QFunction(state_size=state_shape,
                            hidden_sizes=128,
                            action_size=action_size.shape[0])
        
        self.targetQ1 = QFunction(state_size=state_shape,
                                    hidden_sizes=128,
                                    action_size=action_size.shape[0])
        self.targetQ2 = QFunction(state_size=state_shape,
                                    hidden_sizes=128,
                                    action_size=action_size.shape[0])
        
        self.targetQ1.to(self.device)
        self.targetQ2.to(self.device)
        self.targetQ1.eval() # Set the target network to evaluation mode
        self.update_target_net() # Copy the Q network
        
        
        #Define the actor network
        self.policy = PolicyFunction(state_size=state_shape,
                                    hidden_sizes=128,
                                    output_size=action_size.shape[0])
        self.policy_target = PolicyFunction(state_size=state_shape,
                                            hidden_sizes=128,
                                            output_size=action_size.shape[0])
        
        self.policy_target.to(self.device)
        self.policy_target.eval() # Set the target network to evaluation mode
        self.update_target_net() # Copy the Q network
        
        # Define the Optimizer
        self.optimizer = self.initOptim(optim=optim, parameters=self.Q.parameters())

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
        """
        Calculates the TD Target for clipping the Q Network outputs
        """
        if self.use_target_net:
            noise = torch.randn_like(action) * self.epsilon
            #calculate the next action via the target policy network
            next_action = torch.clamp(
                self.policy_target(next_state) + noise, 
                min=to_torch(self._action_space.low), #Minimum action value
                max=to_torch(self._action_space.high)) #Maximum action value
            
            q_prime1 = self.TargetQ1.QValue(next_state, next_action)
            q_prime2 = self.TargetQ2.QValue(next_state, next_action)
            #bootstrapping the minimum of the two Q networks
            return reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2) 
        
        else:
             #Update critic networks
            noise = torch.randn_like(action) * self.eposilon
            #calculate the next action via the policy network
            next_action = torch.clamp(
                self.policy(next_state) + noise, 
                min=to_torch(self._action_space.low), #Minimum action value
                max=to_torch(self._action_space.high)) #Maximum action value
            
            q_prime1 = self.Q1.QValue(next_state, next_action)
            q_prime2 = self.Q2.QValue(next_state, next_action)
            #bootstrapping the minimum of the two Q networks
            return reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2) 
    
    def forward_pass(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor, next_action: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
        # Step 01: Calculate the predicted q value
        predicted_q1_value = self.Q1.QValue(state, action)
        predicted_q2_value = self.Q2.QValue(state, action)

        # Step 02: Calculate the td target
        td_target = self.calc_td_target(reward, done, next_state)

        # Step 03: Finally, we calculate the loss
        loss_q1 = self.criterion(predicted_q1_value, td_target)
        loss_q2 = self.criterion(predicted_q2_value, td_target)
        return loss_q1, loss_q2
        
    
    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        """
        Optimize the Policy Network using the 2 Q Networks
        """
        assert self.isEval == False, "Make sure to put the model in training mode before calling the opt. routine"
        losses = []
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            self.optimizer.zero_grad()

            state, action, reward, next_state, done, info = memory.sample(self.batch_size)

            # Forward step for Q networks
            if self.USE_BF_16:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    q1_loss, q2_loss = self.forward_pass(state, action, reward, next_state, next_action, done)
            else:
                q1_loss, q2_loss = self.forward_pass(state, action, reward, next_state, next_action, done)
            
            #Update actor network every 4th iteration
            if self.train_iter % 4 == 0: #TO DO: NEW HYPERPARAMETER
                #Using Q1 network to optimize the actor
                policy_loss = -self.Q1.Q_value(state, self.policy(state)).mean()
                self.optimizer.zero_grad() #optimize the policy
                actor_loss.backward()
                self.optimizer.step()
                losses.append((q1_loss_value, q1_loss_value, actor_loss.item()))
            
            losses.append((q1_loss_value, q1_loss_value, None))
        
            # if we want to clip our gradients
            if self.use_gradient_clipping:
                # In-place gradient clipping
                torch.nn.utils.clip_grad_value_(parameters=self.Policy.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                torch.nn.utils.clip_grad_value_(parameters=self.Q1.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
                torch.nn.utils.clip_grad_value_(parameters=self.Q2.parameters(), clip_value=self.gradient_clipping_value, foreach=self.use_clip_foreach)
            self.optimizer.step()
            
            # Update the target net 
            self.updateTargetNet(soft_update=self.use_soft_updates)

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
            eps = 0
        
        action = self.policy.predict(state) + self.epsilon*np.random.normal(size=self.action_size.shape[0]) # action in -1 to 1 (+ noise)
        action = self.action_size.low + (action + 1.0) / 2.0 * (self.action_size.high - self.action_size.low) # resacling into the action space
        return action

    def setMode(self, eval=False) -> None:
        """
        Set the Agent in training or evaluation mode
        :param eval: If true = eval mode, False = training mode
        """
        self.isEval = eval
        if self.isEval:
            self.Q.eval()
        else:
            self.Q.train()
    
    def saveModel(self, fileName: str) -> None:
        """
        Saves the model parameters of the agent.
        """
        torch.save(self.Q.state_dict(), fileName)
        logging.info(f"Q network weights saved successfully!")
    
    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        self.Q.load_state_dict(torch.load(file_name))
        logging.info(f"Q network weights loaded successfully!") 
    
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
        



  