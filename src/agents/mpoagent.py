import os
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torch.nn.utils import clip_grad_norm_
from torch.distributions import MultivariateNormal, Categorical

from src.replaymemory import ReplayMemory
from src.util.constants import ADAM, MSELOSS, LINEAR, EXPONENTIAL
from src.agent import Agent 
from src.util.directoryutil import get_path

################################################################################
# Helper Networks
################################################################################

class Actor(nn.Module):
    """
    Policy network for discrete action space that outputs a probability distribution over da actions.
    :param state_dim:
        (B, ds), the dimension of the state space
    :param action_size:
        (B, da), the number of possible actions
    :param hidden_dim:
        (int) hidden layer dimension
    """
    def __init__(self, state_dim: int, action_size, hidden_dim: int = 256, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        
        self.ds = state_dim
        self.da = action_size 
        
        #Feedforward network
        self.net = nn.Sequential(
            nn.Linear(self.ds, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.da),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param:
            (B, ds) the state tensor
        :return: 
            (B, da) probability distribution over actions
        """
        return self.net(state)
        
    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param:
            (B, ds) the state tensor
        :return:
            (B,) the greedy action
        """
        with torch.no_grad():
            action_probs = self.net(state)
            greedyAction = torch.argmax(action_probs, dim = -1)
            return greedyAction
        
class Critic(nn.Module):
    """
    Q-value network for discrete action space.
    :param state_dim:
        (B, ds), the dimension of the state space
    :param action_size:
        (B, da), the number of possible actions
    :param hidden_dim:
        (int) hidden layer dimension
    """
    def __init__(self, state_dim: int, action_size: int, hidden_dim: int):
        super(Critic, self).__init__()
        
        self.ds = state_dim
        self.da = action_size
        
        #Feedforward network
        self.net = nn.Sequential(
            nn.Linear(self.ds, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, self.da),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param state: 
            (B, ds) the state tensor
        :param action: 
            (B, da) the action tensor
        :return: 
            (B,) Q-value
        """
            
        return self.net(x)
    
    def QValue(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Q value for the given state and action
        """
        all_q_values = self.forward(state)
        q_value = all_q_values.gather(dim=1, index=action)
        return q_value
    
    
################################################################################
#  Maximum a Posteriori Policy Optimisation (MPO) Agent
################################################################################

class MPOAgent(Agent):
    """
    The (discrete) MPO Agent. 
    This agent is largely based on the paper "Relative Entropy Regularized Policy Iteration" by Haarnoja et al. (2018)
    https://arxiv.org/pdf/1812.02256.pdf 
    
    Parameters specific to the MPO agent:
    
    :param hidden_dim: 
        (int) hidden layer dimension
    :param sample_action_num: 
        (int) number of actions to sample per state (N), irrelevant for discrete action spaces
    :param dual_constraint ε_dual:
        (float) hard constraint of the dual formulation in the E-step
    :param kl_constraint ε_kl:
        (float) hard constraint for the KL divergence in the M-step
    :param mstep_iteration_num:
        (int) the number of iterations of the M-step
    :param alpha_scale:
        (float) scaling factor of the lagrangian multiplier in the M-step
    
    Note: in the continuous case (not implemented yet), we get even more new hyperparameters as we need to fit a multivariate gaussian policy
    """
    def __init__(self, agent_settings, device, state_dim, action_size, mpo_settings):
        super().__init__(agent_settings, device)
        #TODO: add support for continous action spaces
        
        self.device = device
        self.ds = state_dim  # State space dimensions
        self.da = action_size # Nr of possible actions
        
        self.hidden_dim = mpo_settings.get("HIDDEN_DIM", 128) 
        self.sample_action_num = mpo_settings.get("SAMPLE_ACTION_NUM", 10) 
        self.ε_dual = mpo_settings.get("DUAL_CONSTAINT", 0.1) 
        self.ε_kl = mpo_settings.get("KL_CONSTRAINT", 0.0001) 
        self.mstep_iteration_num = mpo_settings.get("MSTEP_ITER", 5)
        self.α_scale = mpo_settings.get("ALPHA_SCALE", 1.0)
        
        #initialize variables to optimize
        self.η = np.random.rand()
        self.η_kl = 0.0

        #Set up the actor and critic networks
        self.actor = Actor(state_dim, action_size, self.hidden_dim).to(device)
        self.critic = Critic(state_dim, action_size, self.hidden_dim).to(device)
        self.target_actor = Actor(state_dim, action_size, self.hidden_dim).to(device)
        self.target_critic = Critic(state_dim, action_size, self.hidden_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        #Set up the optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 
                                                lr = 0.00001, eps = 0.000001,
                                                )
        self.critic_optimizer =  torch.optim.Adam(self.critic.parameters(), 
                                                  lr = 0.0001, eps = 0.000001)
        
        #Define the loss function for the crtic
        self.norm_loss_q = nn.SmoothL1Loss()
    
    def categorical_kl(self, p1, p2):
        """
        calculates KL between two Categorical distributions
        :param p1: 
            (B, D) the first distribution
        :param p2: 
            (B, D) the second distribution
        """
        #avoid zero division
        p1 = torch.clamp_min(p1, 0.0001)  
        p2 = torch.clamp_min(p2, 0.0001)  
        return torch.mean((p1 * torch.log(p1 / p2)).sum(dim=-1))

    def act(self, state: torch.Tensor) -> int:
        """
        Selects an action based on the current policy and evaluation mode. 
        :param state:
            (B, ds) the current state
        """
        with torch.no_grad():
            
            if self.isEval:
                # if you are in eval mode, get the greedy Action
                greedy_action = self.actor.greedyAction(state)
                return greedy_action.item()
            else:
                # In training mode, use epsilon greedy action sampling
                rdn = np.random.random()
                if rdn <= self.epsilon:
                    # Exploration. Take a random action
                    return np.random.randint(low = 0, high = self.da)
                else:
                    # Exploitation. Take the actions w.r.t. the old policy
                    π_p = self.target_actor(state)
                    π = Categorical(probs=π_p) 
                    action = π.sample()
                    
                return action.item()
        
    def critic_update(self, states, actions, dones, next_states, rewards):
        """ 
        Compute the temporal difference loss for the critic network
        
        :param states: 
            (B, ds)
        :param actions: 
            (B, da) 
        :param dones: 
            (B,)
        :param next_states: 
            (B, ds)
        :param rewards: 
            (B,)
        :return:
            (float) the loss for the critic
        """    
        with torch.no_grad():
            
            # Compute policy probabilities using the actor network
            π = self.target_actor(next_states)  # (B, da)
            π_dist = Categorical(probs=π)
            π_p = π_dist.probs  # (B, da)
            
            # Compute expected Q-values under policy
            next_q_values = self.target_critic(next_states)  # (B, da)
            expected_q = (π_p * next_q_values).sum(dim=1, keepdim=True)  # (B, 1)
            # Calculate the td target
            td_target = rewards + (1 - dones) * self.discount * expected_q
        
        # Calculate the loss using predicted q value and td target
        predicted_q_value = self.critic.QValue(states, actions)
        critic_loss = self.norm_loss_q(predicted_q_value, td_target)

        return critic_loss
    
    def find_qij_dist(self, q_target: torch.Tensor, π_target: torch.Tensor) -> torch.Tensor:
        """
        Find the action weights qij by applying two value constraints in a nonparametric way:
        1. Keep qij close to the target q values. The distribution of qij can computed in closed form by minimizing the dual function
        2. Apply softmax over all actions to normalize q values
        """
        
        q_target_np = q_target.numpy()  # (K, da)
        π_target_np = π_target.numpy()  # (K, da)
        
        def dual(η):
            """
            Dual function for MPO with numerical stabilization.
            
            dual function of the non-parametric variational
            g(η) = η*ε + η*mean(log(x)) where
            x = mean(exp(Q(s, a)/η))
            This equation is correspond to last equation of the [2] p.15
            For numerical stabilization, this can be modified to
            Qj = max(Q(s, a), along=a)
            
            I got this function from p.4 of the paper and the Github
            https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
            """
            # Max Q-value per state
            max_q = np.max(q_target_np, axis=1) 
            # Stabilized exponential term
            x = np.sum(π_target_np * np.exp((q_target_np - max_q[:, None]) / η), axis=1)
            # Avoid log(0) by clamping x
            x = np.clip(x, a_min=1e-8, a_max=None)
            # Dual function value
            g = η * self.ε_dual + np.mean(max_q) + η * np.mean(np.log(x))
            
            return g

        bounds = [(1e-6, None)]
        #Minimize the dual function using the scipy minimize function (1st constraint)
        res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
        #Update the dual variable η
        self.η = res.x[0]
        # Compute action weights (new q values) using dual variable η (see 3rd eq on page 4)
        # Apply softmax over all actions to normailze q values (2nd constriant)
        qij = torch.softmax(q_target / self.η, dim=1) # (K, da)
        return qij

    def optimize(self, memory: ReplayMemory, episode_i: int) -> list[float]:
        """
        Optimize actor and critic networks based on experience replay.
            1. Policy Evaluation: Update Critic via TD Learning
            2. E-Step of Policy Improvement: Sample from Critic and update dual variable η to find action weights qij
            3. M-Step of Policy Improvement: Update Actor via gradient ascent on the Lagrangian function
        
        :param memory:
            (ReplayMemory) the replay memory
        :param episode_i:
            (int) the current episode number
        :return:    
            (list[float]) the losses of the actor and critic networks
        """
        assert self.isEval == False
        #Sampling the whole trajectory will give us better optimization since we maximize the nr of sampled q values in the E step (works esp good on Cartpole env)
        #Feel free to change this to a fixed batch size
        #self.batch_size = len(memory) 
        losses = []
        
        # Number of actions to sample per state, irrelevant for discrete action spacessince we select all da actions per state
        N = self.sample_action_num  
        # Nr of sampled states, here the Batch size
        K = self.batch_size 

        # Identity matrix of shape (da, da)
        self.A_eye = torch.eye(self.da).to(self.device)  # (da, da)
        
        for _ in range(self.opt_iter):
            
            # Sample from replay buffer, dimensions (K, ds), (K, da), (K,), (K, ds) 
            states, actions, rewards, next_states, dones, _ = memory.sample(batch_size = self.batch_size, randomly = True)
            
            # 1: Policy Evaluation: Update Critic (Q-function)
            loss_critic = self.critic_update(states, actions, dones, next_states, rewards)
            # Backward pass in the critic network
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clipping_value)
            self.critic_optimizer.step()

            # 2: E-Step Finding action weights via sampling and non-parametric optimization (4.1 in paper)
            with torch.no_grad():
                
                #Get the q values for the K sampled states and *all* actions
                q_target = self.target_critic(states)
                
                #Get the policy output (probability for all states) for the K states
                π_target = self.target_actor(states) # (K, da)
                π_target_dist = Categorical(probs=π_target)
                π_target_p = π_target_dist.probs  # (K, da)
                
                #Minimize the dual function to find the action weights qij
                qij = self.find_qij_dist(q_target, π_target_p) # (K, da)

            # 3. M step. Policy Improvement (4.2 in paper)
            #Fitting an improved policy using the sampled q-values via gradient optimization on the Policy and the lagrangian function 
            for _ in range(self.mstep_iteration_num):
                
                #Creates action tensor of shape (da, K), where each of the K rows contains repeated indices ranging from 0 to self.da - 1
                actions = torch.arange(self.da).unsqueeze(1).expand(self.da, K).to(self.device)  # (da, K)
                
                #action output of the current parametric policy
                π_p = self.actor.forward(states)  # (K, da)
                π = Categorical(probs=π_p)  # (K,)
                π_log = π.log_prob(actions).T  # (K, da)
                
                #Kl divergence between the current and target policy, used to regularize the MLE loss above 
                kl = self.categorical_kl(p1=π_p, p2=π_target_p).detach()
                
                # Minimize lagrange multiplier α by gradient descent (inner optimiation loop)
                # this equation is derived from last eq of p.5 in the paper,
                # just differentiate with respect to α
                # and update α so that the equation is to be minimized.
                self.η_kl -= self.α_scale * (self.ε_kl - kl).item()

                #Clip the lagrange multipliers to positive values
                if self.η_kl < 0:
                    self.η_kl = 0.0
                
                # Last eq of p.4 in the paper
                # MLE loss between the parametric policy and the sample based distribution qij without Kl regularization
                loss_MLE = torch.mean(qij * π_log) #elementwise multiplication
                    
                #Outer optimization loop
                self.actor_optimizer.zero_grad()
                # last eq of p.5
                loss_actor = -(loss_MLE + self.η_kl * (self.ε_kl - kl))
                loss_actor.backward()
                if self.use_gradient_clipping:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clipping_value)
                self.actor_optimizer.step()
            
            #Keep track of the losses
            losses.append([loss_critic.item(), loss_MLE.item(), loss_actor.item()])
            
        #after each optimization, decay epsilon
        self.adjust_epsilon(episode_i)
        
        #update the target networks
        if episode_i % self.target_net_update_freq == 0:
            self.updateTargetNet(soft_update=self.use_soft_updates, source=self.critic , target=self.target_critic)
            self.updateTargetNet(soft_update=self.use_soft_updates, source=self.actor , target=self.target_actor)
                
        return losses 
    
    def setMode(self, eval: bool = False) -> None:
        """
        Set networks in train or eval mode.
        """
        self.isEval = eval
        if self.isEval:
            self.actor.eval()
            self.critic.eval()
        else:
            self.actor.train()
            self.critic.train()

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        checkpoint = torch.load(file_name, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_policy.load_state_dict(checkpoint["target_policy"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        print(f"Model loaded from {file_name}")

    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "target_policy": self.target_actor.state_dict(),
            "iteration": iteration
        }

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")
        os.makedirs(directory, exist_ok = True)
        torch.save(checkpoint, file_path)
        print(f"Actor and Critic weights saved successfully!")
