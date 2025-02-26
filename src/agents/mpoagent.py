import logging
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize
from torch.distributions import Categorical, MultivariateNormal

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path
from src.util.icmutil import ICM
from src.settings import USE_ENV
from src.util.mathutil import categorical_kl, gaussian_kl

"""
Author : Andre Pfrommer
"""

class Actor(nn.Module):
    """
    Policy network for continuous action space that outputs the mean and covariance matrix of 
    a multivariate Gaussian distribution
    
    - ds the dimension of the state space
    - da the dimension of the action space
    If action space continuous:
    - Mean layer outputs the mean vector with size (da)
    - Cholesky layer outputs the lower triangular matrix of the covariance matrix with size (da, da), 
    thus the output size is (da*(da+1))//2
    If action space discrete:
    - Softmax over all possible discrete actions da
    """
    def __init__(self, state_size, action_space, action_size, hidden_dim: int, continuous: bool, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.continuous = continuous
        self.ds = state_size
        
        #shape = (1, da)
        if self.continuous:
            self.da = action_size
            self.action_low = torch.tensor(action_space.low, device=self.device)[:self.da]
            self.action_high = torch.tensor(action_space.high, device=self.device)[:self.da]
        else:
            self.da = action_size 
            
        #Feedforward network
        self.net = nn.Sequential(
            nn.Linear(self.ds, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        if self.continuous:
            #Output layer for the mean
            self.mean_layer = nn.Linear(hidden_dim, self.da) 
            #Output layer for cholesky factorization of covariance matrix
            self.cholesky_layer = nn.Linear(hidden_dim, (self.da * (self.da + 1)) // 2)
        else:
            #Output layer here softmax over all possible discrete actions da
            self.lin3 = nn.Linear(hidden_dim, self.da)
            self.out = nn.Softmax(dim=-1)

    def forward(self, state: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        forwards input through the network
         First preprocessing the input state
         If action space continuous:
         - Output layer for the mean of multivariate Gaussian distribution
         - Output layer for the covraiance matrix of multivariate Gaussian distribution as cholesky factorization
         If action space discrete:
         - Softmax over all possible discrete actions da
         
        :param state: (B, ds), where B the batch size and ds the dimension of the state space
        :return: mean vector (B, da) and cholesky factorization of covariance matrix (B, da, da)
        """
        # Batch size
        B = state.size(0) 
        
        # Preprocess the input state
        x = self.net(state.to(self.device)).to(self.device)  # (B, 256)
        if self.continuous:
            
            action_low = self.action_low.to(self.device).unsqueeze(0)  # (1, da)
            action_high = self.action_high.to(self.device).unsqueeze(0)  # (1, da)
            
            mean = torch.sigmoid(self.mean_layer(x)).to(self.device)  # (B, da)
            mean = action_low + (self.action_high.unsqueeze(0) - action_low) * mean 
            
            # Output layer for the cholesky factorization of the covariance matrix
            cholesky_vector = self.cholesky_layer(x)  # (B, (da*(da+1))//2)
            cholesky_diag_index = torch.arange(self.da, dtype=torch.long, device=self.device) + 1
            
            # Calculate the index of the diagonal of the cholesky factorization
            cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
            
            # Ensure the diagonal of the cholesky factorization is positive
            cholesky_vector = cholesky_vector.clone()
            cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index]) + 1e-6
            
            # Build Cholesky matrix
            tril_indices = torch.tril_indices(row=self.da, col=self.da, offset=0)
            cholesky = torch.zeros(size=(B, self.da, self.da), dtype=torch.float32, device=self.device)
            
            # Fill the lower triangular matrix of the cholesky factorization defined above
            cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
            
            return mean, cholesky
        else:
            # 2. Output layer here softmax over all possible discrete actions da
            logits = self.out(self.lin3(x))
            return logits, None
    
    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        """
        :state: (B, ds) the current state
        :return: (B,) the greedy action
        """
        assert not self.continuous
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            greedyAction = torch.argmax(action_probs, dim = -1)
            return greedyAction

class Critic(nn.Module):
    """
    Critic (Q function) for MPO estimating the Q value of a state-action pair
    If action space continuous:
    - Input layer for the state and action
    - Output is a scalar Q value for the given action
    If action space discrete:
    - Input layer for only the state
    - Output layer for the Q value over all possible discrete actions da
    """
    def __init__(self, state_size, action_space, action_size, hidden_dim: int, continuous: bool):
        super(Critic, self).__init__()
        self.continuous = continuous
        self.ds = state_size
        if self.continuous:
            self.da = action_size
            # Get q value for the given action
            in_dim = self.ds + self.da
            out_dim = 1
        else:
            self.da = action_size 
            # Get q value over all actions
            in_dim = self.ds
            out_dim = self.da
            
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
 
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        :param state: (B, ds)
        :param action: (B, da)
        :return: Q-value
        """
        if self.continuous:
            h = torch.cat([state, action], dim=1)
        else:
            h = state
            
        q_value = self.net(h) # (B, 1) or (B, da)
        return q_value
    
    def QValue(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Q value for the given state and action
        Only needed for discrete action spaces 
        :param state: (B, ds)
        :param action: (B, 1)
        :return: Q-value
        """
        assert not self.continuous
        
        all_q_values = self.forward(state, action) # (B, da)
        # Select the Q value for the given action index
        action = action.view(-1, 1)
        q_value = all_q_values.gather(dim=1, index=action.long()) # (B, 1)
        return q_value

class MPOAgent(Agent):
    """
    The MPO Agent. 
    This agent is largely based on the paper "Relative Entropy Regularized Policy Iteration" by Haarnoja et al. (2018)
    https://arxiv.org/pdf/1812.02256.pdf and the Github 
    https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
    
    Parameters specific to the MPO agent:
    :param discrete:
        (bool) whether the action space is discrete or continuous
    :param hidden_dim: 
        (int) hidden layer dimension
    :param sample_action_num: 
        (int) number of actions to sample per state (N), irrelevant for discrete action spaces
    :param dual_constraint:
        (float) hard constraint of the dual formulation in the E-step
    :param kl_constraint:
        (float) hard constraint for the KL divergence in the M-step. Used for discrete case 
    :param kl_mean_constraint:
        (float) hard constraint of the mean in the M-step. Used for continuous case
    :param kl_var_constraint:
        (float) hard constraint of the covariance in the M-step. Used for continuous case
    
    """

    def __init__(self, agent_settings, device, state_space, action_space, mpo_settings, env):
        super().__init__(agent_settings, device)
        self.env = env
        
        self.device = device
        self.action_space = action_space
        self.state_space = state_space
        
        # Get the MPO settings
        self.continuous = not mpo_settings.get("DISCRETE", False)
        self.disc_to_cont_trafo = mpo_settings.get("DISC_TO_CONT_TRAFO", False)
        self.number_disc_actions = mpo_settings.get("NUMBER_DISCRETE_ACTIONS", 7)
        # NOTE: We need 7 actions in the discrete hockey environment
        self.train_icm_freq = mpo_settings.get("TRAIN_ICM_FREQ", 32)
        self.hidden_dim = mpo_settings.get("HIDDEN_DIM", 512)  
        self.curiosity = mpo_settings.get("CURIOSITY", None)
        self.sample_action_num = mpo_settings.get("SAMPLE_ACTION_NUM", 64) #N
        self.kl_constraint_scalar = mpo_settings.get("KL_CONSTRAINT_SCALAR", 0.1)
        self.ε_dual = mpo_settings.get("DUAL_CONSTAINT", 0.1) 
        self.ε_kl = mpo_settings.get("KL_CONSTRAINT", 0.01) 
        self.ε_kl_μ = mpo_settings.get("KL_CONSTRAINT_MEAN", 0.01) 
        self.ε_kl_Σ = mpo_settings.get("KL_CONSTRAINT_VAR", 0.0001) 
        
        # Get state and action dimensions
        self.ds = self.state_space.shape[0]
        if self.continuous:
            self.da = self.get_num_actions(action_space)
            self.action_low = torch.tensor(action_space.low, device=self.device)[:self.da]
            self.action_high = torch.tensor(action_space.high, device=self.device)[:self.da]
        else:
            self.da = self.number_disc_actions 
        
        
        # Initialize variables to optimize
        self.η = np.random.rand() #E step, dual variable
        self.η_kl = 0.0
        self.η_µ_kl = 0.0
        self.η_Σ_kl = 0.0 

        # Set up the actor and critic networks
        self.actor = Actor(self.ds, action_space, self.da, self.hidden_dim, self.continuous, self.device).to(self.device)
        self.critic = Critic(self.ds, action_space, self.da, self.hidden_dim, self.continuous).to(self.device)
        self.target_actor = Actor(self.ds, action_space, self.da, self.hidden_dim, self.continuous, self.device).to(self.device)
        self.target_critic = Critic(self.ds, action_space, self.da, self.hidden_dim, self.continuous).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initializing the optimizers
        self.actor_optimizer = self.initOptim(optim = mpo_settings["ACTOR"]["OPTIMIZER"],
                                           parameters = self.actor.parameters())
        self.critic_optimizer = self.initOptim(optim = mpo_settings["CRITIC"]["OPTIMIZER"],
                                           parameters = self.critic.parameters())

        # Define Loss function
        self.criterion = self.initLossFunction(loss_name = mpo_settings["CRITIC"]["LOSS_FUNCTION"])
        
        # Initialize intrinsic curiosity module 
        if self.curiosity is not None:
            self.icm = ICM(self.ds, self.da, 1 - self.continuous, device)
        
    def __repr__(self):
        """
        For printing purposes only
        """
        return f"MPOAgent"
        
    def act(self, state: torch.Tensor) -> np.ndarray:
        """
        Selects an action based on the current policy and evaluation mode. 
        :param state: (B, ds) the current state
        :return: action: (B, da) the action
        """
        with torch.no_grad():
            if self.continuous:
                # Get mean and covariance from the actor
                π_µ, π_A = self.actor(state.unsqueeze(0).to(self.device))  
                if self.isEval:
                    # If you are in eval mode, get the greedy Action
                    action = π_µ.cpu().numpy()[0]
                else:
                    # Define the action distribution as a multivariate Gaussian
                    π = MultivariateNormal(π_µ, scale_tril=π_A)  # (B, da)
                    # Sample action from the multivariate Gaussian
                    proposed_action = π.sample()
                    # Ensure actions are within the valid range
                    action = torch.clamp(proposed_action, self.action_low, self.action_high)
                    action = proposed_action.cpu().numpy()[0]
            else:
                if self.isEval:
                    # If you are in eval mode, get the greedy Action
                    action = self.actor.greedyAction(state).item()
                else:
                    # In training mode, use epsilon greedy action sampling
                    rdn = np.random.random()
                    if rdn <= self.epsilon:
                        # Exploration. Take a random action
                        return np.random.randint(low = 0, high = self.da)
                    else:
                        # Exploitation. Take the actions w.r.t. the policy
                        π_p, _ = self.actor(state)
                        π = Categorical(probs=π_p) 
                        action = π.sample().item()
                        
                if self.disc_to_cont_trafo:
                    action = self.env.discrete_to_continous_action(action)
        return action
    
    def critic_update(self, states: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor, 
                      rewards: torch.Tensor, sample_num=64) -> (torch.Tensor, torch.Tensor):
        """
        Compute the temporal difference loss and update the critic via gradient descent 
        
        :param states: (B, ds)
        :param actions: (B, da) 
        :param dones: (B,1)
        :param next_states: (B, ds)
        :param rewards (B,1)
        :param sample_num:
        :return:(float) the loss for the critic
        """
        B = states.size(0) #Batch size
        
        with torch.no_grad():
            
            if self.continuous:
                #target policy output given the next state
                π_μ, π_A = self.target_actor(next_states)  # (B,)
                #define a multivariate gaussian using the target policy output
                π = MultivariateNormal(π_μ, scale_tril=π_A)  # (B,)
                #sample N actions from the multivariate gaussian
                sampled_next_actions = π.sample((sample_num,)).transpose(0, 1)  # (B, sample_num, da)
                #expand the next state and reward to match the shape of the sampled actions
                expanded_next_states = next_states[:, None, :].expand(-1, sample_num, -1)  # (B, sample_num, ds)
                #Input them into the target q network
                expected_next_q = self.target_critic(
                    expanded_next_states.reshape(-1, self.ds),  # (B * sample_num, ds)
                    sampled_next_actions.reshape(-1, self.da)  # (B * sample_num, da)
                ).reshape(B, sample_num).mean(axis=1) # (B,)
                
            else:
                #Now we get only a single policy output
                π, _ = self.target_actor(next_states)  # (B, da)
                #and a categorical distribution
                π_dist = Categorical(probs=π)
                #and we dont need to sample as we can calculate the probabilities over all da actions 
                π_p = π_dist.probs  # (B, da)
                #Compute the expected q value under these policy probabilities
                next_q = self.target_critic(next_states, None)  # (B, da)
                expected_next_q = (π_p * next_q).sum(dim=1, keepdim=True).squeeze()  # (B,)
                
            #TD target
            q_new = rewards.squeeze(-1) + self.discount * (1 - dones.squeeze(-1)) * expected_next_q  # [200]
        
        #Calculate the loss
        if self.continuous:
            q = self.critic(states, actions).squeeze() # (B,)
        else:
            q = self.critic.QValue(states, actions).squeeze()# (B,)
            
        loss = self.criterion(q_new, q) #(B, B)
        return loss, q_new
    

    def find_qij_dist(self, q_target: torch.Tensor, π_target: torch.Tensor) -> torch.Tensor:
        """
        Find the action weights qij by applying two value constraints in a nonparametric way:
        1. Keep qij close to the target q values. The distribution of qij can computed in closed form by minimizing the dual function
        2. Apply softmax over all actions to normalize q values
        :param q_target: (K, da) or (K, N) the target q values
        :param π_target_np: (K, da) the target policy probabilities, only used in the discrete case
        :return: (K, da) or (K, N) the action weights qij
        """
        q_target_np = q_target.cpu().numpy()
        # If discrete, we can use the policy prob for all possible actions to generate the mean
        π_target_np = π_target.cpu().numpy() if not self.continuous else None 
            
        def dual(η):
            """
            Dual function for MPO with numerical stabilization.
            
            dual function of the non-parametric variational:
            g(η) = η*ε + η*mean(log(x)) where x = mean(exp(Q(s, a)/η))
            This equation corresponds to last equation of the [2] p.15 
            with some adjustments for numerical stability.
            
            :param η: Dual variable for constraint optimization.
            :return: Optimized dual function value.
            """
            # Max Q-value per state
            max_q = np.max(q_target_np, axis=1) 
            # Stabilized exponential term
            if self.continuous:
                x = np.mean(np.exp((q_target_np - max_q[:, None]) / η), axis=1)
            else:
                x = np.sum(π_target_np * np.exp((q_target_np - max_q[:, None]) / η), axis=1)
            # Avoid log(0) by clamping x
            x = np.clip(x, a_min=1e-8, a_max=None)
            # Dual function value
            g = η * self.ε_dual + np.mean(max_q) + η * np.mean(np.log(x))
            return g
        
        # Minimize the dual function using scipy minimize 
        res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=[(1e-3, None)])
        # Update the dual variable η
        self.η = res.x[0]
        
        # Compute action weights (new q values) using dual variable η 
        # Apply softmax over all actions to normailze q values (2nd constriant)
        qij = torch.softmax(q_target / self.η, dim=1) # (K, N) or (K, da)
        return qij

    def expectation_step(self, states: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Expectation step in the MPO algorithm. 
        Maximization of the lower bounnd w.r.t the optimal policy by regularizing the q values towards the current best policy.
        This is done in a nonparametric way by: 
            1. Get the output of the target policy
            2. For each of the K states, sample N actions
            3. Compute the target q values for each action
            NOTE: In the discrete case, we dont sample but compute the q values for all possible actions
            4. Minimize the dual function to find the action weights qij
        
        :states: (B, ds) the batch of states
        :N: (int) the number of actions to sample per state
        :K: (int) the number of sampled states
        """
        N = self.sample_action_num
        K = self.batch_size
        
        with torch.no_grad():
            ################# Continuous E step ##################
            if self.continuous:
                # 1. We first get the mean and covariance of the target policy
                b_μ, b_A = self.target_actor(states) # (K,) K batch size
                dist = MultivariateNormal(b_μ, scale_tril=b_A) # (K,)
                # 2. Sample N actions per state (we have K states)
                sampled_actions = dist.sample((N,))  # (N, K, da)
                expanded_states = states.unsqueeze(0).expand(N, -1, -1)  # (N, K, ds)
                # 3. Get the target q values for the K sampled states and N actions
                target_q = self.target_critic(
                    expanded_states.reshape(-1, self.ds),  # (N * K, ds)
                    sampled_actions.reshape(-1, self.da)  # (N * K, da)
                ).reshape(N, K)  # (N, K)
                # 4. Minimize the dual function to find the action weights qij
                qij = self.find_qij_dist(target_q.transpose(0, 1), None) # (K, N) 
                
                return sampled_actions, qij, b_μ, b_A
            ################## Discrete E step ####################
            else:
                # 1. Here we also get the policy output first, but again we dont sample 
                b, _ = self.target_actor(states) # (K, da)
                dist = Categorical(probs=b)
                # 2. Compute probabilities over all actions
                b_p = dist.probs  # (K, da)
                # 3. Get the target q values over all discrete actions
                target_q = self.target_critic(states, None) # (K, da)
                # 4. Minimize the dual function to find the action weights qij
                qij = self.find_qij_dist(target_q, b_p) # (K, da) 
                
                return None, qij, b, None
        
    def maximization_step(self, states: torch.Tensor, sampled_actions: torch.Tensor, qij: torch.Tensor, 
                          b_μ: torch.Tensor, b_A: torch.Tensor, b_p: torch.Tensor) -> (torch.Tensor, float, float):
        """
        Maximization step in the MPO algorithm.
        Improve the current policy by maximizing π towards the action weights qij.
        This is done by:
            1. Get the output of the current policy to compute the MLE loss using qij
            2. Get the KL divergence between the current and target policy
            3. Update the Lagrangian multipliers by gradient descent (inner minimization loop)
            4. Combine everything to get the final loss for the actor (outer maximization loop)
        
        :states: (B, ds) the batch of states
        :sampled_actions: (N, K, da) the sampled actions
        :qij: (K, N) or (K, da) the action weights
        :b_μ: (K, da) the mean of the target policy, used in the continuous case
        :b_A: (K, da, da) the covariance matrix of the target policy, used in the continuous case
        :b_p: (K, da) the probabilities of the target policy, used in the discrete case
        """
        N = self.sample_action_num
        K = self.batch_size
        ################# Continuous M step ##################
        if self.continuous:
            # 1. Mean and covariance of the current policy
            μ, A = self.actor(states) # (K,)
            # Mulitvariave Gaussian distributions with either the mean or covariance fixed to the target policy output
            π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (K,)
            π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (K,)
            
            # 2. Get the KL divergencies for the above defined distributions
            kl_μ, kl_Σ = gaussian_kl(μi=b_μ, μ=μ, Ai=b_A, A=A)
            
            # 3. Update lagrangian multipliers by gradient descent
            self.η_μ_kl -= 0.01 * (self.ε_kl_μ - kl_μ).detach().item()
            self.η_Σ_kl -= 0.01 * (self.ε_kl_Σ - kl_Σ).detach().item()
            self.η_μ_kl = np.clip(0.0, self.η_μ_kl, 10.0)
            self.η_Σ_kl = np.clip(0.0, self.η_Σ_kl, 10.0)
            
            # 4. Then we add the KL constraints to the loss (outer optimization loop), last eq of p.5
            loss_MLE = torch.mean(
                qij.transpose(0, 1) * (
                    π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                    + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                )
            )
            if self.kl_constraint_scalar is None:
                loss_actor = -(
                    loss_MLE
                    + self.η_μ_kl * (self.ε_kl_μ - kl_μ)
                    + self.η_Σ_kl * (self.ε_kl_Σ - kl_Σ)
                )
            else:
                # If we dont use lagrangian optimization, just use the given factor
                loss_actor = -(
                    loss_MLE
                    + self.kl_constraint_scalar * (self.ε_kl_μ - kl_μ)
                    + self.kl_constraint_scalar * (self.ε_kl_Σ - kl_Σ)
                )
            
            return loss_actor, kl_μ.detach(), kl_Σ.detach()
        ################## Discrete M step ##################
        else:
            # Creates action tensor of shape (da, K), where each of the K rows contains repeated indices ranging from 0 to self.da - 1
            actions = torch.arange(self.da).unsqueeze(1).expand(self.da, K).to(self.device)  # (da, K)
            
            # 1. Action output of the current parametric policy
            π_p, _ = self.actor.forward(states)  # (K, da)
            π = Categorical(probs=π_p)  # (K,)
            π_log = π.log_prob(actions).T  # (K, da)
            
            # 2. KL divergence btw the old and new policy, now a categorical one
            kl = categorical_kl(p1=π_p, p2=b_p).detach()
            
            # 3. Inner optimization loop
            self.η_kl -= 0.01 * (self.ε_kl - kl).detach().item()
            self.η_kl = np.clip(0.0, self.η_kl, 10.0)
            
            # 4. Final loss
            loss_MLE = torch.mean(qij * π_log) # (K, da)
            if self.kl_constraint_scalar is None:
                # Use the optimized scalar
                loss_actor = -(
                    loss_MLE + self.η_kl * (self.ε_kl - kl)
                )
            else:
                # Use the given scalar
                loss_actor = -(
                    loss_MLE 
                    + self.kl_constraint_scalar * (self.ε_kl - kl)
                )
            
            return loss_actor, kl.detach(), None
                    
    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        Optimize actor and critic networks based on experience replay.
            1. Policy Evaluation: Update Critic via TD Learning
            2. E-Step of Policy Improvement: Sample from Critic and update dual variable η to find action weights qij
            3. M-Step of Policy Improvement: Update Actor via gradient ascent on the MLP loss
        
        :param memory: (ReplayMemory) the replay memory
        :param episode_i: (int) the current episode number
        :return: (list[float]) the losses of the actor and critic networks and Kl divergencies
        """
        losses = []
        
        # Nr of actions to sample per state, irrelevant for discrete action spacessince we select all da actions per state
        N = self.sample_action_num 
        # Nr of sampled states, here the Batch size
        K = self.batch_size  
        for i in range(self.opt_iter):
            # Sample from replay buffer, dimensions (K, ds), (K, da), (K,), (K, ds) 
            states, actions, rewards, next_states, dones, info = memory.sample(batch_size=self.batch_size, randomly=True)
            
            # Train the curiosity module 
            if self.curiosity is not None and i % self.train_icm_freq == 0:
                self.icm.train(states, next_states, actions)
            
            # 1: Policy Evaluation: Update Critic (Q-function)
            loss_critic, q_estimates = self.critic_update(states, actions, dones, next_states, rewards, N)
            # Backward pass in the critic network
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clipping_value,
                                         foreach = self.use_clip_foreach)
            self.critic_optimizer.step()

            # 2: E-Step: Finding action weights via sampling and non-parametric optimization (4.1 in paper)
            if self.continuous:
                sampled_actions, qij, b_μ, b_A = self.expectation_step(states)
            else:
                _, qij, b_p, _ = self.expectation_step(states)

            # 3. M step. Policy Improvement (4.2 in paper)
            # Fitting an improved policy using the sampled q-values via gradient optimization on the Policy and the lagrangian function 
            if self.continuous:
                loss_actor, kl_µ, kl_Σ = self.maximization_step(states, sampled_actions, qij, b_μ, b_A, None)
            else:
                loss_actor, kl, _ = self.maximization_step(states, None, qij, None, None, b_p)
            
            # Backward pass in the actor network
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clipping_value,
                                         foreach = self.use_clip_foreach)
            self.actor_optimizer.step()
            
            # Keeping track of the losses and KL divergencies
            if self.continuous:
                losses.append([loss_critic.item(), loss_actor.item(), kl_µ.item(), kl_Σ.item()])
            else:
                losses.append([loss_critic.item(), loss_actor.item(), kl.item(), 0.0])
           
        # Update the epsilon value, used for epsilon-greedy action selection
        self.adjust_epsilon(episode_i)
        
        # Update the target networks
        if episode_i % self.target_net_update_freq == 0:
            self._copy_nets()

        avg_losses = np.mean(losses, axis=0).tolist()
        
        return avg_losses
    
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
    
    def _copy_nets(self) -> None:
        assert self.use_target_net == True
        # Step 01: Copy the actor net
        self.updateTargetNet(soft_update = self.use_soft_updates, source = self.actor,
                             target = self.target_actor)
        # Step 02: Copy the critic net
        self.updateTargetNet(soft_update = self.use_soft_updates, source = self.critic,
                             target = self.target_critic)
    
    def import_checkpoint(self, checkpoint: dict) -> None:
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_actor.load_state_dict(checkpoint["target_actor"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
        }
        return checkpoint

    def reset(self):
        pass
