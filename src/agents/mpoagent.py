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

"""
Author : Andre Pfrommer
"""

class Actor(nn.Module):
    """
    Policy network for continuous action space that outputs the mean and covariance matrix of a multivariate Gaussian distribution
    
    - ds the dimension of the state space
    - da the dimension of the action space
    - Mean layer outputs the mean vector with size (da)
    - Cholesky layer outputs the lower triangular matrix of the covariance matrix with size (da, da), 
    thus the output size is (da*(da+1))//2
    """
    def __init__(self, state_space, action_space, hidden_dim: int, continuous: bool):
        super().__init__()
        self.continuous = continuous
        self.ds = state_space.shape[0]
        
        #shape = (1, da)
        if self.continuous:
            self.da = int(np.prod(action_space.shape)) # e.g. 1 if shape=(1,)
            self.action_low = torch.tensor(action_space.low)
            self.action_high = torch.tensor(action_space.high)
        else:
            self.da = action_space.n
        
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
        #Batch size
        B = state.size(0) 
        
        # 1. Preprocess the input state
        x = self.net(state)  # (B, 256)
        if self.continuous:
            # 2. Output layer for the mean and rescaling into the action space
            action_low = self.action_low.unsqueeze(0)  # (1, da)
            action_high = self.action_high.unsqueeze(0)  # (1, da)
            
            mean = torch.sigmoid(self.mean_layer(x))  # (B, da)
            mean = action_low + (action_high - action_low) * mean 
            
            # 3. Output layer for the cholesky factorization of the covariance matrix
            cholesky_vector = self.cholesky_layer(x)  # (B, (da*(da+1))//2)
            cholesky_diag_index = torch.arange(self.da, dtype=torch.long) + 1
            #Calculate the index of the diagonal of the cholesky factorization
            cholesky_diag_index = (cholesky_diag_index * (cholesky_diag_index + 1)) // 2 - 1
            
            #Ensure the diagonal of the cholesky factorization is positive
            cholesky_vector = cholesky_vector.clone()
            cholesky_vector[:, cholesky_diag_index] = F.softplus(cholesky_vector[:, cholesky_diag_index]) + 1e-6
            
            #stabelize off diagonal elements
            tril_indices = torch.tril_indices(row=self.da, col=self.da, offset=0)
            cholesky_vector[:, tril_indices[0] != tril_indices[1]] = torch.clamp(
                cholesky_vector[:, tril_indices[0] != tril_indices[1]], min=-1.0, max=1.0
            )
            
            #Build Cholesky matrix
            cholesky = torch.zeros(size=(B, self.da, self.da), dtype=torch.float32)
            #Fill the lower triangular matrix of the cholesky factorization defined above
            cholesky[:, tril_indices[0], tril_indices[1]] = cholesky_vector
            
            # Add small value to diagonal
            diag_indices = torch.arange(self.da)
            cholesky[:, diag_indices, diag_indices] += 1e-6
            return mean, cholesky
        else:
            # 2. Output layer here softmax over all possible discrete actions da
            logits = self.out(self.lin3(x))
            return logits, None
    
    def greedyAction(self, state: torch.Tensor) -> torch.Tensor:
        """
        :param:
            (B, ds) the state tensor
        :return:
            (B,) the greedy action
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
    :param env: OpenAI gym environment
    """
    def __init__(self, state_space, action_space, hidden_dim: int, continuous: bool):
        super(Critic, self).__init__()
        self.continuous = continuous
        self.ds = state_space.shape[0]
        if self.continuous:
            self.da = int(np.prod(action_space.shape))
        else:
            self.da = action_space.n
        if self.continuous:
            self.net = nn.Sequential(
                nn.Linear(self.ds + self.da, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, 1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(self.ds, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LeakyReLU(),
                nn.Linear(hidden_dim, self.da),
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
        """
        assert not self.continuous
        
        all_q_values = self.forward(state, action) # (B, da)
        q_value = all_q_values.gather(dim=1, index=action.long().unsqueeze(1)) # (B, 1)
        
        return q_value
    
    
################################################################################
# Maximum a posteriori policy optimization (MPO) Agent
################################################################################

class MPOAgent(Agent):
    """
    The MPO Agent. 
    This agent is largely based on the paper "Relative Entropy Regularized Policy Iteration" by Haarnoja et al. (2018)
    https://arxiv.org/pdf/1812.02256.pdf and the Github 
    https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
    
    Parameters specific to the MPO agent:
    
    :param hidden_dim: 
        (int) hidden layer dimension
    :param sample_action_num: 
        (int) number of actions to sample per state (N), irrelevant for discrete action spaces
    :param mstep_iteration_num:
        (int) the number of iterations of the M-step. Lets keep this to 1 first
    :param dual_constraint:
        (float) hard constraint of the dual formulation in the E-step
    :param kl_constraint:
        (float) hard constraint for the KL divergence in the M-step. Used for discrete case 
    :param kl_mean_constraint:
        (float) hard constraint of the mean in the M-step. Used for continuous case
    :param kl_var_constraint:
        (float) hard constraint of the covariance in the M-step. Used for continuous case
    
    """
    def __init__(self, agent_settings, device, state_space, action_space, mpo_settings):
        super().__init__(agent_settings, device)
        
        #Continous or discrete action space
        self.continuous = True
        if agent_settings.get("NUMBER_DISCRETE_ACTIONS") is not None and agent_settings.get("NUMBER_DISCRETE_ACTIONS") > 0:
            self.continuous = False

        self.device = device
        self.action_space = action_space
        self.state_space = state_space
        
        self.ds = self.state_space.shape[0]
        if self.continuous:
            self.da = int(np.prod(action_space.shape))
            self.action_low = torch.tensor(action_space.low, device = self.device)
            self.action_high = torch.tensor(action_space.high, device = self.device)
        else:
            self.da = action_space.n
        
        self.hidden_dim = mpo_settings.get("HIDDEN_DIM", 128) 
        self.sample_action_num = mpo_settings.get("SAMPLE_ACTION_NUM", 64) #N
        self.mstep_iteration_num = mpo_settings.get("MSTEP_ITER", 1)
        self.ε_dual = mpo_settings.get("DUAL_CONSTAINT", 0.1) 
        self.ε_kl = mpo_settings.get("KL_CONSTRAINT", 0.1) 
        self.ε_kl_μ = mpo_settings.get("KL_CONSTRAINT_MEAN", 0.01) 
        self.ε_kl_Σ = mpo_settings.get("KL_CONSTRAINT_VAR", 0.0001) 
        
        # Clipping values for lagrangian multipliers
        self.α_max = mpo_settings.get("ALPHA_MAX", 1.0)
        self.α_μ_max = mpo_settings.get("ALPHA_MAX_MU", 1.0)
        self.α_Σ_max = mpo_settings.get("ALPHA_MAX_VAR", 1.0)
        
        #initialize variables to optimize
        self.η = np.random.rand() #E step, dual variable
        #self.η_kl = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device = self.device) #discrete M step
        #self.η_µ_kl = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device = self.device) #continuous M step
        #self.η_Σ_kl = torch.tensor(0.0, dtype=torch.float32, requires_grad=True, device = self.device) #continuous M step
        self.η_kl = 0.0
        self.η_μ_kl = 0.0
        self.η_Σ_kl = 0.0

        #Set up the actor and critic networks
        self.actor = Actor(state_space, action_space, self.hidden_dim, self.continuous).to(self.device)
        self.critic = Critic(state_space, action_space, self.hidden_dim, self.continuous).to(self.device)
        self.target_actor = Actor(state_space, action_space, self.hidden_dim, self.continuous).to(self.device)
        self.target_critic = Critic(state_space, action_space, self.hidden_dim, self.continuous).to(self.device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # Initializing the optimizers
        self.actor_optimizer = self.initOptim(optim = mpo_settings["ACTOR"]["OPTIMIZER"],
                                           parameters = self.actor.parameters())
        self.critic_optimizer = self.initOptim(optim = mpo_settings["CRITIC"]["OPTIMIZER"],
                                           parameters = self.critic.parameters())

        #Define Loss function
        self.criterion = self.initLossFunction(loss_name = mpo_settings["CRITIC"]["LOSS_FUNCTION"])
        
        #Initialize intrinsic curiosity module 
        self.icm = ICM(self.ds, self.da, 1 - self.continuous)
        
    def __repr__(self):
        """
        For printing purposes only
        """
        return f"MPOAgent"
        
    def act(self, state: torch.Tensor):
        """
        Selects an action based on the current policy and evaluation mode. 
        :param state:
            (B, ds) the current state
        :return:
            action: (B, da) the action
        """
        with torch.no_grad():
            if self.continuous:
                π_µ, π_A = self.actor(state.unsqueeze(0))  # Get mean and covariance from the actor
                
                if self.isEval:
                    # if you are in eval mode, get the greedy Action
                    action = π_µ.numpy()[0]
                else:
                    # Define the action distribution as a multivariate Gaussian
                    π = MultivariateNormal(π_µ, scale_tril=π_A)  # (B, da)
                    # Sample action from the multivariate Gaussian
                    proposed_action = π.sample()
                    # Ensure actions are within the valid range
                    action = torch.clamp(proposed_action, self.action_low, self.action_high)
                    action = proposed_action.numpy()[0]
            else:
                if self.isEval:
                    # if you are in eval mode, get the greedy Action
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
        return action
    
    def critic_update(self, states: torch.Tensor, actions: torch.Tensor, dones: torch.Tensor, next_states: torch.Tensor, rewards: torch.Tensor, sample_num=64):
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
    
    def gaussian_kl(self, μi, μ, Ai, A):
        """
        Decoupled KL between two multivariate gaussian distributions f (updated policy) and g (previous policy).

        C_μ = KL(g(x|μi,Σi)||f(x|μ,Σi))
        C_Σ = KL(g(x|μi,Σi)||f(x|μi,Σ))
        :param μi: (B, n) mean fixed to the mean of the previous policy g
        :param μ: (B, n) mean of the updated policy f
        :param Ai: (B, n, n) lower triangular matrix of the covariance of the previous policy g
        :param A: (B, n, n) lower triangular matrix of the covariance of the updated policy f
        :return: C_μ, C_Σ: scalar
            mean and covariance terms of the KL
        :return: mean of determinanats of Σi, Σ
        ref : https://stanford.edu/~jduchi/projects/general_notes.pdf page.13
        """
        def bt(m):
            return m.transpose(dim0=-2, dim1=-1)

        def btr(m):
            return m.diagonal(dim1=-2, dim2=-1).sum(-1)
        
        n = A.size(-1)
        μi = μi.unsqueeze(-1)  # (B, n, 1)
        μ = μ.unsqueeze(-1)  # (B, n, 1)
        Σi = Ai @ bt(Ai)  # (B, n, n)
        Σ = A @ bt(A)  # (B, n, n)
        
        # determinant can be minus due to numerical calculation error
        # https://github.com/daisatojp/mpo/issues/11. 
        # Trying to fix inf values via logdet
        Σi_logdet = Σi.logdet()  # (B,)
        Σ_logdet = Σ.logdet()  # (B,)
        #Σi_det = torch.clamp_min(Σi_det, 1e-4)
        #Σ_det = torch.clamp_min(Σ_det, 1e-4)
        
        #Inverse of the covariance matrices
        Σi_inv = Σi.inverse()  # (B, n, n)
        Σ_inv = Σ.inverse()  # (B, n, n)
        
        inner_μ = ((μ - μi).transpose(-2, -1) @ Σi_inv @ (μ - μi)).squeeze()  # (B,)
        inner_Σ = Σ_logdet - Σi_logdet - n + btr(Σ_inv @ Σi) # (B,)
        
        #Mean and covariance terms of the KL divergence
        C_μ = 0.5 * torch.mean(inner_μ)
        C_Σ = 0.5 * torch.mean(inner_Σ)
        
        return C_μ, C_Σ

    def find_qij_dist(self, q_target: torch.Tensor, π_target: torch.Tensor) -> torch.Tensor:
        """
        Find the action weights qij by applying two value constraints in a nonparametric way:
        1. Keep qij close to the target q values. The distribution of qij can computed in closed form by minimizing the dual function
        2. Apply softmax over all actions to normalize q values
        :param q_target:
            (K, da) or (K, N) the target q values
        :param π_target_np:
            (K, da) the target policy probabilities, only used in the discrete case
        """
        q_target_np = q_target.numpy()
        if not self.continuous:
            π_target_np = π_target.numpy() 
            
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
            #Stabilized exponential term
            if self.continuous:
                x = np.mean(np.exp((q_target_np - max_q[:, None]) / η), axis=1)
            else:
                x = np.sum(π_target_np * np.exp((q_target_np - max_q[:, None]) / η), axis=1)
            # Avoid log(0) by clamping x
            x = np.clip(x, a_min=1e-8, a_max=None)
            # Dual function value
            g = η * self.ε_dual + np.mean(max_q) + η * np.mean(np.log(x))
            return g
        
        #Minimize the dual function using the scipy minimize function (1st constraint)
        res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=[(1e-6, None)])
        #Update the dual variable η
        #print(f"η: {res.x[0]}")
        self.η = res.x[0]
        
        # Compute action weights (new q values) using dual variable η (see 3rd eq on page 4)
        # Apply softmax over all actions to normailze q values (2nd constriant)
        qij = torch.softmax(q_target / self.η, dim=1) # (K, N) or (K, da)
        return qij

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        Optimize actor and critic networks based on experience replay.
            1. Policy Evaluation: Update Critic via TD Learning
            2. E-Step of Policy Improvement: Sample from Critic and update dual variable η to find action weights qij
            3. M-Step of Policy Improvement: Update Actor via gradient ascent on the MLP loss
        
        :param memory:
            (ReplayMemory) the replay memory
        :param episode_i:
            (int) the current episode number
        :return:    
            (list[float]) the losses of the actor and critic networks
        """
        losses = []
        
        #Nr of actions to sample per state, irrelevant for discrete action spacessince we select all da actions per state
        N = self.sample_action_num 
        #Nr of sampled states, here the Batch size
        K = self.batch_size  
        for _ in range(self.opt_iter):
            # Sample from replay buffer, dimensions (K, ds), (K, da), (K,), (K, ds) 
            states, actions, rewards, next_states, dones, info = memory.sample(batch_size=self.batch_size, randomly=True)
            # Train the curiosity module 
            # self.icm.train(states, next_states, actions)
            
            # 1: Policy Evaluation: Update Critic (Q-function)
            loss_critic, q_estimates = self.critic_update(states, actions, dones, next_states, rewards, N)
            # Backward pass in the critic network
            self.critic_optimizer.zero_grad()
            loss_critic.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clipping_value, 2)
            self.critic_optimizer.step()

            # 2: E-Step Finding action weights via sampling and non-parametric optimization (4.1 in paper)
            with torch.no_grad():
                if self.continuous:
                    #We first get the mean and covariance of the target policy
                    b_μ, b_A = self.target_actor(states) # (K,) K batch size
                    dist = MultivariateNormal(b_μ, scale_tril=b_A) # (K,)
                    #sample N actions per state (we have K states)
                    sampled_actions = dist.sample((N,))  # (N, K, da)
                    expanded_states = states.unsqueeze(0).expand(N, -1, -1)  # (N, K, ds)
                    #Get the target q values for the K sampled states and N actions
                    target_q = self.target_critic(
                        expanded_states.reshape(-1, self.ds),  # (N * K, ds)
                        sampled_actions.reshape(-1, self.da)  # (N * K, da)
                    ).reshape(N, K)  # (N, K)
                    #Minimize the dual function to find the action weights qij
                    qij = self.find_qij_dist(target_q.transpose(0, 1), None) # (K, N) 
                else:
                    #Here we also get the policy output first, but again we dont sample 
                    b, _ = self.target_actor(states) # (K, da)
                    dist = Categorical(probs=b)
                    #compute probabilities over all actions
                    b_p = dist.probs  # (K, da)
                    #Get the target q values over all discrete actions
                    target_q = self.target_critic(states, None) # (K, da)
                    #Minimize the dual function to find the action weights qij
                    qij = self.find_qij_dist(target_q, b_p) # (K, da) 

            # 3. M step. Policy Improvement (4.2 in paper)
            #Fitting an improved policy using the sampled q-values via gradient optimization on the Policy and the lagrangian function 
            for _ in range(self.mstep_iteration_num):
                if self.continuous:
                    #mean and covariance of the current policy
                    μ, A = self.actor(states) # (K,)
                    #Mulitvariave Gaussian distributions with either the mean or covariance fixed to the target policy output
                    π1 = MultivariateNormal(loc=μ, scale_tril=b_A)  # (K,)
                    π2 = MultivariateNormal(loc=b_μ, scale_tril=A)  # (K,)
                    
                    #get the KL divergencies for the above defined distributions
                    kl_μ, kl_Σ = self.gaussian_kl(μi=b_μ, μ=μ, Ai=b_A, A=A)
                    
                    # Update lagrangian multipliers α by gradient descent
                    
                    #lagrangian_loss = self.η_μ_kl * (self.ε_kl_μ - kl_μ) + self.η_Σ_kl * (self.ε_kl_Σ - kl_Σ)
                    #self.lagrangian_optimizer_cont.zero_grad()
                    #lagrangian_loss.backward(retain_graph=True)
                    #self.lagrangian_optimizer_cont.step()
                    
                    #self.η_μ_kl = torch.clamp(self.η_μ_kl.detach(), 0.0, self.α_μ_max)
                    #self.η_Σ_kl = torch.clamp(self.η_Σ_kl.detach(), 0.0, self.α_Σ_max)
                    
                    #η_μ_kl_np = self.η_μ_kl.detach().item()
                    #η_Σ_kl_np = self.η_Σ_kl.detach().item()
                    
                    self.η_μ_kl -= 0.01 * (self.ε_kl_μ - kl_μ).detach().item()
                    self.η_Σ_kl -= 0.01 * (self.ε_kl_Σ - kl_Σ).detach().item()
                
                    self.η_μ_kl = np.clip(self.η_μ_kl, 0.0, self.α_μ_max)
                    self.η_Σ_kl = np.clip(self.η_Σ_kl, 0.0, self.α_Σ_max)
                    
                    
                    #First we compute the known MLE Loss without the KL constraints 
                    #Note that here we have 2 actor objectives, so we have to compute the log prob for both
                    loss_MLE = torch.mean(
                        qij.transpose(0, 1) * (
                            π1.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                            + π2.expand((N, K)).log_prob(sampled_actions)  # (N, K)
                        )
                    )
                    #Then we add the KL constraints to the loss (outer optimization loop)
                    # last eq of p.5
                    loss_actor = -(
                            loss_MLE
                            + self.η_μ_kl * (self.ε_kl_μ - kl_μ)
                            + self.η_Σ_kl * (self.ε_kl_Σ - kl_Σ)
                    )
                else:
                    #Creates action tensor of shape (da, K), where each of the K rows contains repeated indices ranging from 0 to self.da - 1
                    actions = torch.arange(self.da).unsqueeze(1).expand(self.da, K).to(self.device)  # (da, K)
                    
                    #action output of the current parametric policy
                    π_p, _ = self.actor.forward(states)  # (K, da)
                    π = Categorical(probs=π_p)  # (K,)
                    π_log = π.log_prob(actions).T  # (K, da)
                    
                    #KL divergence btw the old and new policy, now a categorical one
                    kl = self.categorical_kl(p1=π_p, p2=b_p).detach()
                    
                    if np.isnan(kl.item()):  # This should not happen
                        raise RuntimeError('kl is nan')
                    
                    #Inner optimization loop
                    self.η_kl -= 1.0 * (self.ε_kl - kl).item()
                    #lagrangian_loss = self.η_kl * (self.ε_kl - kl)
                    #self.lagrangian_optimizer_disc.zero_grad()
                    #lagrangian_loss.backward(retain_graph=True)
                    #self.lagrangian_optimizer_disc.step()
                    #η_kl_np = self.η_kl.detach().item()
                    
                    #Clipping
                    self.η_kl = np.clip(self.η_kl, 0.0, self.α_max)
                    
                    #MLE loss
                    loss_MLE = torch.mean(qij * π_log) # (K, da)
                    
                    #Final loss
                    loss_actor = -(loss_MLE + self.η_kl * (self.ε_kl - kl))
                
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                if self.use_gradient_clipping:
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clipping_value, 2)
                self.actor_optimizer.step()
            
            #Keep track of the losses
            losses.append([loss_critic.item(), loss_actor.item(), self.η_μ_kl, self.η_Σ_kl])
            
        #Update the epsilon value
        self.adjust_epsilon(episode_i)
        
        #update the target networks
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
        logging.info(f"Iteration: {iteration} MPO checkpoint saved successfully!")
        
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
        #self.η_μ_kl = checkpoint["lagrangian_µ"]
        #self.η_Σ_kl = checkpoint["lagrangian_Σ"]

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_actor": self.target_actor.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            #"lagrangian_µ": self.η_μ_kl,
            #"lagrangian_Σ": self.η_Σ_kl,
        }
        return checkpoint

    def reset(self):
        raise NotImplementedError
