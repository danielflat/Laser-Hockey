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
            nn.Linear(self.ds + self.da, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        """
        :param state: 
            (B, ds) the state tensor
        :param action: 
            (B, da) the action tensor
        :return: 
            (B,) Q-value
        """
            
        h = torch.cat([state, action], dim=1)  # (B, ds+da)
        return self.net(h)
    
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
        #self.ε_kl = mpo_settings.get("KL_CONSTRAINT", 0.001) 
        self.ε_kl = agent_settings.get("EPSILON")
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
        actor_optim_cfg = agent_settings.get("OPTIMIZER", None)
        critic_optim_cfg = agent_settings.get("OPTIMIZER", None)
        self.actor_optimizer = self.initOptim(actor_optim_cfg, self.actor.parameters())
        self.critic_optimizer = self.initOptim(critic_optim_cfg, self.critic.parameters())
        
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
        
    def update_critic_td(self, states, actions, dones, next_states, rewards):
        """
        Compute the temporal difference loss and update the critic via gradient descent
        
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
        
        B = states.size(0) #Batch size
        with torch.no_grad():
            
            #target policy output given the next state
            π_p = self.target_actor(next_states)  # (B, da)
            
            #define a Categorical dist using the target policy output
            π = Categorical(probs=π_p)  # (B,)
            
            π_prob = π.expand((self.da, B)).log_prob(
                torch.arange(self.da)[..., None].expand(-1, B).to(self.device)  # (da, B)
            ).exp().transpose(0, 1)  # (B, da)
            
            #Create a one-hot encoded action tensor
            sampled_next_actions = self.A_eye.unsqueeze(0).expand(B, -1, -1) # (B, da, da)
            expanded_next_states = next_states.reshape(B, 1, self.ds).expand((B, self.da, self.ds))  # (K, da, ds)
            
            #Q value for the next states averaged over all possible actions weighted by the policy probabilities
            expected_next_q = (
                self.target_critic(
                    expanded_next_states.reshape(-1, self.ds),  # (B * da, ds)
                    sampled_next_actions.reshape(-1, self.da)  # (B * da, da)
                ).reshape(B, self.da) * π_prob  # (B, da)
            ).sum(dim=-1)  # (B,)
            
            q_new = rewards.squeeze() + self.discount * (1 - dones.squeeze()) * expected_next_q # (B)
            
            
        #Update the critic using the q_target
        self.critic_optimizer.zero_grad()
        
        one_hot_actions = self.A_eye[actions.squeeze().long()]  # (B, da)
        q = self.critic(states, one_hot_actions).squeeze(-1)  # (B,)
        
        loss = self.norm_loss_q(q_new, q)
        loss.backward()
        if self.use_gradient_clipping:
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.gradient_clipping_value)
        self.critic_optimizer.step()
        
        return loss

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
            loss_critic = self.update_critic_td(states, actions, dones, next_states, rewards)

            # 2: E-Step Finding action weights via sampling and non-parametric optimization (4.1 in paper)
            with torch.no_grad():
                
                #Creates action tensor of shape (da, K), where each of the K rows contains repeated indices ranging from 0 to self.da - 1
                actions = torch.arange(self.da).unsqueeze(1).expand(self.da, K).to(self.device)  # (da, K)
                
                #Get the policy output  (distribution object) for the K sampled states
                b_p = self.target_actor(states)  # (K, da)
                
                #Categorical distribution of the actions
                b = Categorical(probs=b_p)  
                
                #Get the probability for all the possible actions
                b_prob = b.expand((self.da, K)).log_prob(actions).exp()  # (da, K)
                
                #Create a one-hot encoded action tensor 
                expanded_actions = self.A_eye.unsqueeze(0).expand(K, -1, -1)  # (K, da, da)
                
                #Create a tensor of the states with a similar shape as the action tensor
                expanded_states = states.reshape(K, 1, self.ds).expand((K, self.da, self.ds))  # (K, da, ds)
                
                #Get the q values for the K sampled states and *all* actions
                target_q = (
                    self.target_critic(
                        expanded_states.reshape(-1, self.ds),  # (K * da, ds)
                        expanded_actions.reshape(-1, self.da)  # (K * da, da)
                    ).reshape(K, self.da)  # (K, da)
                ).transpose(0, 1)  # (da, K)
                
                #Convert the tensors to numpy arrays
                b_prob_np = b_prob.cpu().transpose(0, 1).numpy()  # (K, da)
                
                #categorical probability distribution of the actions, in the paper this is q(ai|sj)
                target_q_np = target_q.cpu().transpose(0, 1).numpy()  # (K, da)
                
                    
            #Optimize dual variable η
            def dual(η):
                """
                dual function of the non-parametric variational
                g(η) = η*ε + η*mean(log(sum(π(a|s)*exp(Q(s, a)/η))))
                We have to multiply π by exp because this is expectation.
                This equation is correspond to last equation of the [2] p.15
                For numerical stabilization, this can be modified to
                Qj = max(Q(s, a), along=a)
                g(η) = η*ε + mean(Qj, along=j) + η*mean(log(sum(π(a|s)*(exp(Q(s, a)-Qj)/η))))
                
                I got this function from p.4 of the paper and the Github
                https://github.com/daisatojp/mpo/blob/master/mpo/mpo.py
                """
                max_q = np.max(target_q_np, 1)
                return η * self.ε_dual + np.mean(max_q) \
                    + η * np.mean(np.log(np.sum(
                        b_prob_np * np.exp((target_q_np - max_q[:, None]) / η), axis=1)))
    
    
            bounds = [(1e-6, None)]
            #Minimize the dual function using the scipy minimize function
            res = minimize(dual, np.array([self.η]), method='SLSQP', bounds=bounds)
            self.η = res.x[0]
            
            # Compute action weights (new q values) using dual variable η (see 3rd eq on page 4)
            qij = torch.softmax(target_q / self.η, dim=0) # (da, K)

            # 3. M step. Policy Improvement (4.2 in paper)
            #Fitting an improved policy using the sampled q-values via gradient optimization on the Policy and the lagrangian function 
            for _ in range(self.mstep_iteration_num):
                
                #action output of the current parametric policy
                π_p = self.actor.forward(states)  # (K, da)
                π = Categorical(probs=π_p)  # (K,)
                
                # Last eq of p.4 in the paper
                # MLE loss between the parametric policy and the sample based distribution qij without Kl regularization
                loss_MLE = torch.mean(
                    qij * π.expand((self.da, K)).log_prob(actions)
                )
                #Kl divergence between the current and target policy, used to regularize the MLE loss above 
                kl = self.categorical_kl(p1=π_p, p2=b_p)
                

                # Update lagrange multipliers by gradient descent (inner optimiation loop)
                # this equation is derived from last eq of p.5 in the paper,
                # just differentiate with respect to α
                # and update α so that the equation is to be minimized.
                self.η_kl -= self.α_scale * (self.ε_kl - kl).detach().item()

                # Clip the lagrange multipliers to positive values
                if self.η_kl < 0:
                    self.η_kl = 0.0
                    
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
        self.UpdateTargetNets(self.use_soft_updates)
                
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
    
    def UpdateTargetNets(self, soft_update: bool) -> None:
        """
        Updates the target network with the weights of the original one
        If soft_update is True, we perform a soft update via \tau \cdot \theta + (1 - \tau) \cdot \theta'
        """
        assert self.use_target_net == True
        with torch.no_grad():
            for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1 - self.tau) if soft_update #Soft update
                    else param.data #Hard update
                )
            for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1 - self.tau) if soft_update 
                    else param.data
                )

