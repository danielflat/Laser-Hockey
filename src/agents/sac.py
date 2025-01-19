import os
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.agent import Agent  # Adjust import to match your project structure
from src.replaymemory import ReplayMemory
from src.util.constants import MSE_LOSS
from src.util.directoryutil import get_path


################################################################################
# Helper Networks
################################################################################

class Actor(nn.Module):
    """
    Policy network for SAC that outputs actions inside the given Box range.
    
    Assumes:
      - `action_dim` is a gym.spaces.Box object, e.g. Box(-2.0, 2.0, (1,), float32)
      - The shape of the action space can be multi-dimensional (e.g. shape=(2,)).
      - The range can be symmetric or asymmetric, but we do the basic linear rescaling 
        from [-1,1] to [low, high].
    """

    def __init__(self, state_size, action_space, action_size: int, hidden_dim: int = 256, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        
        # Extract info from the Box space
        # e.g. if action_dim = Box(-2, 2, (1,), float32)
        # then action_dim.low might be array([-2.]), action_dim.high might be array([ 2.])
        # shape = (1, )
        self.action_low = torch.tensor(action_space.low, dtype = torch.float32, device = self.device)
        self.action_high = torch.tensor(action_space.high, dtype = torch.float32, device = self.device)
        # Scale & bias for linear mapping from [-1,1] to [low, high]
        self.action_scale = ((self.action_high - self.action_low) / 2.0)[:action_size]
        self.action_bias = ((self.action_high + self.action_low) / 2.0)[:action_size]

        # Simple feedforward network
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # Output heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_size)
        self.log_std_head = nn.Linear(hidden_dim, action_size)
        
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Returns the mean and log_std of a Gaussian distribution over the raw (pre-tanh) actions.
        """
        base_out = self.net(x)
        mean = self.mean_head(base_out)
        log_std = self.log_std_head(base_out)
        # Clamp log_std for numerical stability
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def sample(self, state: torch.Tensor):
        """
        Sample an action using the reparameterization trick, transform it 
        to the valid Box range, and compute the log probability of that action.
        
        Returns:
          - action in [low, high]
          - log_prob of that action
          - (optionally) the 'mean' action (also scaled to the Box range) for debugging
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # shape: [batch_size, act_size]
        
        # Compute log prob BEFORE transformation
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)
        
        # ---- Tanh and scale to [low, high] range ----
        # Tanh-squash
        y_t = torch.tanh(x_t)
        
        # Scale + bias
        # shape of action_scale & action_bias: same as action_dim, e.g. (1,) or (N,)
        action = self.action_scale * y_t + self.action_bias
        
        # ---- Tanh correction to the log-prob ----
        # derivative of tanh(x) = 1 - tanh^2(x)
        # but we often use a stable formula:
        #     log(1 - y_t^2) = log(1 - tanh(x_t)^2).
        # or the standard approach for removing the logdet of the Jacobian:
        #     log_prob -= sum(2*(np.log(2) - x_t - softplus(-2x_t)))
        # plus we need to account for linear scaling by action_scale:
        
        # 1) Standard Tanh Correction:
        # log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2*x_t))).sum(dim=-1, keepdim=True)
        # 2) Correction for the linear scale:
        # For a transform z -> scale * z, the logdet factor is sum(log(scale)).
        # We subtract that from the log-prob.
        
        log_prob -= (2.0 * (np.log(2.0) - x_t - F.softplus(-2.0 * x_t))).sum(dim=-1, keepdim=True)
        # Subtract log(scale) for each dimension:
        # If self.action_scale is multi-dimensional, take the product or sum of logs:
        log_prob -= torch.log(self.action_scale).sum(dim=-1, keepdim=True)
        
        # If you'd like to store or return a "mean action" for debugging:
        # scale + bias for the mean as well
        mean_action = self.action_scale * torch.tanh(mean) + self.action_bias
        
        return action, log_prob, mean_action

class Critic(nn.Module):
    """
    Critic network (Q function) for SAC.
    Maps (state, action) -> Q-value.
    """

    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Returns the predicted Q-value for the given (state, action).
        """
        return self.net(torch.cat([state, action], dim=1))


################################################################################
# Soft Actor-Critic Agent
################################################################################

class SoftActorCritic(Agent):
    def __init__(
        self,
        agent_settings: dict,
        device: torch.device,
            state_space,
            action_space,
        sac_settings: dict
    ):
        """
        The SAC agent. Inherits from the base `Agent` class and implements
        the required abstract methods. 
        """

        learn_alpha = sac_settings["LEARN_ALPHA"]
        target_entropy = sac_settings["TARGET_ENTROPY"]
        init_alpha = sac_settings["INIT_ALPHA"]
        hidden_dim = sac_settings["HIDDEN_DIM"]

        super().__init__(agent_settings, device)

        # Number of dimensions
        state_size = state_space.shape[0]
        action_size = self.get_num_actions(action_space)

        # Set up networks
        self.actor = Actor(state_size, action_space, action_size, hidden_dim).to(device)
        self.critic1 = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic2 = Critic(state_size, action_size, hidden_dim).to(device)

        # Target critics for stable Q-target estimation
        self.critic1_target = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic2_target = Critic(state_size, action_size, hidden_dim).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Whether to learn temperature alpha or keep it fixed
        self.learn_alpha = learn_alpha
        self.log_alpha = torch.tensor(np.log(init_alpha), dtype=torch.float32, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp().detach().item()  # for logging
        self.target_entropy = target_entropy if target_entropy is not None else -action_space.shape[0]

        # Optimization hyperparameters
        # Example: we can keep separate configs for actor, critic, and alpha
        # or you can use your single config with "initOptim()" from the base.
        actor_optim_cfg = sac_settings["OPTIMIZER"]
        critic_optim_cfg = sac_settings["OPTIMIZER"]
        alpha_optim_cfg = sac_settings["OPTIMIZER"]

        # Initialize optimizers
        self.actor_optimizer = self.initOptim(actor_optim_cfg, self.actor.parameters(), disable_weight_decay=True)
        self.critic1_optimizer = self.initOptim(critic_optim_cfg, self.critic1.parameters(), disable_weight_decay=True)
        self.critic2_optimizer = self.initOptim(critic_optim_cfg, self.critic2.parameters(), disable_weight_decay=True)
        if self.learn_alpha:
            self.alpha_optimizer = self.initOptim(alpha_optim_cfg, [self.log_alpha], disable_weight_decay=True)

        # Define a loss function (MSE for Q-updates)
        # You can also choose SmoothL1 if you prefer.
        self.critic_loss_fn = self.initLossFunction(agent_settings.get("CRITIC_LOSS", MSE_LOSS))

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"SACAgent"

    def act(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns an *action index* or an *action vector* depending on your environment.
        For continuous action environments, we'll just return a float vector in [-1, 1].
        """
        self.actor.eval()
        with torch.no_grad():
            action, _, _ = self.actor.sample(x.to(self.device))
        # If your environment expects a NumPy array, detach and move to CPU:
        action = action.squeeze().cpu().numpy()

        return action

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        Perform one or more optimization steps of the SAC algorithm:
          1. Sample a batch from replay memory
          2. Compute targets for Q-values
          3. Update Q networks
          4. Update policy network
          5. Update temperature alpha (if learn_alpha == True)
          6. Update target networks (soft update)
        Returns a list of losses for logging.
        """

        # Adjust epsilon if needed (though SAC typically doesn't use epsilon exploration)
        self.adjust_epsilon(episode_i)

        # We might run several gradient updates each time
        losses = []
        for _ in range(self.opt_iter):
            # 1. Sample from replay
            states, actions, rewards, next_states, dones, _ = memory.sample(self.batch_size, randomly=True)

            # 2. Compute next actions and next log probs using the current actor
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.actor.sample(next_states)
                # Evaluate next Q-values from target networks
                q_next_1 = self.critic1_target(next_states, next_actions)
                q_next_2 = self.critic2_target(next_states, next_actions)
                q_next = torch.min(q_next_1, q_next_2) - self.log_alpha.exp() * next_log_probs
                # Compute target
                q_target = rewards + self.discount * (1 - dones) * q_next

            # 3. Update Q networks (critic1 and critic2)
            q1_pred = self.critic1(states, actions)
            q2_pred = self.critic2(states, actions)

            critic1_loss = self.critic_loss_fn(q1_pred, q_target)
            critic2_loss = self.critic_loss_fn(q2_pred, q_target)
            # Zero grad & step
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.critic1.parameters(), self.gradient_clipping_value)
            self.critic1_optimizer.step()

            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.critic2.parameters(), self.gradient_clipping_value)
            self.critic2_optimizer.step()

            # 4. Update policy network
            # We re-sample the action for current states
            action_sample, log_prob_sample, _ = self.actor.sample(states)
            q1_val = self.critic1(states, action_sample)
            q2_val = self.critic2(states, action_sample)
            q_val = torch.min(q1_val, q2_val)
            actor_loss = (self.log_alpha.exp().detach() * log_prob_sample - q_val).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clipping_value)
            self.actor_optimizer.step()

            # 5. Update temperature alpha if learn_alpha == True
            if self.learn_alpha:
                alpha_loss = -(self.log_alpha.exp() * (log_prob_sample + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.use_gradient_clipping:
                    nn.utils.clip_grad_norm_([self.log_alpha], self.gradient_clipping_value)
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().detach().item()
            else:
                alpha_loss = torch.tensor(0.0)

            # 6. Update target networks
            if self.use_soft_updates:
                # soft update of target networks
                with torch.no_grad():
                    for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )
                    for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                        target_param.data.copy_(
                            target_param.data * (1.0 - self.tau) + param.data * self.tau
                        )
            else:
                # Hard update if needed after certain freq
                if episode_i % self.target_net_update_freq == 0:
                    self.critic1_target.load_state_dict(self.critic1.state_dict())
                    self.critic2_target.load_state_dict(self.critic2.state_dict())

            # Keep track of losses
            total_critic_loss = 0.5 * (critic1_loss.item() + critic2_loss.item())
            losses.append([total_critic_loss, actor_loss.item(), alpha_loss.item()])

        # Return average losses over the optimization steps
        avg_losses = np.mean(losses, axis=0).tolist()
        return avg_losses

    def setMode(self, eval: bool = False) -> None:
        """
        Set networks in train or eval mode.
        """
        if eval:
            self.actor.eval()
            self.critic1.eval()
            self.critic2.eval()
            self.critic1_target.eval()
            self.critic2_target.eval()
        else:
            self.actor.train()
            self.critic1.train()
            self.critic2.train()
            self.critic1_target.train()
            self.critic2_target.train()

    def saveModel(self, model_name: str, iteration: int) -> None:
        """
        Saves the model parameters of the agent.
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
            "iteration": iteration
        }

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")
        os.makedirs(directory, exist_ok = True)
        torch.save(checkpoint, file_path)
        print(f"Actor and Critic weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        """
        Loads the model parameters of the agent.
        """
        checkpoint = torch.load(file_name, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])
        self.log_alpha = torch.tensor(
            checkpoint["log_alpha"],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device
        )
        self.alpha = self.log_alpha.exp().detach().item()
        print(f"Model loaded from {file_name}")

    def import_checkpoint(self, checkpoint: dict) -> None:
        """
        Loads model parameters from a checkpoint dictionary.
        The dictionary should contain keys: 'actor', 'critic1', 'critic2', 
        'critic1_target', 'critic2_target', and 'log_alpha'.
        """
        # Load actor and critic network parameters
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target"])

        # Update log_alpha and the current alpha value
        self.log_alpha = torch.tensor(
            checkpoint["log_alpha"],
            dtype=torch.float32,
            requires_grad=True,
            device=self.device
        )
        self.alpha = self.log_alpha.exp().detach().item()

        print("Checkpoint imported successfully.")

    def export_checkpoint(self) -> dict:
        """
        Exports the current model parameters to a checkpoint dictionary.
        The returned dictionary contains the actor, critics, their target networks,
        log_alpha, and optionally other bookkeeping information.
        """
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
        }
        return checkpoint
