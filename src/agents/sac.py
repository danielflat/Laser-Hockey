import os
from typing import List
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.agent import Agent
# from src.replaymemory import ReplayMemory  # Not strictly needed if we place everything in one file
from src.util.constants import MSE_LOSS
from src.util.directoryutil import get_path

################################################################################
# Helper Networks (Actor / Critic)
################################################################################

class Actor(nn.Module):
    def __init__(self, state_size, action_space, action_size: int,
                 hidden_dim: int = 256, device: torch.device = None):
        super().__init__()
        self.device = device if device else torch.device("cpu")

        # Some environment's action_space might be a Box(...) with .low and .high
        self.action_low = torch.tensor(action_space.low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_space.high, dtype=torch.float32, device=self.device)
        # Scale & bias for linear mapping from [-1,1] to [low, high]
        self.action_scale = ((self.action_high - self.action_low) / 2.0)[:action_size]
        self.action_bias = ((self.action_high + self.action_low) / 2.0)[:action_size]

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_size)
        self.log_std_head = nn.Linear(hidden_dim, action_size)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        base_out = self.net(x)
        mean = self.mean_head(base_out)
        log_std = self.log_std_head(base_out)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        Sample an action with reparameterization, 
        apply tanh-squash, scale to [low, high], 
        and compute log_prob of that transformed action.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()

        # Reparam trick
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # shape: [batch, act_dim]
        log_prob = normal.log_prob(x_t).sum(dim=-1, keepdim=True)

        # Tanh-squash
        y_t = torch.tanh(x_t)
        action = self.action_scale * y_t + self.action_bias

        # Tanh correction
        log_prob -= (2.0 * (np.log(2.0) - x_t - F.softplus(-2.0 * x_t))).sum(dim=-1, keepdim=True)
        # Scale correction
        log_prob -= torch.log(self.action_scale).sum(dim=-1, keepdim=True)

        mean_action = self.action_scale * torch.tanh(mean) + self.action_bias
        return action, log_prob, mean_action

class Critic(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
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
        The SAC agent.
        """
        super().__init__(agent_settings, device)

        learn_alpha   = sac_settings["LEARN_ALPHA"]
        target_entropy= sac_settings["TARGET_ENTROPY"]
        init_alpha    = sac_settings["INIT_ALPHA"]
        hidden_dim    = sac_settings["HIDDEN_DIM"]

        # Dimensions
        state_size  = state_space.shape[0]
        action_size = self.get_num_actions(action_space)

        # Actor + Critics
        self.actor    = Actor(state_size, action_space, action_size, hidden_dim, device=self.device).to(self.device)
        self.critic1  = Critic(state_size, action_size, hidden_dim).to(self.device)
        self.critic2  = Critic(state_size, action_size, hidden_dim).to(self.device)
        self.critic1_target = Critic(state_size, action_size, hidden_dim).to(self.device)
        self.critic2_target = Critic(state_size, action_size, hidden_dim).to(self.device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # Temperature
        self.learn_alpha = learn_alpha
        self.log_alpha   = torch.tensor(np.log(init_alpha), dtype=torch.float32, requires_grad=True, device=self.device)
        self.alpha       = self.log_alpha.exp().item()
        self.target_entropy = target_entropy if target_entropy is not None else -action_size

        # Opt configs
        actor_optim_cfg  = sac_settings["OPTIMIZER"]
        critic_optim_cfg = sac_settings["OPTIMIZER"]
        alpha_optim_cfg  = sac_settings["OPTIMIZER"]

        self.actor_optimizer   = self.initOptim(actor_optim_cfg,   self.actor.parameters(),   disable_weight_decay=True)
        self.critic1_optimizer= self.initOptim(critic_optim_cfg,  self.critic1.parameters(), disable_weight_decay=True)
        self.critic2_optimizer= self.initOptim(critic_optim_cfg,  self.critic2.parameters(), disable_weight_decay=True)
        if self.learn_alpha:
            self.alpha_optimizer= self.initOptim(alpha_optim_cfg, [self.log_alpha], disable_weight_decay=True)

        # Loss fn
        self.critic_loss_fn = self.initLossFunction(agent_settings.get("CRITIC_LOSS", MSE_LOSS))

    def __repr__(self):
        return "SACAgent"

    def act(self, state_np: np.ndarray) -> np.ndarray:
        """
        Expects a single observation (shape [obs_dim]) as a NumPy array.
        Returns an action in the environment's expected shape (also NumPy).
        """
        self.actor.eval()
        with torch.no_grad():
            # Convert to torch
            x_t = torch.as_tensor(state_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            action_t, _, _ = self.actor.sample(x_t)
            # shape of action_t: (1, act_dim)
        action_np = action_t.squeeze(0).cpu().numpy()  # shape (act_dim,)
        return action_np

    def optimize(self, memory, episode_i: int) -> List[float]:
        """
        One or more SAC update steps.
        """
        self.adjust_epsilon(episode_i)

        losses = []
        for _ in range(self.opt_iter):
            # Sample from replay (already returns torch Tensors on self.device)
            states, actions, rewards, next_states, dones, _ = memory.sample(self.batch_size, randomly=True)

            with torch.no_grad():
                # Next actions and next log probs
                next_act, next_log_probs, _ = self.actor.sample(next_states)
                q_next_1 = self.critic1_target(next_states, next_act)
                q_next_2 = self.critic2_target(next_states, next_act)
                q_next = torch.min(q_next_1, q_next_2) - self.log_alpha.exp() * next_log_probs
                q_target= rewards + self.discount * (1 - dones) * q_next

            # Critic 1 & 2 loss
            q1_pred = self.critic1(states, actions)
            q2_pred = self.critic2(states, actions)
            critic1_loss = self.critic_loss_fn(q1_pred, q_target)
            critic2_loss = self.critic_loss_fn(q2_pred, q_target)

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

            # Actor loss
            new_act, log_prob, _ = self.actor.sample(states)
            q1_val = self.critic1(states, new_act)
            q2_val = self.critic2(states, new_act)
            q_val = torch.min(q1_val, q2_val)
            actor_loss = (self.log_alpha.exp().detach() * log_prob - q_val).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.use_gradient_clipping:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.gradient_clipping_value)
            self.actor_optimizer.step()

            # Alpha loss
            if self.learn_alpha:
                alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                if self.use_gradient_clipping:
                    nn.utils.clip_grad_norm_([self.log_alpha], self.gradient_clipping_value)
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()
            else:
                alpha_loss = torch.tensor(0.0)

            # Soft update targets (if self.use_soft_updates)
            if self.use_soft_updates:
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
                if episode_i % self.target_net_update_freq == 0:
                    self.critic1_target.load_state_dict(self.critic1.state_dict())
                    self.critic2_target.load_state_dict(self.critic2.state_dict())

            total_critic_loss = 0.5 * (critic1_loss.item() + critic2_loss.item())
            losses.append([total_critic_loss, actor_loss.item(), alpha_loss.item()])

        avg_losses = np.mean(losses, axis=0).tolist()
        return avg_losses

    def setMode(self, eval: bool = False) -> None:
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
        self.alpha = self.log_alpha.exp().item()
        print(f"Model loaded from {file_name}")

    def import_checkpoint(self, checkpoint: dict) -> None:
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
        self.alpha = self.log_alpha.exp().item()
        print("Checkpoint imported successfully.")

    def export_checkpoint(self) -> dict:
        checkpoint = {
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "critic1_target": self.critic1_target.state_dict(),
            "critic2_target": self.critic2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu().numpy(),
        }
        return checkpoint

    def reset(self):
        raise NotImplementedError
