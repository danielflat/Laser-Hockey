import os
from typing import List

import numpy as np
import torch
from torch import device, nn

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.util.directoryutil import get_path


class QFunction(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: List[int], action_size: int):
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
        # Action and State concatenated as input into the q network
        concat_input = torch.cat((state, action), dim = 1)
        return self.network(concat_input)


class PolicyFunction(nn.Module):
    def __init__(self, state_size: int, hidden_sizes: List[int], action_size: int):
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
        Calculates the Actor values over all actions for the given state
        """
        action = self.network(state)
        return action

    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Calculates the detached Actor Value as a numpy array 
        """
        with torch.no_grad():
            action = self.forward(x).squeeze().cpu().numpy()
            action = np.expand_dims(action, axis = 0)
            return action


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
        action_size = self.get_num_actions(action_space)

        self.policy_delay = td3_settings["POLICY_DELAY"]
        self.noise_clip = td3_settings["NOISE_CLIP"]

        # Here we have 2 Q Networks
        self.Q1 = QFunction(state_size = state_size,
                            hidden_sizes = [128, 128],
                            action_size = action_size).to(device)
        self.Q2 = QFunction(state_size = state_size,
                            hidden_sizes = [128, 128],
                            action_size = action_size).to(device)
        self.policy = PolicyFunction(state_size = state_size,
                                     hidden_sizes = [128, 128, 64],
                                     action_size = action_size).to(device)
        self.targetQ1 = QFunction(state_size = state_size,
                                  hidden_sizes = [128, 128],
                                  action_size = action_size).to(device)
        self.targetQ2 = QFunction(state_size = state_size,
                                  hidden_sizes = [128, 128],
                                  action_size = action_size).to(device)
        self.policy_target = PolicyFunction(state_size = state_size,
                                            hidden_sizes = [128, 128, 64],
                                            action_size = action_size).to(device)
        # Set the target nets in eval
        self.targetQ1.eval()
        self.targetQ2.eval()
        self.policy_target.eval()

        # Copying the weights of the Q and Policy networks to the target networks
        self._copy_nets(soft_update = False)

        # Initializing the optimizers, TO DO: Use different learning rates for Q and Policy networks
        self.optimizer_q = self.initOptim(optim = td3_settings["OPTIMIZER"],
                                          parameters = list(self.Q1.parameters()) + list(self.Q2.parameters()))
        self.optimizer_policy = self.initOptim(optim = td3_settings["OPTIMIZER"],
                                               parameters = self.policy.parameters())

        # Define Loss function
        self.criterion = self.initLossFunction(loss_name = td3_settings["LOSS_FUNCTION"])

    def __repr__(self):
        """
        For printing purposes only
        """
        return f"TD3Agent"

    def _copy_nets(self, soft_update: bool):
        self.updateTargetNet(soft_update = soft_update, source = self.Q1, target = self.targetQ1)
        self.updateTargetNet(soft_update = soft_update, source = self.Q2, target = self.targetQ2)
        self.updateTargetNet(soft_update = soft_update, source = self.policy, target = self.policy_target)

    def act(self, state: torch.Tensor) -> np.ndarray:
        """
        The Agent chooses an action.
        In Evaluation mode, we set the noise eps = 0
        In Training mode, we sample an action using actor network and exploration noise
        :param state: The state
        """
        if self.isEval:
            self.epsilon = 0

        with torch.no_grad():
            # action squeezed in -1 to 1 via tanh (+ noise)
            action_deterministic = self.policy_target.forward(state)
            action = action_deterministic + torch.randn_like(action_deterministic) * self.epsilon
            # rescale the action to the action space
            action = action * self.action_scale + self.action_bias
            action = torch.clamp(action, self.action_low, self.action_high)

        action = action.squeeze().cpu().numpy()

        action = np.expand_dims(action, axis = 0)
        return action

    def calc_td_target(self, reward: torch.Tensor, done: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculates the TD Target
        1. Sample the next action from the target policy network using gaussian exploration noise
        2. Calculate the Q values for the next state and next action using both target Q networks
        3. Calculate the TD Target by bootstrapping the minimum of the two Q networks
        """

        with torch.no_grad():
            # Exploration noise
            noise = torch.clamp(
                torch.randn(self.action_space.shape[0], device = self.device) * self.epsilon,
                min = -self.noise_clip,
                max = self.noise_clip)

            # 1. Next action via the target policy network
            next_action = self.policy_target.forward(next_state) * self.action_scale + self.action_bias
            next_action = torch.clamp(
                next_action + noise,
                min = self.action_low,  # Minimum action value
                max = self.action_high)  # Maximum action value

            # 2. Forward pass for both Q networks
            q_prime1 = self.targetQ1.forward(next_state, next_action)
            q_prime2 = self.targetQ2.forward(next_state, next_action)

            # 3. Bootstrapping the minimum of the two Q networks
            td_target = reward + (1 - done) * self.discount * torch.min(q_prime1, q_prime2)

        return td_target

    def optimize(self, memory: ReplayMemory, episode_i: int) -> List[float]:
        """
        Compute forward and backward pass for the Q and Policy networks
        """
        assert self.isEval == False
        # Storing losses in a list for logging as we run several optimization steps
        losses = []
        # We start at i=1 to prevent a direct update of the weights
        for i in range(1, self.opt_iter + 1):
            # Sample from the replay memory
            state, action, reward, next_state, done, info = memory.sample(self.batch_size, randomly = True)

            # Forward pass for Q networks
            td_target = self.calc_td_target(reward, done, next_state)
            q1_loss = self.criterion(self.Q1.forward(state, action), td_target)
            q2_loss = self.criterion(self.Q2.forward(state, action), td_target)
            q_loss = q1_loss + q2_loss

            # Backward step for Q networks
            self.optimizer_q.zero_grad()
            q_loss.backward()
            if self.use_gradient_clipping:
                torch.nn.utils.clip_grad_value_(parameters = list(self.Q1.parameters()) + list(self.Q2.parameters()),
                                                clip_value = self.gradient_clipping_value,
                                                foreach = self.use_clip_foreach)
            self.optimizer_q.step()

            # Get the target for Policy network
            q_1 = self.Q1.forward(state, self.policy.forward(state))
            policy_loss = -torch.mean(q_1)

            # Backward step for Policy network
            if i % self.policy_delay == 0:
                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                if self.use_gradient_clipping:
                    torch.nn.utils.clip_grad_value_(parameters = self.policy.parameters(),
                                                    clip_value = self.gradient_clipping_value,
                                                    foreach = self.use_clip_foreach)
                self.optimizer_policy.step()

            # Logging the losses
            losses.append([q_loss.item(), policy_loss.item()])

        # after each optimization, decay epsilon
        self.adjust_epsilon(episode_i)

        # after each optimization, update target network
        self._copy_nets(soft_update = self.use_soft_updates)

        return losses

    def setMode(self, eval = False) -> None:
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
        checkpoint = {
            "actor": self.policy.state_dict(),
            "critic1": self.Q1.state_dict(),
            "critic2": self.Q2.state_dict(),
            "critic1_target": self.targetQ1.state_dict(),
            "critic2_target": self.targetQ2.state_dict(),
            "iteration": iteration
        }

        directory = get_path(f"output/checkpoints/{model_name}")
        file_path = os.path.join(directory, f"{model_name}_{iteration:05}.pth")
        os.makedirs(directory, exist_ok = True)
        torch.save(checkpoint, file_path)
        print(f"Actor and Critic weights saved successfully!")

    def loadModel(self, file_name: str) -> None:
        try:
            checkpoint = torch.load(file_name, map_location = self.device)
            self.policy.load_state_dict(checkpoint["actor"])
            self.Q1.load_state_dict(checkpoint["critic1"])
            self.Q2.load_state_dict(checkpoint["critic2"])
            self.targetQ1.load_state_dict(checkpoint["critic1_target"])
            self.targetQ2.load_state_dict(checkpoint["critic2_target"])
            print(f"Model loaded successfully from {file_name}")
        except FileNotFoundError:
            print(f"Error: File {file_name} not found.")
        except Exception as e:
            print(f"An error occurred while loading the model: {str(e)}")

    def updateTargetNets(self, soft_update: bool) -> None:
        """
        Updates the target network with the weights of the original one
        If soft_update is True, we perform a soft update via \tau \cdot \theta + (1 - \tau) \cdot \theta'
        """
        assert self.use_target_net == True
        with torch.no_grad():
            for target_param, param in zip(self.targetQ1.parameters(), self.Q1.parameters()):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1 - self.tau) if soft_update  # Soft update
                    else param.data  # Hard update
                )
            for target_param, param in zip(self.targetQ2.parameters(), self.Q2.parameters()):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1 - self.tau) if soft_update
                    else param.data
                )
            for target_param, param in zip(self.policy_target.parameters(), self.policy.parameters()):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1 - self.tau) if soft_update
                    else param.data
                )

    def import_checkpoint(self, checkpoint: dict) -> None:
        raise NotImplementedError

    def export_checkpoint(self) -> dict:
        raise NotImplementedError
