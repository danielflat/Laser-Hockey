import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np

from torch.nn import init
from src.util.mathutil import int_to_one_hot
"""
Author: Andre Pfrommer

Intrinsic curiosity module after the paper 
"Curiosity-driven Exploration by Self-supervised Prediction" by Pathak et al. (2017):
https://arxiv.org/abs/1705.05363 
And the github implementation:
https://github.com/bonniesjli/icm
"""
device = torch.device("cpu")

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class ICMModel(nn.Module):
    """ICM model for non-vision based tasks"""
    def __init__(self, state_size: int, action_size: int):
        super(ICMModel, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.resnet_time = 4
        self.device = device
        
        # Net to encode state and next state as feature vectors
        self.feature = nn.Sequential(
            nn.Linear(self.state_size, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
        )
        # Inverse net to predict action given state and next state features
        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, self.action_size)
        )
        # Residual block in the Resnet structure
        self.residual = [nn.Sequential(
            nn.Linear(self.action_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
        ).to(self.device)] * 2 * self.resnet_time

        # First forward net
        self.forward_net_1 = nn.Sequential(
            nn.Linear(self.action_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256)
        )
        # Second forward net in the Resnet structure
        self.forward_net_2 = nn.Sequential(
            nn.Linear(self.action_size + 256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256),
            Swish(),
            nn.Linear(256, 256)
        )

    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor):
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
            
        encode_state = self.feature(state) #(B, 256)
        encode_next_state = self.feature(next_state) #(B, 256)
        
        # get predicted action via the inverse net
        pred_action = torch.cat((encode_state, encode_next_state), 1)
        pred_action = self.inverse_net(pred_action) #(B, da)

        # get pred next state via the forward net
        pred_next_state_feature_orig = torch.cat((encode_state, action), 1)
        pred_next_state_feature_orig = self.forward_net_1(pred_next_state_feature_orig) #(B, 256)

        # residual
        for i in range(self.resnet_time):
            pred_next_state_feature = self.residual[i * 2](torch.cat((pred_next_state_feature_orig, action), 1))
            pred_next_state_feature_orig = self.residual[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), 1)) + pred_next_state_feature_orig

        pred_next_state_feature = self.forward_net_2(torch.cat((pred_next_state_feature_orig, action), 1)) #(B, 256)

        real_next_state_feature = encode_next_state #(B, 256)
        return real_next_state_feature, pred_next_state_feature, pred_action
    
class ICM():
    """Intrinsic Curisity Module"""
    def __init__(
        self, 
        state_size: int,
        action_size: int,
        device = device,
        eta = 0.01):
        
        self.model = ICMModel(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 1e-4)
        
        self.state_size = state_size
        self.action_size = action_size
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.eta = eta
        self.device = device
    
    def compute_intrinsic_reward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute intrinsic rewards for parallel transitions.
        
        :param state: (B, ds) the batch of states
        :param next_state: (B, ds) the batch of next states
        :param action: (B, da) the batch of possible actions
        :return: (B, ) the intrinsic rewards
        """
        state = torch.FloatTensor(state)
        next_state = torch.FloatTensor(next_state)
        
        # Forward pass through the model
        real_next_state_feature, pred_next_state_feature, _ = self.model(
            state, next_state, action
        )
        
        # Compute intrinsic reward as the squared error between real and predicted feature version of the next state
        intrinsic_reward = (
            self.eta / 2 * (real_next_state_feature - pred_next_state_feature).pow(2).sum(dim=1)
        )
        
        intrinsic_reward = intrinsic_reward.clone().detach().to(device, dtype=torch.float32)
        return intrinsic_reward
    
    def train(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor) -> None:
        """
        Train the ICM model
        :param states: (B, ds) the batch of states
        :param next_states: (B, ds) the batch of next states
        :param actions: (B, da) the batch of actions
        """
        real_next_state_feature, pred_next_state_feature, pred_action = self.model(
            states, next_states, actions)
        
        # MSE Loss for the inverse model
        inverse_loss = self.MSE_loss(
            pred_action, actions)
        
        # MSE Loss for the forward model
        forward_loss = self.MSE_loss(
            pred_next_state_feature, real_next_state_feature.detach())
        
        loss = inverse_loss + forward_loss
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()