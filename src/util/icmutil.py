import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn import init
from src.util.mathutil import int_to_one_hot

# Activation Function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ICM(nn.Module):
    """
    Author: Andre Pfrommer

    Intrinsic curiosity module after the paper 
    "Curiosity-driven Exploration by Self-supervised Prediction" by Pathak et al. (2017):
    https://arxiv.org/abs/1705.05363 
    And the github implementation:
    https://github.com/bonniesjli/icm
    """
    def __init__(self, state_size: int, action_size: int, discrete: bool, device: torch.device, eta: float = 0.2):
        
        super(ICM, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.resnet_depth = 4
        self.eta = eta
        self.discrete = discrete
        self.device = device
        
        # Feature encoder network
        self.feature = nn.Sequential(
            nn.Linear(state_size, 256), Swish(),
            nn.Linear(256, 256), Swish(),
            nn.Linear(256, 256)
        )
        
        # Inverse model that predicts the action taken
        self.inverse_net = nn.Sequential(
            nn.Linear(256 * 2, 256), Swish(),
            nn.Linear(256, 256), Swish(),
            nn.Linear(256, action_size)
        )
        
        # Residual blocks in the Resnet structure
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_size + 256, 256), Swish(),
                nn.Linear(256, 256)
            ) for _ in range(self.resnet_depth * 2)
        ])
        
        # Both forward models
        self.forward_net_1 = self._build_forward_net()
        self.forward_net_2 = self._build_forward_net()
        
        # Initialize Optimizer and loss functions
        self.optimizer = optim.Adam(self.parameters(), lr=1e-4)
        self.CE_loss = nn.CrossEntropyLoss() 
        self.MSE_loss = nn.MSELoss()
    
    def _build_forward_net(self):
        return nn.Sequential(
            nn.Linear(self.action_size + 256, 256), Swish(),
            nn.Linear(256, 256), Swish(),
            nn.Linear(256, 256), Swish(),
            nn.Linear(256, 256), Swish(),
            nn.Linear(256, 256)
        )
    
    def forward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Forward pass through the ICM module.
            1. Get the feature representations of the state and next state
            2. Predict the action taken using the inverse model
            3. Predict the feature representation of the next state using the forward model
        
        :param state: Current state (B, ds)
        :param next_state: Next state (B, ds)
        :param action: Action taken (B, da)
        :return: Encoded next state, predicted next state feature, predicted action
        """
        state, next_state, action = self._ensure_batch(state, next_state, action)
        
        state = state.to(self.device)
        next_state = next_state.to(self.device)
        action = action.to(self.device)
        
        # 1. Encoding state and next state
        encode_state = self.feature(state)
        encode_next_state = self.feature(next_state)
        
        # 2. Predicting the action 
        pred_action = self.inverse_net(torch.cat((encode_state, encode_next_state), dim=1))
        
        # 3. Predicting the next state encoding 
        pred_next_state_feature = self._compute_forward_model(encode_state, action)
        
        # Return the true next state encoding, the predicted next state encoding 
        # and the predicted action to compute the losses of the nets
        return encode_next_state, pred_next_state_feature, pred_action
    
    def _ensure_batch(self, *tensors):
        return [t.unsqueeze(0) if len(t.shape) == 1 else t for t in tensors]

    def _compute_forward_model(self, encode_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward model that predicts the feature representation of the next state
        We do this using a residual network structure.
        
        :param encode_state: (B, 256) the feature representation of the current state
        :param action: (B, da) the action taken
        :return: (B, 256) the predicted next state in feature space
        """
        # First we get the feature representation of the state and the action
        pred_next_state_feature = self.forward_net_1(torch.cat((encode_state, action), dim=1))
        
        # Then we pass it through the residual blocks
        for i in range(self.resnet_depth):
            residual_input = torch.cat((pred_next_state_feature, action), dim=1)
            pred_next_state_feature = self.residual_blocks[i * 2](residual_input)
            pred_next_state_feature = self.residual_blocks[i * 2 + 1](
                torch.cat((pred_next_state_feature, action), dim=1)
            ) + pred_next_state_feature
        
        # Finally we pass the output through the second forward net
        encoded_next_state = self.forward_net_2(torch.cat((pred_next_state_feature, action), dim=1))
        return encoded_next_state

    def compute_intrinsic_reward(self, state: torch.Tensor, next_state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Compute the curiosity reward.
        For this, we first get the state and next state encodings via the forward model.
        Then, we compute the squared error between the real and predicted next state encodings.
        This gives us an estimation of how well the model can predict the next state in the feature space.
            - If the loss is large, the model is uncertain about the next state and the agent is curious.
            - If the loss is small, the model is certain about the next state and the agent should take conservative actions.
        
        :param state: (B, ds) the batch of states
        :param next_state: (B, ds) the batch of next states
        :param action: (B, da) the batch of possible actions
        :return: (B, ) the intrinsic rewards
        """
        if self.discrete:
            action = action.clone().detach().long().to(self.device)
            action_onehot = int_to_one_hot(action, self.action_size)
        else:
            action_onehot = action
        
        # Forward pass through the model
        real_next_state_feature, pred_next_state_feature, _ = self.forward(
            state, next_state, action_onehot
        )
        
        # Compute intrinsic reward as the squared error between real and predicted feature version of the next state
        intrinsic_reward = (
            self.eta / 2 * (real_next_state_feature - pred_next_state_feature).pow(2).sum(dim=1)
        )
        
        intrinsic_reward = intrinsic_reward.clone().detach().to(self.device, dtype=torch.float32)
        return intrinsic_reward

    def train(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor) -> None:
        """
        Train the ICM model.
        Here we train both the inverse and forward models. 
        
        :param states: (B, ds) the batch of states
        :param next_states: (B, ds) the batch of next states
        :param actions: (B, da) the batch of actions
        """
        if self.discrete:
            actions = actions.clone().detach().long().to(self.device)
            #Onehot encoding
            action_onehot = torch.FloatTensor(len(actions), self.action_size).to(self.device)
            action_onehot.zero_()
            action_onehot.scatter_(1, actions.view(-1, 1), 1)
        else:
            action_onehot = actions
        
        real_next_state_feature, pred_next_state_feature, pred_action = self.forward(
            states, next_states, action_onehot)
        
        # Get the inverse loss as the CE loss between the predicted and real actions
        if self.discrete:
            inverse_loss = self.CE_loss(
                pred_action, action_onehot)
        else:
            inverse_loss = self.MSE_loss(
                pred_action, action_onehot)
            
        # Get the forward loss as the MSE loss between the predicted and real next state features
        forward_loss = self.MSE_loss(
            pred_next_state_feature, real_next_state_feature.detach())
        
        # Optimize both losses
        loss = inverse_loss + forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()