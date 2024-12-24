# import numpy as np
#
# from src.agent import Agent
#
# import torch
# from torch import device, nn
# import torch.nn.functional as F
#
# class Actor(nn.Module):
#     def __init__(self, input_shape: int, output_shape: int):
#         super().__init__()
#         self.f1 = nn.Linear(input_shape, 128)
#         self.a1 = nn.LeakyReLU()
#         self.f2 = nn.Linear(128, 128)
#         self.a2 = nn.LeakyReLU()
#         self.f3 = nn.Linear(128, output_shape)
#
#     def forward(self, x: torch.Tensor):
#         x = self.f1(x)
#         x = self.a1(x)
#         x = self.f2(x)
#         x = self.a2(x)
#         x = self.f3(x)
#         return x
#
#
# class Critic(nn.Module):
#     def __init__(self, input_shape: int):
#         super().__init__()
#         self.f1 = nn.Linear(input_shape, 128)
#         self.a1 = nn.LeakyReLU()
#         self.f2 = nn.Linear(128, 128)
#         self.a2 = nn.LeakyReLU()
#         self.f3 = nn.Linear(128, 1)
#
#     def forward(self, x: torch.Tensor):
#         x = self.f1(x)
#         x = self.a1(x)
#         x = self.f2(x)
#         x = self.a2(x)
#         x = self.f3(x)
#         return x
#
# class DDPGAgent(Agent):
#     def __init__(self, state_shape: int, action_size: int, options: dict, optim: dict, hyperparams: dict,
#                  device: device):
#         super().__init__(hyperparams, options)
#
#         self.isEval = None
#
#         self.state_shape = state_shape
#         self.action_size = action_size
#
#         # Hyperparams
#         self.opt_iter = hyperparams["OPT_ITER"]
#         self.batch_size = hyperparams["BATCH_SIZE"]
#         self.discount = hyperparams["DISCOUNT"]
#         self.epsilon = hyperparams["EPSILON"]
#         self.epsilon_start = hyperparams["EPSILON"]
#         self.epsilon_min = hyperparams["EPSILON_MIN"]
#         self.epsilon_decay = hyperparams["EPSILON_DECAY"]
#         self.gradient_clipping_value = hyperparams["GRADIENT_CLIPPING_VALUE"]
#         self.target_net_update_freq = hyperparams["TARGET_NET_UPDATE_FREQ"]
#         self.tau = hyperparams["TAU"]
#
#         # Options
#         self.use_target_net = options["USE_TARGET_NET"]
#         self.use_soft_updates = options["USE_SOFT_UPDATES"]
#         self.use_gradient_clipping = options["USE_GRADIENT_CLIPPING"]
#         self.epsilon_decay_strategy = options["EPSILON_DECAY_STRATEGY"]
#         self.device: device = device
#         self.use_clip_foreach = options["USE_CLIP_FOREACH"]
#         self.USE_BF_16 = options["USE_BF16"]
#
#         # Define the Q-Network
#         self.Q = QFunction(state_size=state_shape[0],
#                            hidden_size=128,
#                            action_size=action_size)
#         self.Q.to(self.device)
#
#         # If you want to use a target network, it is defined here
#         if self.use_target_net:
#             self.targetQ = QFunction(state_size=state_shape[0],
#                                      hidden_size=128,
#                                      action_size=action_size)
#             self.targetQ.to(self.device)
#             self.targetQ.eval()  # Set it always to Eval mode
#             self.updateTargetNet(soft_update=False)  # Copy the Q network
#
#         # Define the Optimizer
#         self.optimizer = self.initOptim(optim=optim, parameters=self.Q.parameters())
#
#         # Define Loss function
#         self.criterion = self.initLossFunction(loss_name=options["LOSS_FUNCTION"])
#
#     def act(self, state: torch.Tensor) -> float:
#         """
#         The Agent chooses an action.
#         In Evaluation mode, we always exploit the best action.
#         In Training mode, we sample an action based on epsilon greedy with the given epsilon hyperparam.
#         :param state: The state
#         :return: The action TODO: Yet only as discrete variable available
#         """
#
#         # In evaluation mode, we always exploit
#         if self.isEval:
#             return self.pol.greedyAction(state.unsqueeze(0)).item()
#
#         # In training mode, use epsilon greedy action sampling
#         rdn = np.random.random()
#         if rdn <= self.epsilon:
#             # Exploration
#             return np.random.random(low=0, high=self.action_size)
#         else:
#             # Exploitation
#             return self.Q.greedyAction(state.unsqueeze(0)).item()
#
