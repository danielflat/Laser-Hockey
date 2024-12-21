"""
This is the config file for the training run main.py.
"""

import torch

from src.util.constants import ADAM, DQN, EXPONENTIAL, PENDULUM, SMOOTHL1

CONFIG = {
    "SEED": 24,  # The seed that we want to use
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # On which machine is it running?
    "USE_TF32": True,  # Uses TF32 instead of Float32. Makes it faster, but you have lower precision
    "USE_ENV": PENDULUM,  # The used environment
    "USE_ALGO": DQN,  # The used algorithm
    "OPTIONS": {
        "SELF_TRAINING": False,  # If the agent should play against itself like in AlphaGo  # If a target net is used
        "USE_TARGET_NET": True,  # If a target net is used
        "USE_SOFT_UPDATES": False,  # If the target network is updated. True = softly, False = hardly
        "LOSS_FUNCTION": SMOOTHL1,  # Which optimizer to use?
        "USE_GRADIENT_CLIPPING": False,  # If the gradients should be clipped
        "SHOW_PLOTS": True,  # If you want to plot statistics after each episode
        "EPSILON_DECAY_STRATEGY": EXPONENTIAL,
        # What kind of strategy should be picked in order to decay epsilon during training
        "USE_TF32": True,  # Uses TF32 instead of Float32. Makes it faster, but you have lower precision
        "USE_BF16": True,  # Uses BF16 in forward pass or not. Makes it faster, but you have lower precision
        "USE_CLIP_FOREACH": torch.cuda.is_available(),
        # USE the foreach implementation of gradient clipping. Only relevant if 'USE_GRADIENT_CLIPPING' is True
    },
    "HYPERPARAMS": {
        "NUM_EPISODES": 10,  # How many training episodes should be run?
        "NUM_TEST_EPISODES": 100,  # How many test episodes should be run?
        "OPT_ITER": 32,  # How many iterations should be done for gradient descent after each episode?
        "BATCH_SIZE": 128,  # The batch size for doing gradient descent
        "BUFFER_SIZE": 10000,  # How many items can be stored in the replay buffer?
        "DISCOUNT": 0.95,  # The discount factor for the TD error
        "EPSILON": 0.05,  # The initial exploration rate for the epsilon greedy algo
        "EPSILON_MIN": 0.001,  # Minimum exploration rate
        "EPSILON_DECAY": 1.0,
        # If EPSILON_DECAY_STRATEGY == Linear, it determines either the amount of episodes until `EPSILON_MIN`. If EPSILON_DECAY_STRATEGY == EXPONENTIAL, it determines the rate of decay per episode. (if EXPONENTIAL: =1 in this case means no decay)
        "TAU": 0.001,  # Soft update parameter
        "GRADIENT_CLIPPING_VALUE": 1.0,  # The gradient clipping value
        "NUMBER_DISCRETE_ACTIONS": 5,  # If None, you use a continuous action space, else you use a discrete action set
        "TARGET_NET_UPDATE_FREQ": 20,
        # int: Gives the frequency when to update the target net. If target net is disabled, this param is not relevant. If == 1, you update at every step.
    },
    "OPTIMIZER": {
        "OPTIM_NAME": ADAM,  # Which optimizer to use
        "LEARNING_RATE": 3e-4,  # The learning rate for the agent
        "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
        "EPS": 1e-8,  # eps Adam param
        "WEIGHT_DECAY": 1e-2,  # The weight decay rate
        "USE_FUSION": torch.cuda.is_available()
        # if we have CUDA, we can use the fusion implementation of Adam -> Faster
    },
}

# Convenient Constants
SEED = CONFIG["SEED"]
DEVICE = CONFIG["DEVICE"]
USE_TF32 = CONFIG["USE_TF32"]
USE_ENV = CONFIG["USE_ENV"]
USE_ALGO = CONFIG["USE_ALGO"]
OPTIONS = CONFIG["OPTIONS"]
HYPERPARAMS = CONFIG["HYPERPARAMS"]
OPTIMIZER = CONFIG["OPTIMIZER"]
