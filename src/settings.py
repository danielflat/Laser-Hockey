"""
This is the config file for the training run main.py.
"""
from time import localtime, strftime

import torch

from src.util.constants import ADAM, DDPG_ALGO, EXPONENTIAL, MSELOSS, PENDULUM, SMOOTHL1

_DEFAULT_OPTIMIZER = {
    "OPTIM_NAME": ADAM,  # Which optimizer to use
    "LEARNING_RATE": 3e-4,  # The learning rate for the agent
    "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
    "EPS": 1e-8,  # eps Adam param
    "WEIGHT_DECAY": 1e-2,  # The weight decay rate
    "USE_FUSION": torch.cuda.is_available()
    # if we have CUDA, we can use the fusion implementation of Adam -> Faster
}
_DEFAULT_LOSS_FUNCTION = SMOOTHL1

SETTINGS = {
    # The settings for the main.py
    "MAIN": {
        "SEED": 24,  # The seed that we want to use
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # On which machine is it running?
        "USE_TF32": True,  # Uses TF32 instead of Float32. Makes it faster, but you have lower precision
        "USE_ENV": PENDULUM,  # The used environment
        "RENDER_MODE": None,  # The render mode. Supported: None or HUMAN
        "NUMBER_DISCRETE_ACTIONS": None,
        # If None, you use a continuous action space, else you use a discrete action set
        "USE_ALGO": DDPG_ALGO,  # The used algorithm. Either DQN_ALGO or PPO_ALGO
        "BUFFER_SIZE": 1000000,  # How many items can be stored in the replay buffer?
        "MODEL_NAME": strftime('%y-%m-%d %H_%M_%S', localtime()),
        # under which name we want to store the logging results and the checkpoints
        "NUM_TRAINING_EPISODES": 2000,  # How many training episodes should be run?
        "NUM_TEST_EPISODES": 100,  # How many test episodes should be run?
        "EPISODE_UPDATE_ITER": 1,
        # after how many episodes should the model be updated? =1, update your agent after every episode
        "SHOW_PLOTS": True,  # If you want to plot statistics after each episode
        "CHECKPOINT_ITER": 20,  # saves a checkpoint of this model after x iterations

    },
    # The settings for the agent.py
    "AGENT": {
        # GENERAL SETTINGS
        "USE_BF16": False,  # Uses BF16 in forward pass or not. Makes it faster, but you have lower precision
        "SELF_TRAINING": False,  # If the agent should play against itself like in AlphaGo
        "USE_COMPILE": False,  # if torch.compile should be used for the networks
        "OPT_ITER": 32,  # How many iterations should be done of gradient descent when calling agent.optimize()?
        "BATCH_SIZE": 128,  # The batch size for doing gradient descent
        "DISCOUNT": 0.95,  # The discount factor for the TD error

        # TARGET NET STRATEGY
        "USE_TARGET_NET": True,  # If a target net is used
        "USE_SOFT_UPDATES": False,  # If the target network is updated. True = softly, False = hardly
        "TARGET_NET_UPDATE_FREQ": 100,
        # int: Gives the frequency when to update the target net. If target net is disabled, this param is not relevant. If == 1, you update at every step.
        "TAU": 0.001,  # Soft update parameter

        # EPSILON GREEDY STRATEGY
        "EPSILON_DECAY_STRATEGY": EXPONENTIAL,
        # What kind of strategy should be picked in order to decay epsilon during training
        "EPSILON": 0,  # The initial exploration rate for the epsilon greedy algo
        "EPSILON_MIN": 0.001,  # Minimum exploration rate
        "EPSILON_DECAY": 0.999,
        # If EPSILON_DECAY_STRATEGY == Linear, it determines either the amount of episodes until `EPSILON_MIN`. If EPSILON_DECAY_STRATEGY == EXPONENTIAL, it determines the rate of decay per episode. (if EXPONENTIAL: =1 in this case means no decay)

        # BACKWARD STEP STRATEGY
        "USE_GRADIENT_CLIPPING": False,  # If the gradients should be clipped
        "GRADIENT_CLIPPING_VALUE": 1.0,  # The gradient clipping value
        "USE_CLIP_FOREACH": torch.cuda.is_available(),
        # USE the foreach implementation of gradient clipping. Only relevant if 'USE_GRADIENT_CLIPPING' is True
    },
    # The specific settings for the DQN agent
    "DQN": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": SMOOTHL1,
    },
    # The specific settings for the PPO agent
    "PPO": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": MSELOSS,
        "EPS_CLIP": 0.2,  # the clipping hyperparam for the ppo algo
    },
    # The specific settings for the DDPG agent
    "DDPG": {
        # Specific settings for the actor network. Each network e.g. can have another optimizer
        "ACTOR": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,
                "LEARNING_RATE": 0.00001,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-2,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
                # if we have CUDA, we can use the fusion implementation of Adam -> Faster
            },
        },
        # Specific settings for the critic network
        "CRITIC": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,  # Which optimizer to use
                "LEARNING_RATE": 0.0001,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-2,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
                # if we have CUDA, we can use the fusion implementation of Adam -> Faster
            },
            "LOSS_FUNCTION": SMOOTHL1,
        },
        "NOISE_FACTOR": 0.1,
        "NOISE_NAME": "Normal",
        "NOISE_PARAMS": {
            "MEAN": 0,
            "STD": 0.1
        }
    },
    # The specific settings for the TD3 agent
    "TD3": {
        "POLICY_DELAY": 2,  # The delay of the policy optimization
        "NOISE_CLIP": 1,  # The gaussian noise clip value
    },
    "SAC": {
        "LEARN_ALPHA": True, # Whether to learn the temperature alpha
        "TARGET_ENTROPY": None, # Target entropy for automatic alpha
        "INIT_ALPHA": 0.2,
        "HIDDEN_DIM": 256
    },
    "MPO": {
        "HIDDEN_DIM": 64,
        "SAMPLE_ACTION_NUM" : 10,
        "DUAL_CONSTAINT": 0.1,
        "KL_CONSTRAINT": 0.01,
        "MSTEP_ITER": 10,
        "ALPHA_SCALE": 1.0
    }
}

# Convenient Constants
MAIN_SETTINGS = SETTINGS["MAIN"]
AGENT_SETTINGS = SETTINGS["AGENT"]
DQN_SETTINGS = SETTINGS["DQN"]
PPO_SETTINGS = SETTINGS["PPO"]
DDPG_SETTINGS = SETTINGS["DDPG"]
TD3_SETTINGS = SETTINGS["TD3"]
SAC_SETTINGS = SETTINGS["SAC"]
MPO_SETTINGS = SETTINGS["MPO"]
