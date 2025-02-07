"""
This is the config file for the training run main.py.
"""
import torch
from time import localtime, strftime

from src.util.constants import ADAM, ADAMW, DDPG_ALGO, DQN_ALGO, EXPONENTIAL, HOCKEY, MSE_LOSS, PENDULUM, PINK_NOISE, \
    SMOOTH_L1_LOSS, TDMPC2_ALGO
from src.util.directoryutil import get_path

_DEFAULT_OPTIMIZER = {
    "OPTIM_NAME": ADAM,  # Which optimizer to use
    "LEARNING_RATE": 3e-4,  # The learning rate for the agent
    "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
    "EPS": 1e-8,  # eps Adam param
    "WEIGHT_DECAY": 1e-2,  # The weight decay rate
    "USE_FUSION": torch.cuda.is_available()
    # if we have CUDA, we can use the fusion implementation of Adam -> Faster
}
_DEFAULT_LOSS_FUNCTION = SMOOTH_L1_LOSS
_DEFAULT_NOISE = {
    "NOISE_TYPE": PINK_NOISE,
    "NOISE_FACTOR": 0.5,
    "NOISE_PARAMS": {
        # Params for white noise
        "MEAN": 0,
        "STD": 0.1,

        # Params for OU noise
        "THETA": 0.15,
        "DT": 1e-2,
    }
}

SETTINGS = {
    # The settings for the main.py
    "MAIN": {
        # General settings
        "SEED": 24,  # The seed that we want to use
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # On which machine is it running?
        "USE_TF32": False,  # Uses TF32 instead of Float32. Makes it faster, but you have lower precision

        # Environment settings
        "USE_ENV": HOCKEY,  # The used environment
        "PROXY_REWARDS": True,  # If the agent should get proxy rewards (works only with HOCKEY)
        "RENDER_MODE": None,  # The render mode. Supported: None for no rendering or HUMAN for rendering
        "NUMBER_DISCRETE_ACTIONS": None,
        # If None, you use a continuous action space, else you use a discrete action set
        "USE_ALGO": TDMPC2_ALGO,  # The used algorithm for the main agent. SEE SUPPORTED_ALGORITHMS

        # Defining training loop
        "BUFFER_SIZE": 1_000,  # How many items can be stored in the replay buffer?
        "NUM_TRAINING_EPISODES": 10_000,  # How many training episodes should be run?
        "NUM_TEST_EPISODES": 1_000,  # How many test episodes should be run?
        "EPISODE_UPDATE_ITER": 1,
        # after how many episodes should the model be updated? =1, update your agent after every episode
        "CURIOSITY": None,
        # Proportion of curiosity reward calculated by ICM to be added to the real reward. If None no curiosity exploration is used

        # PLOTTING
        "SHOW_PLOTS": False,  # If you want to plot statistics after each episode
        "PLOT_FREQUENCY": 100,  # After how many episodes you want to refresh the plots
        "BATTLE_STATISTICS_FREQUENCY": 100,
        # After how many episodes you want to log the battle statistics on the console

        # CHECKPOINT: You can set a checkpoint name. It can either be None or the path
        # e.g. `get_path("output/checkpoints/25-01-16 09_15_28/25-01-16 09_15_28_00640.pth")`
        "CHECKPOINT_NAME": get_path("good_checkpoints/25-01-25 14_24_21_000174500.pth"),
        "CHECKPOINT_ITER": 500,  # saves a checkpoint of this model after x iterations
        "MODEL_NAME": strftime('%y-%m-%d %H_%M_%S', localtime()),
        # under which name we want to store the logging results and the checkpoints

        # SELF-Play Settings
        "SELF_PLAY": True,  # If the agent should play against itself like in AlphaGo
        "SELF_PLAY_FREQUENCY": 10,
        # Frequency of self-play episodes. Play 1/#Number against an agent from the other pool. Play #Number-1/#Number against a version of itself
        "SELF_PLAY_KEEP_AGENT_FREQUENCY": 5000,
        # Put a checkpoint of your agent after x iterations into your opponent pool?
        "SELF_PLAY_UPDATE_FREQUENCY": 500,  # After how many iterations do you want to hard-update the self_opponent?
        "WEIGHTING_RULE": lambda win_rate: (1 - win_rate) + 0.1,
        # The rule for weighting the opponents in the opponent_pool
    },

    # The settings for the agent.py
    "AGENT": {
        # GENERAL SETTINGS
        "USE_BF16": False,  # Uses BF16 in forward pass or not. Makes it faster, but you have lower precision
        "USE_COMPILE": False,  # if torch.compile should be used for the networks
        "OPT_ITER": 32,  # How many iterations should be done of gradient descent when calling agent.optimize()?
        "BATCH_SIZE": 256,  # The batch size for doing gradient descent
        "DISCOUNT": 0.99,  # The discount factor for the TD error

        # TARGET NET STRATEGY
        "USE_TARGET_NET": True,  # If a target net is used
        "USE_SOFT_UPDATES": True,  # If the target network is updated. True = softly, False = hardly
        "TARGET_NET_UPDATE_FREQ": 1,
        # int: Gives the frequency when to update the target net. If target net is disabled, this param is not relevant. If == 1, you update at every step.
        "TAU": 0.001,  # Soft update parameter

        # EPSILON GREEDY STRATEGY
        "EPSILON_DECAY_STRATEGY": EXPONENTIAL,
        # What kind of strategy should be picked in order to decay epsilon during training
        "EPSILON": 0.1,  # The initial exploration rate for the epsilon greedy algo
        "EPSILON_MIN": 0.001,  # Minimum exploration rate
        "EPSILON_DECAY": 0.999,
        # If EPSILON_DECAY_STRATEGY == Linear, it determines either the amount of episodes until `EPSILON_MIN`. If EPSILON_DECAY_STRATEGY == EXPONENTIAL, it determines the rate of decay per episode. (if EXPONENTIAL: =1 in this case means no decay)

        # BACKWARD STEP STRATEGY
        "USE_GRADIENT_CLIPPING": False,  # If the gradients should be clipped
        "GRADIENT_CLIPPING_VALUE": 1.0,  # The gradient clipping value
        "USE_NORM_CLIPPING": True,  # If the norm of the gradients should be clipped
        "NORM_CLIPPING_VALUE": 20.0,  # The gradient clipping value
        "USE_CLIP_FOREACH": torch.cuda.is_available(),
        # USE the foreach implementation of gradient clipping. Only relevant if 'USE_GRADIENT_CLIPPING' is True
    },
    # The specific settings for the DQN agent
    "DQN": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": SMOOTH_L1_LOSS,
        "CHECKPOINT_NAME": None,  # which checkpoint should be used for the DQN Hockey agent?
    },
    # The specific settings for the PPO agent
    "PPO": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": MSE_LOSS,
        "EPS_CLIP": 0.2,  # the clipping hyperparam for the ppo algo
        "CHECKPOINT_NAME": None,  # which checkpoint should be used for the PPO Hockey agent?
    },
    # The specific settings for the DDPG agent
    "DDPG": {
        # Specific settings for the actor network. Each network e.g. can have another optimizer
        "ACTOR": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAMW,
                "LEARNING_RATE": 3e-4,  # The learning rate for the agent
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
                "OPTIM_NAME": ADAMW,  # Which optimizer to use
                "LEARNING_RATE": 3e-3,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-2,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
                # if we have CUDA, we can use the fusion implementation of Adam -> Faster
            },
            "LOSS_FUNCTION": MSE_LOSS,
        },
        "NOISE": _DEFAULT_NOISE,
        "CHECKPOINT_NAME": get_path("good_checkpoints/hockey_ddpg_smoothl1_25-01-22 17_36_56_100000.pth"),
        # which checkpoint should be used for the DDPG Hockey agent?
    },
    # The specific settings for the TD3 agent
    "TD3": {
        "ACTOR": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,
                "LEARNING_RATE": 0.0001,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-2,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
            },
        },
        # Specific settings for the critic network
        "CRITIC": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,  # Which optimizer to use
                "LEARNING_RATE": 0.001,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-2,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
            },
            "LOSS_FUNCTION": SMOOTH_L1_LOSS,
        },
        "POLICY_DELAY": 2,  # The delay of the policy optimization
        "NOISE_CLIP": 0.1,  # The gaussian noise clip value
        "HIDDEN_DIM": 128,
        "NUM_LAYERS": 5,  # num hidden layers, only changed if target_net == false
        "BATCHNORM_MOMENTUM": 0.9,  # momentum for batchnorm, only used if target_net == false
        "NOISE": _DEFAULT_NOISE,
        "CHECKPOINT_NAME": get_path("good_checkpoints/hockey_ddpg_smoothl1_25-01-22 17_36_56_100000.pth"),
        # which checkpoint should be used for the TD3 Hockey agent?
    },
    "SAC": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": _DEFAULT_LOSS_FUNCTION,
        "LEARN_ALPHA": True,  # Whether to learn the temperature alpha
        "TARGET_ENTROPY": None,  # Target entropy for automatic alpha
        "INIT_ALPHA": 0.2,
        "HIDDEN_DIM": 256,
        "CHECKPOINT_NAME": None,  # which checkpoint should be used for the SAC Hockey agent?
        "NUM_EPOCHS": 1_000,
        "NUM_EPISODES_PER_TRAINING_EPOCH": 20,
        "NUM_EPISODES_PER_VALIDATION_EPOCH": 20,
        "VALIDATION_FREQ": 10,
        "TRAIN_FREQ": 1
    },
    "MPO": {
        "ACTOR": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,
                "LEARNING_RATE": 3e-4,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-5,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
            },
        },
        # Specific settings for the critic network
        "CRITIC": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,
                "LEARNING_RATE": 3e-4,
                "BETAS": (0.9, 0.999),
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-5,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
            },
            "LOSS_FUNCTION": SMOOTH_L1_LOSS,
        },
        "LAGRANGIANS": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,
                "LEARNING_RATE": 0.1,
                "BETAS": (0.9, 0.999),
                "EPS": 1e-8,
                "WEIGHT_DECAY": 1e-4,
                "USE_FUSION": torch.cuda.is_available()
            },
        },
        "SAMPLE_ACTION_NUM": 32,  # Number of actions to sample for nonparametric policy optimization
        "MSTEP_ITER": 1,
        "DISCRETE": False,
        # All other Hyperparameters are set in the MPO class
        "CHECKPOINT_NAME": None,  # which checkpoint should be used for the PPO Hockey agent?
    },
    "TD_MPC2": {
        "OPTIMIZER": {
            "OPTIM_NAME": ADAMW,  # Which optimizer to use
            "LEARNING_RATE": 3e-4,  # The learning rate for the agent
            "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
            "EPS": 1e-8,  # eps Adam param
            "WEIGHT_DECAY": 1e-2,  # The weight decay rate
            "USE_FUSION": torch.cuda.is_available()
            # if we have CUDA, we can use the fusion implementation of Adam -> Faster
        },
        "CONSISTENCY_LOSS_FUNCTION": MSE_LOSS,  # Which Loss function to use for the consistency loss
        "REWARD_LOSS_FUNCTION": MSE_LOSS,  # Which Loss function to use for the reward loss
        "Q_LOSS_FUNCTION": MSE_LOSS,  # Which Loss function to use for the q loss
        "CONSISTENCY_COEF": 20,  # factor for the consistency loss
        "REWARD_COEF": 0.1,  # factor for the reward loss
        "Q_COEF": 0.1,  # factor for the Q loss
        "ENTROPY_COEF": 1e-4,
        "ENC_LR_SCALE": 0.3,

        "NOISE": _DEFAULT_NOISE,  # Which noise should we add
        "HORIZON": 3,  # How many steps do we want to consider while doing predictions

        "MMPI_ITERATIONS": 6,  # How many iterations of MPPI should we use for planning
        "NUM_TRAJECTORIES": 8,
        "NUM_SAMPLES": 256,
        "NUM_ELITES": 64,
        "MIN_STD": 0.05,
        "MAX_STD": 2,
        "TEMPERATURE": 0.5,
        "LATENT_SIZE": 512,
        "LOG_STD_MIN": -10,
        "LOG_STD_MAX": 2,
        "CHECKPOINT_NAME": None,  # which checkpoint should be used for the TD-MPC-2 Hockey agent?
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
TD_MPC2_SETTINGS = SETTINGS["TD_MPC2"]

SEED = MAIN_SETTINGS["SEED"]
DEVICE = MAIN_SETTINGS["DEVICE"]
USE_TF32 = MAIN_SETTINGS["USE_TF32"]
USE_ENV = MAIN_SETTINGS["USE_ENV"]
PROXY_REWARDS = MAIN_SETTINGS["PROXY_REWARDS"]
RENDER_MODE = MAIN_SETTINGS["RENDER_MODE"]
NUMBER_DISCRETE_ACTIONS = MAIN_SETTINGS["NUMBER_DISCRETE_ACTIONS"]
USE_ALGO = MAIN_SETTINGS["USE_ALGO"]
SELF_PLAY = MAIN_SETTINGS["SELF_PLAY"]
MODEL_NAME = MAIN_SETTINGS["MODEL_NAME"]
BUFFER_SIZE = MAIN_SETTINGS["BUFFER_SIZE"]
NUM_TRAINING_EPISODES = MAIN_SETTINGS["NUM_TRAINING_EPISODES"]
NUM_TEST_EPISODES = MAIN_SETTINGS["NUM_TEST_EPISODES"]
EPISODE_UPDATE_ITER = MAIN_SETTINGS["EPISODE_UPDATE_ITER"]
SHOW_PLOTS = MAIN_SETTINGS["SHOW_PLOTS"]
CHECKPOINT_ITER = MAIN_SETTINGS["CHECKPOINT_ITER"]
CHECKPOINT_NAME = MAIN_SETTINGS["CHECKPOINT_NAME"]
CURIOSITY = MAIN_SETTINGS["CURIOSITY"]
SELF_PLAY_FREQUENCY = MAIN_SETTINGS["SELF_PLAY_FREQUENCY"]
SELF_PLAY_KEEP_AGENT_FREQUENCY = MAIN_SETTINGS["SELF_PLAY_KEEP_AGENT_FREQUENCY"]
PLOT_FREQUENCY = MAIN_SETTINGS["PLOT_FREQUENCY"]
BATTLE_STATISTICS_FREQUENCY = MAIN_SETTINGS["BATTLE_STATISTICS_FREQUENCY"]
SELF_PLAY_UPDATE_FREQUENCY = MAIN_SETTINGS["SELF_PLAY_UPDATE_FREQUENCY"]
WEIGHTING_RULE = MAIN_SETTINGS["WEIGHTING_RULE"]

BATCH_SIZE = AGENT_SETTINGS["BATCH_SIZE"]

# SAC Params
SAC_NUM_EPOCHS = SAC_SETTINGS["NUM_EPOCHS"]
SAC_NUM_EPISODES_PER_TRAINING_EPOCH = SAC_SETTINGS["NUM_EPISODES_PER_TRAINING_EPOCH"]
SAC_NUM_EPISODES_PER_VALIDATION_EPOCH = SAC_SETTINGS["NUM_EPISODES_PER_VALIDATION_EPOCH"]
SAC_VALIDATION_FREQ = SAC_SETTINGS["VALIDATION_FREQ"]
SAC_TRAIN_FREQ = SAC_SETTINGS["TRAIN_FREQ"]
