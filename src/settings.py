"""
This is the config file for the training run main.py.
"""
import torch
from time import localtime, strftime

from src.util.constants import ADAM, EXPONENTIAL, MSE_LOSS, PENDULUM, PINK_NOISE, \
    SMOOTH_L1_LOSS, TDMPC2_ALGO, HOCKEY, MPO_ALGO

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
        "SEED": 24,  # The seed that we want to use
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # On which machine is it running?
        "USE_TF32": False,  # Uses TF32 instead of Float32. Makes it faster, but you have lower precision
        "USE_ENV": PENDULUM,  # The used environment
        "RENDER_MODE": None,  # The render mode. Supported: None for no rendering or HUMAN for rendering
        "NUMBER_DISCRETE_ACTIONS": None,
        # If None, you use a continuous action space, else you use a discrete action set
        "SELF_PLAY": False,  # If the agent should play against itself like in AlphaGo
        "USE_ALGO": TDMPC2_ALGO,  # The used algorithm for the main agent. SEE SUPPORTED_ALGORITHMS
        "BUFFER_SIZE": 1_000,  # How many items can be stored in the replay buffer?
        "MODEL_NAME": strftime('%y-%m-%d %H_%M_%S', localtime()),
        # under which name we want to store the logging results and the checkpoints
        "NUM_TRAINING_EPISODES": 10_000,  # How many training episodes should be run?
        "NUM_TEST_EPISODES": 100,  # How many test episodes should be run?
        "EPISODE_UPDATE_ITER": 1,
        # after how many episodes should the model be updated? =1, update your agent after every episode
        "SHOW_PLOTS": False,  # If you want to plot statistics after each episode
        "CHECKPOINT_ITER": 20,  # saves a checkpoint of this model after x iterations
        "CURIOSITY": None,  #Proportion of curiosity reward calculated by ICM to be added to the real reward. If None no curiosity exploration is used
    },
    # The settings for the agent.py
    "AGENT": {
        # GENERAL SETTINGS
        "USE_BF16": False,  # Uses BF16 in forward pass or not. Makes it faster, but you have lower precision
        "USE_COMPILE": False,  # if torch.compile should be used for the networks
        "OPT_ITER": 1,  # How many iterations should be done of gradient descent when calling agent.optimize()?
        "BATCH_SIZE": 200,  # The batch size for doing gradient descent
        "DISCOUNT": 0.95,  # The discount factor for the TD error

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
    },
    # The specific settings for the PPO agent
    "PPO": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": MSE_LOSS,
        "EPS_CLIP": 0.2,  # the clipping hyperparam for the ppo algo
    },
    # The specific settings for the DDPG agent
    "DDPG": {
        # Specific settings for the actor network. Each network e.g. can have another optimizer
        "ACTOR": {
            "OPTIMIZER": {
                "OPTIM_NAME": ADAM,
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
                "OPTIM_NAME": ADAM,  # Which optimizer to use
                "LEARNING_RATE": 3e-3,  # The learning rate for the agent
                "BETAS": (0.9, 0.999),  # The beta1, beta2 parameters of Adam
                "EPS": 1e-8,  # eps Adam param
                "WEIGHT_DECAY": 1e-2,  # The weight decay rate
                "USE_FUSION": torch.cuda.is_available()
                # if we have CUDA, we can use the fusion implementation of Adam -> Faster
            },
            "LOSS_FUNCTION": SMOOTH_L1_LOSS,
        },
        "NOISE": _DEFAULT_NOISE,

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
        "NUM_LAYERS": 5, #num hidden layers, only changed if target_net == false
        "BATCHNORM_MOMENTUM": 0.9, #momentum for batchnorm, only used if target_net == false
        "NOISE": _DEFAULT_NOISE
    },
    "SAC": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": _DEFAULT_LOSS_FUNCTION,
        "LEARN_ALPHA": True,  # Whether to learn the temperature alpha
        "TARGET_ENTROPY": None,  # Target entropy for automatic alpha
        "INIT_ALPHA": 0.2,
        "HIDDEN_DIM": 256
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
        "DISCRETE": False
        # All other Hyperparameters are set in the MPO class
    },
    "TD_MPC2": {
        "OPTIMIZER": _DEFAULT_OPTIMIZER,
        "LOSS_FUNCTION": _DEFAULT_LOSS_FUNCTION,
        "NOISE": _DEFAULT_NOISE,
        "HORIZON": 3,  # How many steps do we want to consider while doing predictions
        "MMPI_ITERATIONS": 6,
        "NUM_TRAJECTORIES": 8,
        "NUM_SAMPLES": 256,
        "NUM_ELITES": 64,
        "MIN_STD": 0.05,
        "MAX_STD": 2,
        "TEMPERATURE": 0.5,
        "LATENT_SIZE": 512,
        "LOG_STD_MIN": -10,
        "LOG_STD_MAX": 2,
        "ENTROPY_COEF": 1e-4,
        "ENC_LR_SCALE": 0.3,
        "GRAD_CLIP_NORM": 20,
        "LR": 3e-4,
        "CONSISTENCY_COEF": 20,
        "REWARD_COEF": 0.1,
        "Q_COEF": 0.1,
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
