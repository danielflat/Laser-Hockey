"""
Here you can define some utility constants that you can use in your code to prevent TYPOS. Please use them in order to add functionality to our program.

Author: Daniel
"""

# All envs
HOCKEY = "Hockey-v0"
PENDULUM = "Pendulum-v1"
CARTPOLE = "CartPole-v1"
LUNARLANDER = "LunarLander-v3"

# All Algos
DQN_ALGO = "DQN_Algo"
PPO_ALGO = "PPO_Algo"
DDPG_ALGO = "DDPG_Algo"
TD3_ALGO = "TD3_Algo"
SAC_ALGO = "SAC_Algo"
MPO_ALGO = "MPO_Algo"
RANDOM_ALGO = "Random_Algo"
WEAK_COMP_ALGO = "Weak_Comp_Algo"
STRONG_COMP_ALGO = "Strong_Comp_Algo"

# All Optimizers
ADAMW = "AdamW"
ADAM = "Adam"

# All Loss functions
L1_LOSS = "L1"
SMOOTH_L1_LOSS = "SmoothL1Loss"
MSE_LOSS = "MSELoss"
CROSS_ENTROPY_LOSS = "CrossEntropyLoss"

# All Epsilon Greedy Strategies
LINEAR = "Linear"
EXPONENTIAL = "Exponential"

# All Render modes
HUMAN = "human"

# All Noise strategies
WHITE_NOISE = "White_Noise"
OU_NOISE = "OU_Noise"
PINK_NOISE = "Pink_Noise"

SUPPORTED_ENVIRONMENTS = {
    HOCKEY,
    PENDULUM,
    CARTPOLE,
    LUNARLANDER
}
SUPPORTED_ALGORITHMS = {
    DQN_ALGO,
    PPO_ALGO,
    DDPG_ALGO,
    TD3_ALGO,
    SAC_ALGO,
    MPO_ALGO,
    RANDOM_ALGO,
    WEAK_COMP_ALGO,
    STRONG_COMP_ALGO,
}
SUPPORTED_OPTIMIZERS = {
    ADAMW,
    ADAM
}
SUPPORTED_LOSS_FUNCTIONS = {
    L1_LOSS,
    SMOOTH_L1_LOSS,
    MSE_LOSS,
    CROSS_ENTROPY_LOSS,
}
SUPPORTED_EPSILON_DECAY_STRATEGIES = {
    LINEAR,
    EXPONENTIAL,
}
SUPPORTED_RENDER_MODES = {
    None,
    HUMAN
}
SUPPORTED_NOISE_TYPES = {
    WHITE_NOISE,
    OU_NOISE,
    PINK_NOISE
}
