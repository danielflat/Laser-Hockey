"""
Here you can define some utility constants that you can use in your code to prevent TYPOS. Please use them in order to add functionality to our program.

Author: Daniel
"""
HOCKEY = "Hockey-v0"
PENDULUM = "Pendulum-v1"
CARTPOLE = "CartPole-v1"
LUNARLANDER = "LunarLander-v3"
DQN_ALGO = "DQN_Algo"
PPO_ALGO = "PPO_Algo"
TD3_ALGO = "TD3_Algo"
SAC_ALGO = "SAC_Algo"
MPO_ALGO = "MPO_Algo"
ADAMW = "AdamW"
ADAM = "Adam"
L1 = "L1"
SMOOTHL1 = "SmoothL1"
MSELOSS = "MSELoss"
LINEAR = "Linear"
EXPONENTIAL = "Exponential"
HUMAN = "human"
SUPPORTED_ENVIRONMENTS = {
    HOCKEY,
    PENDULUM,
    CARTPOLE,
    LUNARLANDER
}
SUPPORTED_ALGORITHMS = {
    DQN_ALGO,
    PPO_ALGO,
    TD3_ALGO,
    MPO_ALGO
}
SUPPORTED_OPTIMIZERS = {
    ADAMW,
    ADAM
}
SUPPORTED_LOSS_FUNCTIONS = {
    L1,
    SMOOTHL1,
    MSELOSS
}
SUPPORTED_EPSILON_DECAY_STRATEGIES = {
    LINEAR,
    EXPONENTIAL,
}
SUPPORTED_RENDER_MODES = {
    None,
    HUMAN
}
