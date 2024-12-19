"""
Here you can define some utility constants that you can use in your code to prevent TYPOS. Please use them in order to add functionality to our program.

Author: Daniel
"""
HOCKEY = "Hockey-v0"
PENDULUM = "Pendulum-v1"
CARTPOLE = "CartPole-v1"
DQN = "DQN"
ADAMW = "ADAMW"
L1 = "L1"
SUPPORTED_ENVIRONMENTS = {
    HOCKEY,
    PENDULUM,
    CARTPOLE
}
SUPPORTED_ALGORITHMS = {
    DQN
}
SUPPORTED_OPTIMIZERS = {
    ADAMW
}
SUPPORTED_LOSS_FUNCTIONS = {
    L1
}