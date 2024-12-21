from itertools import count

import torch

from src.config import DEVICE, HYPERPARAMS, OPTIMIZER, OPTIONS
from src.util.constants import DQN, HUMAN, PENDULUM
from src.util.contract import initAgent, initEnv
from src.util.directoryutil import get_path

# USEFUL CONSTANTS
TEST_CHECK_POINT_NAME = "24-12-21 18_36_31_00020.pth"
TEST_USE_ENV = PENDULUM
TEST_USE_ALGO = DQN
TEST_NUMBER_DISCRETE_ACTIONS = 9
TEST_SEED = 42
TEST_RENDER_MODE = HUMAN

if __name__ == '__main__':

    env = initEnv(TEST_USE_ENV, TEST_RENDER_MODE, TEST_NUMBER_DISCRETE_ACTIONS)

    agent = initAgent(use_algo=TEST_USE_ALGO, env=env, options=OPTIONS, optim=OPTIMIZER, hyperparams=HYPERPARAMS,
                      device=DEVICE)
    check_point = get_path(f"output/checkpoints/{TEST_CHECK_POINT_NAME}")
    agent.setMode(eval=True)
    agent.loadModel(file_name=check_point)

    state, info = env.reset(seed=TEST_SEED)
    env.render()

    total_steps = 0
    total_reward = 0

    for i_test in count():
        state = torch.tensor(state, device=DEVICE)
        action = agent.act(state)

        next_step, reward, terminated, truncated, info = env.step(action)
        total_steps += 1
        total_reward += reward

        state = next_step

        done = terminated or truncated

        if done:
            break
        # if i_test % 200 == 0:
        #     state, info = env.reset()
    print(f"FINISHED TEST! Total reward: {total_steps} | Total reward: {total_reward}")
