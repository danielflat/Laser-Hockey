import copy

import time

from itertools import count
import numpy as np
import pandas as pd
import torch

from src.agent import Agent
from src.settings import AGENT_SETTINGS, DQN_SETTINGS, PPO_SETTINGS, MPO_SETTINGS, TD_MPC2_SETTINGS
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, HUMAN, PENDULUM, HALFCHEETAH, RANDOM_ALGO, \
    SAC_ALGO, STRONG_COMP_ALGO, \
    TDMPC2_ALGO, WEAK_COMP_ALGO
from src.util.contract import initAgent, initEnv, initSeed
from src.util.directoryutil import get_path

"""
Is the class that evaluates the TD-MPC2 with the standard hockey agents.
After running, it creates the "plots/eval_tdmpc2_hockey_results.tex" which is a table used in the paper.
Below, settings can be adjusted for the evaluation.

Author:
Daniel Flat
"""

# Settings for this class
BEST_TDMPC2_CHECKPOINT = get_path(
    "final_checkpoints/tdmpc2-v2-all-i7 25-02-20 17_44_47_000067500.pth")  # Which checkpoint do you want to test
BEST_DDPG_CHECKPOINT = get_path(
    "final_checkpoints/hockey_ddpg_smoothl1_25-01-22 17_36_56_100000.pth")  # Which checkpoint do you want to test
TOURNAMENT_RESULTS_FILE_NAME = get_path("plots/eval_tdmpc2_hockey_results.tex")
NUM_GAMES_PER_MATCH = 1000  # The number of games each pair is playing against
TABLE_CAPTION = "TD-MPC2 hockey performance against standard opponents"
TABLE_LABEL = "tab:eval_tdmpc2_hockey"

TOURNAMENT_USE_ENV = HOCKEY  # On which environment do you want to test?
TOURNAMENT_USE_PROXY_REWARDS = True  # On which environment do you want to test?
TOURNAMENT_NUMBER_DISCRETE_ACTIONS = None  # if you want to use discrete actions or continuous. If > 0, you use the DiscreteActionWrapper
TEST_SEED = 1000000  # Set a test seed if you want to
TEST_RENDER_MODE = HUMAN  # For whom do you want to render? None or HUMAN
TEST_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # which device are you using?


def play_matches(env, player1, player2):
    episode_steps = []
    episode_rewards = []
    num_won = 0
    num_draw = 0
    num_loss = 0
    for i_test in range(1, NUM_GAMES_PER_MATCH + 1):
        t_start = time.time()
        total_steps = 0
        total_reward = 0

        if isinstance(player1, Agent):
            player1.reset()
        if isinstance(player2, Agent):
            player2.reset()

        state, info = env.reset(seed=TEST_SEED + i_test)
        state2 = env.obs_agent_two()
        env.render()

        for _ in count():
            if TOURNAMENT_USE_ENV == HOCKEY:
                env.render()

            state = torch.tensor(state, device=TEST_DEVICE, dtype=torch.float32)
            state2 = torch.tensor(state2, device=TEST_DEVICE, dtype=torch.float32)

            action1 = player1.act(state)
            action2 = player2.act(state2)

            next_step, reward, terminated, truncated, info = env.step(np.hstack([action1, action2]))

            total_steps += 1
            total_reward += reward

            state = next_step
            state2 = env.obs_agent_two()
            done = terminated or truncated

            if done:
                episode_steps.append(total_steps)
                episode_rewards.append(total_reward)

                if info["winner"] == 1:
                    num_won += 1
                    episode_result = "WON"
                elif info["winner"] == -1:
                    num_loss += 1
                    episode_result = "LOST"
                else:
                    num_draw += 1
                    episode_result = "DRAW"

                t_end = time.time()
                t_required = t_end - t_start
                print(
                    f"Episode: {i_test} | {player1} vs. {player2} | Result: {episode_result} | Total steps: {total_steps} | Total reward: {total_reward} | Req. Time: {t_required:.4} sec.")
                break
    return episode_steps, episode_rewards, num_won, num_draw, num_loss


def eval_mini_tournament():
    initSeed(seed=TEST_SEED, device=TEST_DEVICE)
    print(f"Test seed: {TEST_SEED}, Test Device: {TEST_DEVICE}")

    env = initEnv(TOURNAMENT_USE_ENV, TEST_RENDER_MODE, TOURNAMENT_NUMBER_DISCRETE_ACTIONS,
                  proxy_rewards=TOURNAMENT_USE_PROXY_REWARDS)

    tdmpc_agent = initAgent(use_algo=TDMPC2_ALGO, env=env, agent_settings=AGENT_SETTINGS,
                            device=TEST_DEVICE,
                            checkpoint_name=BEST_TDMPC2_CHECKPOINT)
    tdmpc_agent.setMode(eval=True)

    random_agent = initAgent(use_algo=RANDOM_ALGO, env=env, agent_settings=AGENT_SETTINGS,
                             device=TEST_DEVICE,
                             checkpoint_name=None)
    random_agent.setMode(eval=True)

    weak_agent = initAgent(use_algo=WEAK_COMP_ALGO, env=env, agent_settings=AGENT_SETTINGS,
                           device=TEST_DEVICE,
                           checkpoint_name=None)
    weak_agent.setMode(eval=True)

    strong_agent = initAgent(use_algo=STRONG_COMP_ALGO, env=env, agent_settings=AGENT_SETTINGS,
                             device=TEST_DEVICE,
                             checkpoint_name=None)
    strong_agent.setMode(eval=True)

    ddpg_agent = initAgent(use_algo=DDPG_ALGO, env=env, agent_settings=AGENT_SETTINGS,
                           device=TEST_DEVICE,
                           checkpoint_name=BEST_DDPG_CHECKPOINT)
    ddpg_agent.setMode(eval=True)

    game1_episode_steps, game1_episode_rewards, game1_num_won, game1_num_draw, game1_num_loss = play_matches(env,
                                                                                                             player1=tdmpc_agent,
                                                                                                             player2=random_agent)
    game2_episode_steps, game2_episode_rewards, game2_num_won, game2_num_draw, game2_num_loss = play_matches(env,
                                                                                                             player1=tdmpc_agent,
                                                                                                             player2=weak_agent)
    game3_episode_steps, game3_episode_rewards, game3_num_won, game3_num_draw, game3_num_loss = play_matches(env,
                                                                                                             player1=tdmpc_agent,
                                                                                                             player2=strong_agent)
    game4_episode_steps, game4_episode_rewards, game4_num_won, game4_num_draw, game4_num_loss = play_matches(env,
                                                                                                             player1=tdmpc_agent,
                                                                                                             player2=ddpg_agent)

    avg_episode_length = [np.mean(game1_episode_steps), np.mean(game2_episode_steps), np.mean(game3_episode_steps),
                          np.mean(game4_episode_steps)]
    wins = [game1_num_won, game2_num_won, game3_num_won, game4_num_won]
    draws = [game1_num_draw, game2_num_draw, game3_num_draw, game4_num_draw]
    losses = [game1_num_loss, game2_num_loss, game3_num_loss, game4_num_loss]
    winner1 = tdmpc_agent if game1_num_won > game1_num_loss else random_agent if game1_num_loss > game1_num_won else "-"
    winner2 = tdmpc_agent if game2_num_won > game2_num_loss else weak_agent if game2_num_loss > game2_num_won else "-"
    winner3 = tdmpc_agent if game3_num_won > game3_num_loss else strong_agent if game3_num_loss > game3_num_won else "-"
    winner4 = tdmpc_agent if game4_num_won > game4_num_loss else ddpg_agent if game4_num_loss > game4_num_won else "-"
    winners = [winner1, winner2, winner3, winner4]

    # Example DataFrame
    data = {
        "Match Number": [1, 2, 3, 4],
        "Player 1": [tdmpc_agent, tdmpc_agent, tdmpc_agent, tdmpc_agent],
        "Player 2": [random_agent, weak_agent, strong_agent, ddpg_agent],
        "Number of Games": [NUM_GAMES_PER_MATCH, NUM_GAMES_PER_MATCH, NUM_GAMES_PER_MATCH, NUM_GAMES_PER_MATCH],
        "Avg. Game Length": avg_episode_length,
        "Player 1 won": wins,
        "Player 2 won": losses,
        "Number of Draws": draws,
        "Who won?": winners,
    }

    df = pd.DataFrame(data)

    # Convert to LaTeX format
    latex_code = df.to_latex(index=False, float_format="%.2f")

    latex_with_resize = (
            "\\begin{table}[h]\n"
            "\\centering\n"
            "\\resizebox{\\columnwidth}{!}{%\n"
            + latex_code +
            "}\n"
            "\\caption{" + TABLE_CAPTION + "}\n"
                                           "\\label{" + TABLE_LABEL + "}\n"
                                                                      "\\end{table}"
    )

    # Sanity check! Let's print the results
    print(latex_with_resize)

    # Writing the LaTeX table into a .tex file
    with open(TOURNAMENT_RESULTS_FILE_NAME, "w") as f:
        f.write(latex_with_resize)

    print("Tournament finished! 🏁")


if __name__ == '__main__':
    eval_mini_tournament()
