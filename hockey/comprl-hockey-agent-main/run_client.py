from __future__ import annotations

import argparse
import torch

import hockey.hockey_env as h_env
import numpy as np

from comprl.client import Agent, launch_client

from src.agents.tdmpc2agent import TDMPC2Agent
from src.agents.mpoagent import MPOAgent
from src.settings import AGENT_SETTINGS, DEVICE, TD_MPC2_SETTINGS, MPO_SETTINGS
from src.util.constants import HOCKEY, TDMPC2_ALGO, MPO_ALGO
from src.util.contract import initAgent, initEnv


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak = weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        print(f"Game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class TDMPC2ServerAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.env = initEnv(use_env = HOCKEY, render_mode = None, number_discrete_actions = None, proxy_rewards = False)
        self.agent = initAgent(use_algo = TDMPC2_ALGO, env = self.env,
                               checkpoint_name=TD_MPC2_SETTINGS["CHECKPOINT_NAME"], device=DEVICE)
        # set the agent into eval mode
        self.agent.setMode(eval = True)

    def get_step(self, observation: list[float]) -> list[float]:
        state = torch.tensor(observation, dtype = torch.float32).to(DEVICE)
        action = self.agent.act(state)
        return action.tolist()  # tolist(), since the server requires a list[float]

    def on_start_game(self, game_id) -> None:
        self.agent.reset()
        print(f"Game started with ID: {game_id}")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}. Stats: {stats}"
        )
        
class MPOServerAgent(Agent):
    def __init__(self) -> None:
        super().__init__()
        self.env = initEnv(use_env = HOCKEY, render_mode = None, number_discrete_actions = None, proxy_rewards = False)
        self.agent = initAgent(use_algo = MPO_ALGO, env = self.env,
                               checkpoint_name = MPO_SETTINGS["CHECKPOINT_NAME"], device = DEVICE)
        self.agent.setMode(eval = True)

    def get_step(self, observation: list[float]) -> list[float]:
        state = torch.tensor(observation, dtype = torch.float32).to(DEVICE)
        action = self.agent.act(state)
        if isinstance(action, int):
            action = self.env.discrete_to_continous_action(action)
        return action  

    def on_start_game(self, game_id) -> None:
        self.agent.reset()
        print(f"Game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type = str,
        choices=["weak", "strong", "random", "mpo", "tdmpc2"],
        default = "weak",
        help = "Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak = True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak = False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "tdmpc2":
        agent = TDMPC2ServerAgent()
    elif args.agent == "mpo":
        agent = MPOServerAgent()
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
