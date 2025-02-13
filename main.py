import copy
import logging
import torch
import yaml

from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, BUFFER_SIZE, CHECKPOINT_NAME, DDPG_SETTINGS, DEVICE, DQN_SETTINGS, \
    MODEL_NAME, MPO_SETTINGS, NUMBER_DISCRETE_ACTIONS, PROXY_REWARDS, RENDER_MODE, SAC_NUM_EPISODES_PER_TRAINING_EPOCH, \
    SAC_VALIDATION_FREQ, SEED, SELF_PLAY, SETTINGS, \
    TD_MPC2_SETTINGS, USE_ALGO, USE_ENV, USE_TF32
from src.training_loops.mpo_training import do_mpo_hockey_training
from src.training_loops.other_algos_training import do_hockey_testing, do_hockey_training, do_other_env_testing, \
    do_other_env_training
from src.training_loops.sac_training import do_sac_hockey_training
from src.training_loops.tdmpc2_training import do_tdmpc2_hockey_training, do_tdmpc2agent_other_env_training
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, RANDOM_ALGO, SAC_ALGO, STRONG_COMP_ALGO, \
    TDMPC2_ALGO, WEAK_COMP_ALGO
from src.util.contract import initAgent, initEnv, initSeed, initValEnv, setupLogging
from src.util.plotutil import plot_sac_training_metrics, plot_sac_validation_metrics

"""
This is the main file of this project.
Here, you can find the main training loop.
In order to set the parameters for training, you can change the values in the settings.py file.

Author: Daniel Flat
"""

def main():
    # Let's first set the seed
    initSeed(seed = SEED, device = DEVICE)

    # If you want to use TF32 instead of Float 32, you can activate it here. Might not be available for old GPUs
    if USE_TF32:
        torch.set_float32_matmul_precision("high")

    # Setup Logging
    setupLogging(model_name = MODEL_NAME)

    # Log the settings.py such that we can save the settings under which we did the training
    logging.info(yaml.dump(SETTINGS, default_flow_style = False, sort_keys = False, allow_unicode = True))

    # Initialize the environment
    env = initEnv(USE_ENV, RENDER_MODE, NUMBER_DISCRETE_ACTIONS, PROXY_REWARDS)

    if USE_ENV == HOCKEY:
        val_env = initValEnv()

    # Choose which algorithm to pick to initialize the agent
    agent = initAgent(USE_ALGO, env = env, device = DEVICE, agent_settings = AGENT_SETTINGS,
                      checkpoint_name = CHECKPOINT_NAME)

    # Init the memory
    memory = ReplayMemory(capacity = BUFFER_SIZE, device = DEVICE)

    # Training loop
    agent.setMode(eval = False)  # Set the agent in training mode
    logging.info(f"The configuration was valid! Start training 💪")

    opponent_pool = None

    # If we play Hockey, our training loop is different
    if USE_ENV == HOCKEY:
        # Only in the Hockey env, we need some opponent_pool to play against
        random_agent = initAgent(use_algo = RANDOM_ALGO, env = env, device = DEVICE, checkpoint_name = None)
        weak_comp_agent = initAgent(use_algo = WEAK_COMP_ALGO, env = env, device = DEVICE, checkpoint_name = None)
        strong_comp_agent = initAgent(use_algo = STRONG_COMP_ALGO, env = env, device = DEVICE, checkpoint_name = None)
        # dqn_agent = initAgent(use_algo = DQN_ALGO, env = env, device = DEVICE,
        #                      checkpoint_name = DQN_SETTINGS["CHECKPOINT_NAME"])
        # ppo_agent = initAgent(use_algo = PPO_ALGO, env = env, device = DEVICE, checkpoint_name = PPO_SETTINGS["CHECKPOINT_NAME"])
        ddpg_agent = initAgent(use_algo = DDPG_ALGO, env = env, device = DEVICE,
                               checkpoint_name = DDPG_SETTINGS["CHECKPOINT_NAME"])
        # td3_agent = initAgent(use_algo = TD3_ALGO, env = env, device = DEVICE, checkpoint_name = TD3_SETTINGS["CHECKPOINT_NAME"])
        # sac_agent = initAgent(use_algo = SAC_ALGO, env = env, device = DEVICE, checkpoint_name = SAC_SETTINGS["CHECKPOINT_NAME"])
        mpo_agent = initAgent(use_algo=MPO_ALGO, env=env, device=DEVICE,
                              checkpoint_name=MPO_SETTINGS["CHECKPOINT_NAME"])
        tdmpc2_agent = initAgent(use_algo=TDMPC2_ALGO, env=env, device=DEVICE,
                                 checkpoint_name=TD_MPC2_SETTINGS["CHECKPOINT_NAME"])

        # Currently, we do not allow the opponent networks to train as well. This might be an extra feature
        random_agent.setMode(eval = True)
        weak_comp_agent.setMode(eval=True)
        strong_comp_agent.setMode(eval=True)
        # dqn_agent.setMode(eval = True)
        # ppo_agent.setMode(eval = True)
        ddpg_agent.setMode(eval=True)
        # td3_agent.setMode(eval = True)
        # sac_agent.setMode(eval = True)
        mpo_agent.setMode(eval=True)
        tdmpc2_agent.setMode(eval=True)

        opponent_pool = {
            # RANDOM_ALGO: random_agent,
            # WEAK_COMP_ALGO: weak_comp_agent,
            # STRONG_COMP_ALGO: strong_comp_agent,
            #DQN_ALGO: dqn_agent,
            # PPO_ALGO: ppo_agent,
            #f"{DDPG_ALGO}_Checkpoint": ddpg_agent,
            # f"{TD3_ALGO}_Checkpoint": td3_agent,
            # f"{SAC_ALGO}_Checkpoint": sac_agent,
            f"{MPO_ALGO}_Checkpoint": mpo_agent,
            #f"{TDMPC2_ALGO}_Checkpoint": tdmpc2_agent,
        }

        # if you want to use self-play, we have to init the self opponent agent
        self_opponent = None
        if SELF_PLAY:
            # First, create a copy of the main agent
            self_opponent = initAgent(use_algo = USE_ALGO, env = env, device = DEVICE, checkpoint_name = None)
            self_opponent.setMode(eval = True)
            self_opponent.import_checkpoint(agent.export_checkpoint())

        if USE_ALGO == TDMPC2_ALGO or USE_ALGO == DDPG_ALGO:
            do_tdmpc2_hockey_training(env = env, agent = agent, memory = memory,
                                      opponent_pool = copy.deepcopy(opponent_pool),
                                      self_opponent = self_opponent)
        elif USE_ALGO == SAC_ALGO:
            (
                all_rewards,
                all_wins,
                all_critic_losses,
                all_actor_losses,
                all_alpha_losses,
                all_episode_steps,
                val_win_rates,
                val_avg_rewards
            ) = do_sac_hockey_training(env, val_env, agent, memory)

            plot_sac_training_metrics(
                all_rewards,
                all_wins,
                all_critic_losses,
                all_actor_losses,
                all_alpha_losses,
                episodes_per_epoch = SAC_NUM_EPISODES_PER_TRAINING_EPOCH,
                save = True
            )

            plot_sac_validation_metrics(
                val_win_rates,
                val_avg_rewards,
                val_interval = SAC_VALIDATION_FREQ,
                save = True
            )

        elif USE_ALGO == MPO_ALGO:
            do_mpo_hockey_training(env = env, val_env = val_env, agent = agent, memory = memory,
                                   opponent_pool = copy.deepcopy(opponent_pool),
                                   self_opponent = self_opponent)
        else:
            do_hockey_training(env = env, agent = agent, memory = memory, opponent_pool = copy.deepcopy(opponent_pool))

    # If you use another env (e.g. Pendulum), train normally
    else:
        # df: TDMPC2 and DDPG have a separate training loop because they collect whole episodes before saving them into the buffer
        if USE_ALGO == TDMPC2_ALGO or USE_ALGO == DDPG_ALGO:
            do_tdmpc2agent_other_env_training(env = env, agent = agent, memory = memory)
        else:
            do_other_env_training(env = env, agent = agent, memory = memory)

    # Testing loop
    logging.info("Training is done! Now we will do some testing!")
    agent.setMode(eval = True)  # Set the agent in eval mode

    if USE_ALGO is not SAC_ALGO:
        if USE_ENV == HOCKEY:
            if SELF_PLAY:
                opponent_pool[USE_ALGO] = copy.deepcopy(agent)
            do_hockey_testing(env = env, agent = agent, opponent_pool = copy.deepcopy(opponent_pool))
        else:
            do_other_env_testing(env = env, agent = agent)
    logging.info(f"Finished! 🚀")


if __name__ == '__main__':
    main()
