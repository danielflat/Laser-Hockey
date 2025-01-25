import logging
import numpy as np

import random
from itertools import count

import time
import torch

from src.agent import Agent
from src.replaymemory import ReplayMemory
from src.settings import AGENT_SETTINGS, BATTLE_STATISTICS_FREQUENCY, CHECKPOINT_ITER, DDPG_SETTINGS, DEVICE, \
    DQN_SETTINGS, EPISODE_UPDATE_ITER, \
    MAIN_SETTINGS, \
    MODEL_NAME, MPO_SETTINGS, \
    NUM_TRAINING_EPISODES, PLOT_FREQUENCY, PPO_SETTINGS, \
    RENDER_MODE, SAC_SETTINGS, \
    SEED, SELF_PLAY, SELF_PLAY_FREQUENCY, SELF_PLAY_UPDATE_FREQUENCY, SETTINGS, \
    SHOW_PLOTS, TD3_SETTINGS, TD_MPC2_SETTINGS, USE_ALGO, \
    CURIOSITY, BATCH_SIZE, EPISODE_UPDATE_ITER, MODEL_NAME, NUM_TRAINING_EPISODES, RENDER_MODE, SEED
from src.util.constants import DDPG_ALGO, DQN_ALGO, HOCKEY, MPO_ALGO, PPO_ALGO, RANDOM_ALGO, SAC_ALGO, STRONG_COMP_ALGO, \
    TD3_ALGO, TDMPC2_ALGO, WEAK_COMP_ALGO, MPO_ALGO
from src.util.contract import initAgent, initEnv, initValEnv, initSeed, setupLogging
from src.util.plotutil import plot_training_metrics, plot_mpo_training_metrics


def do_mpo_hockey_training(env, val_env, agent, memory, opponent_pool: dict, self_opponent = Agent):
    """
    TODO: Add support for discrete action space 
    """
    
    logging.info("Starting MPO training!")

    all_rewards       = []
    all_wins          = []
    all_steps         = []
    all_critic_losses = []
    all_actor_losses  = []
    all_kl_µ          = []
    all_kl_Σ          = []
    
    val_opponent_metrics = []  #Storing opponent metrics gained from validation fkt

    total_steps = 0
    total_episodes = 0

    for i_episode in range(1, NUM_TRAINING_EPISODES + 1):
        t_start = time.time()
        total_reward = 0
        total_episodes += 1
        
        #Select self opponent in half the games
        if self_opponent is not None and i_episode % SELF_PLAY_FREQUENCY != 0:
            opponent = self_opponent
            opponent_name = USE_ALGO
        else:
            opponent = random.choice(list(opponent_pool.values()))
        
        # Reset the environment with starting condition and get first states
        if i_episode % 2:
            state, info = env.reset(seed=SEED + i_episode, one_starting=False)
        else:
            state, info = env.reset(seed=SEED + i_episode, one_starting=True)

        state_opponent = env.obs_agent_two()
        
        # Convert states to tensors
        state = torch.tensor(state, device = DEVICE, dtype = torch.float32)
        state_opponent = torch.tensor(state_opponent, device = DEVICE, dtype = torch.float32)
        
        # We start an Episode!
        for step in count(start = 1):
            # Rendering?
            env.render(mode = RENDER_MODE)

            # Agent and opponent act
            action_agent = agent.act(state) #Numpy array
            action_opp   = opponent.act(state_opponent) #Numpy array
            
            # We do a step in the env
            next_state, reward, terminated, truncated, info = env.step(np.concatenate([action_agent, action_opp]))
            next_state_opp = env.obs_agent_two()
            done = terminated or truncated
            
            # Some reward adjustments
            if info["winner"] == 1:
                reward = 10
            elif info["winner"] == -1:
                reward = -10
            elif info["winner"] == 0 and done:
                reward = -5
            
            if CURIOSITY is not None:
                reward += CURIOSITY * agent.icm.compute_intrinsic_reward(state, action_agent, next_state)
            
            # Tracking rewards
            total_reward += reward
            
            # Convert quantities into tensors
            action_agent = torch.tensor(action_agent, device = DEVICE, dtype = torch.float32)
            reward = torch.tensor(reward, device = DEVICE, dtype = torch.float32)
            done = torch.tensor(terminated or truncated, device = DEVICE, dtype = torch.int) 
            next_state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)
            next_state_opp = torch.tensor(next_state_opp, device = DEVICE, dtype = torch.float32)

            # Store transitions in replay buffer
            memory.push(state, action_agent, reward, next_state, done, info)

            #Update the states of agent and opponent
            state = next_state
            state_opponent = next_state_opp
            
            # Done with the episode
            if done:
                break
        
        # Saving statistics
        all_rewards.append(total_reward)
        all_steps.append(step)
        winner = (info["winner"] == 1)
        all_wins.append(1 if winner else 0)
        win_rate = sum(all_wins) / len(all_wins)
        
        # After some episodes, we optimize
        if (i_episode % EPISODE_UPDATE_ITER == 0) and (len(memory) >= 100 * BATCH_SIZE):
            losses = agent.optimize(memory, episode_i=total_episodes)
            
            # And we log the losses in the episode
            logging.info(
                f"[Episode {i_episode}/{NUM_TRAINING_EPISODES}] Steps={step}, "
                f"Reward={total_reward:.2f}, WinRate={win_rate:.2f}, "
                f"CriticLoss={losses[0]:.3f}, ActorLoss={losses[1]:.3f}, "
                f"KL_µ={losses[2]:.3f}, KL_Σ={losses[3]:.3f}"
            )
            # Saving these losses and lagrangians
            all_critic_losses.append(losses[0])
            all_actor_losses.append(losses[1])
            all_kl_µ.append(losses[2])
            all_kl_Σ.append(losses[3])
        
        # Update the self-opponent with the current agent weights
        if SELF_PLAY and i_episode % SELF_PLAY_UPDATE_FREQUENCY == 0:
            self_opponent.import_checkpoint(agent.export_checkpoint())
        
        # Validation, checkpointing and plotting
        if (i_episode + 1) % (CHECKPOINT_ITER) == 0 and (len(memory) >= 100 * BATCH_SIZE):
            assert CHECKPOINT_ITER % EPISODE_UPDATE_ITER == 0, "Checkpointing freq should be a multiple of update freq!"
            
            # Validation
            opponent_metrics = validate_mpo_hockey(agent, val_env, opponent_pool, num_episodes=100, 
                                                   seed_offset=SEED + total_episodes)
            val_opponent_metrics.append(opponent_metrics)
            
            # Saving the model
            agent.saveModel(MODEL_NAME, total_episodes)
            
        # Plotting
        if SHOW_PLOTS and (i_episode + 1) % PLOT_FREQUENCY == 0 and (len(memory) >= 100 * BATCH_SIZE):
            assert PLOT_FREQUENCY >= CHECKPOINT_ITER, "Plotting freq should be larger than checkpointing freq!"
            assert PLOT_FREQUENCY >= EPISODE_UPDATE_ITER, "Plotting freq should be larger than update freq!"
            
            plot_mpo_training_metrics(all_critic_losses, all_actor_losses, all_kl_µ, all_kl_Σ, val_opponent_metrics)

    logging.info("Finished MPO training!")
    
    # Returning all the stats if desired
    return all_rewards, all_wins, all_critic_losses, all_actor_losses, all_kl_µ, all_kl_Σ, val_opponent_metrics

def validate_mpo_hockey(agent, val_env, opponent_pool: dict, num_episodes: int, seed_offset: int = 0):
    """
    Validation function to test the agent
    Returns rewards, wins, draws, and losses for each opponent in opponent pool.
    """
    opponent_metrics = {
        opponent_name: {"wins": 0, "draws": 0, "losses": 0, "rewards": []}
        for opponent_name in opponent_pool.keys()
    }

    # Play num_episodes against each opponent
    for opponent_name, opponent in opponent_pool.items():
        for ep_i in range(num_episodes):
            # Reset the environment
            state, info = val_env.reset(seed=seed_offset + ep_i)
            state_opp = val_env.obs_agent_two()
            
            # Convert states to tensors
            state = torch.tensor(state, device=DEVICE, dtype=torch.float32)
            state_opp = torch.tensor(state_opp, device = DEVICE, dtype = torch.float32)
            
            episode_reward = 0.0
            done = False

            while not done:
                # Agent and opponent act
                action = agent.act(state)
                opponent_action = opponent.act(state_opp)

                # Step the environment
                next_state, reward, terminated, truncated, info = val_env.step(np.concatenate([action, opponent_action]))
                next_state_opp = val_env.obs_agent_two()
                done = terminated or truncated
                
                #tracking rewards
                episode_reward += reward
                
                next_state = torch.tensor(next_state, device = DEVICE, dtype = torch.float32)
                next_state_opp = torch.tensor(next_state_opp, device = DEVICE, dtype = torch.float32)
                
                state = next_state
                state_opp = next_state_opp

            # Update metrics
            opponent_metrics[opponent_name]["rewards"].append(episode_reward)
            if info.get("winner", 0) == 1:  
                opponent_metrics[opponent_name]["wins"] += 1
            elif info.get("winner", 0) == -1: 
                opponent_metrics[opponent_name]["losses"] += 1
            else: 
                opponent_metrics[opponent_name]["draws"] += 1

    # Calculate win/draw/losing rates
    for opponent_name, metrics in opponent_metrics.items():
        total_games = num_episodes
        metrics["win_rate"] = metrics["wins"] / total_games
        metrics["draw_rate"] = metrics["draws"] / total_games
        metrics["loss_rate"] = metrics["losses"] / total_games

    # Logging the results
    for opponent_name, metrics in opponent_metrics.items():
        logging.info(
            f"[Validation vs {opponent_name}] "
            f"WinRate={metrics['win_rate']:.2f}, DrawRate={metrics['draw_rate']:.2f}, LossRate={metrics['loss_rate']:.2f}"
        )

    return opponent_metrics

