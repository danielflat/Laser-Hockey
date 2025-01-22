import logging
import numpy as np

from src.settings import BATCH_SIZE, BUFFER_SIZE, CHECKPOINT_ITER, MODEL_NAME, RENDER_MODE, \
    SAC_NUM_EPISODES_PER_TRAINING_EPOCH, \
    SAC_NUM_EPISODES_PER_VALIDATION_EPOCH, SAC_NUM_EPOCHS, \
    SAC_TRAIN_FREQ, SAC_VALIDATION_FREQ, SEED


def warmup_sac_hockey(
        env,
        memory,
        min_buffer_size: int = None,
        action_dim: int = 4
):
    """
    Fill the replay buffer with random transitions until it has >= min_buffer_size entries.
    To reduce overhead, we do local buffering and then push them in bulk.
    """
    if min_buffer_size is None:
        min_buffer_size = int(0.1 * BUFFER_SIZE)

    logging.info(f"Starting warm-up phase; collecting random transitions until memory >= {min_buffer_size}.")

    episodes = 0
    local_buffer = []

    while len(memory) < min_buffer_size:
        state, info = env.reset()
        state_opponent = env.obs_agent_two()
        done = False

        while not done:
            # Random actions for warmup
            a_agent = np.random.uniform(low = -1.0, high = 1.0, size = action_dim)
            a_opp = np.random.uniform(low = -1.0, high = 1.0, size = action_dim)
            joint_action = np.concatenate([a_agent, a_opp])

            next_state, reward, terminated, truncated, info = env.step(joint_action)
            done = terminated or truncated

            # Opponent agent "observations" if you want them
            next_state_opponent = env.obs_agent_two()
            reward_opponent = env.get_reward_agent_two(env.get_info_agent_two())

            # Store agent transition
            local_buffer.append((
                state.astype(np.float32),
                a_agent.astype(np.float32),
                float(reward),
                next_state.astype(np.float32),
                bool(done),
                info
            ))

            # Store opponent transition, if desired
            local_buffer.append((
                state_opponent.astype(np.float32),
                a_opp.astype(np.float32),
                float(reward_opponent),
                next_state_opponent.astype(np.float32),
                bool(done),
                info
            ))

            state = next_state
            state_opponent = next_state_opponent

            # Periodically flush to replay memory to reduce overhead
            if len(local_buffer) >= 1000:
                memory.push_batch(local_buffer)
                local_buffer.clear()

        episodes += 1

    # Flush any remainder
    if local_buffer:
        memory.push_batch(local_buffer)
        local_buffer.clear()

    logging.info(f"Warm-up done after {episodes} episodes. Replay buffer size: {len(memory)}")


def validate_sac_hockey(agent, val_env, num_episodes: int, seed_offset: int = 0):
    val_rewards = []
    val_wins = []

    for ep_i in range(num_episodes):
        # Optionally seed the environment so that each validation run is deterministic/reproducible
        state, info = val_env.reset(seed = seed_offset + ep_i)

        episode_reward = 0.0
        done = False

        while not done:
            # Just get the action for the agent; val_env handles the opponent internally
            action = agent.act(state)  # agent in "evaluation" mode if desired
            next_state, reward, terminated, truncated, info = val_env.step(action)

            episode_reward += reward
            done = terminated or truncated
            state = next_state

        val_rewards.append(episode_reward)

        # Check the winner from info. Suppose info["winner"] == 1 means agent won
        is_win = (info.get("winner", 0) == 1)
        val_wins.append(1 if is_win else 0)

    return val_rewards, val_wins


def do_sac_hockey_training(env, val_env, agent, memory):
    """
    Trains the SAC agent on the Hockey environment, using a simpler epoch structure.
    Periodically runs validation episodes in val_env (with an internal opponent)
    to track performance without affecting training.
    """

    # Step 1: Warm-up
    warmup_sac_hockey(env, memory)

    VAL_INTERVAL = 1  # Validate every 10 epochs (i.e., after every 10th epoch)

    logging.info("Starting SAC training with epochs...")

    all_rewards = []
    all_wins = []
    all_critic_losses = []
    all_actor_losses = []
    all_alpha_losses = []
    all_episode_steps = []

    # Arrays to store validation performance
    val_win_rates = []  # We'll store one win rate value for each validation
    val_avg_rewards = []  # (Optional) to store average validation rewards

    total_steps = 0
    total_episodes = 0

    for epoch_i in range(SAC_NUM_EPOCHS):
        epoch_rewards = []
        epoch_wins = []
        epoch_c_losses = []
        epoch_a_losses = []
        epoch_al_losses = []
        epoch_steps = []

        for ep_i in range(SAC_NUM_EPISODES_PER_TRAINING_EPOCH):
            total_episodes += 1
            state, info = env.reset(seed = SEED + total_episodes, one_starting = True)
            state_opponent = env.obs_agent_two()

            episode_done = False
            episode_reward = 0.0
            step_losses = []

            step_count = 0

            while not episode_done:
                if RENDER_MODE == "human":
                    env.render()

                # Agent acts on NumPy state
                a_agent = agent.act(state)
                a_opp = agent.act(state_opponent)
                joint_action = np.concatenate([a_agent, a_opp])

                next_state, reward, terminated, truncated, info = env.step(joint_action)
                reward_opp = env.get_reward_agent_two(env.get_info_agent_two())
                next_state_opp = env.obs_agent_two()

                done = terminated or truncated

                # Store transitions in replay
                memory.push(
                    state,
                    a_agent,
                    float(reward),
                    next_state,
                    done,
                    info
                )
                memory.push(
                    state_opponent,
                    a_opp,
                    float(reward_opp),
                    next_state_opp,
                    done,
                    info
                )

                # Call optimize() periodically
                if (total_steps % SAC_TRAIN_FREQ == 0) and (len(memory) >= 100 * BATCH_SIZE):
                    losses = agent.optimize(memory, episode_i = total_episodes)
                    step_losses.append(losses)

                episode_reward += reward
                step_count += 1
                total_steps += 1

                # Move on
                state = next_state
                state_opponent = next_state_opp
                episode_done = done

            epoch_steps.append(step_count)
            epoch_rewards.append(episode_reward)
            winner = (info["winner"] == 1)
            epoch_wins.append(1 if winner else 0)

            # If we had updates, average them for logging
            if len(step_losses) > 0:
                arr = np.array(step_losses)  # shape [num_updates, 3]
                mean_losses = arr.mean(axis = 0)  # [critic, actor, alpha]
            else:
                mean_losses = [0.0, 0.0, 0.0]
            epoch_c_losses.append(mean_losses[0])
            epoch_a_losses.append(mean_losses[1])
            epoch_al_losses.append(mean_losses[2])

        # Aggregate stats for this epoch
        epoch_avg_rew = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        epoch_win_rate = float(np.mean(epoch_wins))
        epoch_avg_steps = float(np.mean(epoch_steps)) if epoch_steps else 0.0
        epoch_avg_c = float(np.mean(epoch_c_losses))
        epoch_avg_a = float(np.mean(epoch_a_losses))
        epoch_avg_al = float(np.mean(epoch_al_losses))

        # Logging
        logging.info(
            f"[Epoch {epoch_i + 1}/{SAC_NUM_EPOCHS}] Eps={total_episodes}, Steps={total_steps}, "
            f"AvgRew={epoch_avg_rew:.2f}, WinRate={epoch_win_rate:.2f}, "
            f"CriticLoss={epoch_avg_c:.3f}, ActorLoss={epoch_avg_a:.3f}, AlphaLoss={epoch_avg_al:.3f}, "
            f"AvgDuration={epoch_avg_steps:.1f}"
        )

        # Store in big arrays
        all_rewards.extend(epoch_rewards)
        all_wins.extend(epoch_wins)
        all_critic_losses.extend(epoch_c_losses)
        all_actor_losses.extend(epoch_a_losses)
        all_alpha_losses.extend(epoch_al_losses)
        all_episode_steps.extend(epoch_steps)

        if (epoch_i + 1) % SAC_VALIDATION_FREQ == 0:
            # We can seed the validation environment differently if desired:
            val_rewards, val_wins_ = validate_sac_hockey(
                agent, val_env, num_episodes = SAC_NUM_EPISODES_PER_VALIDATION_EPOCH,
                seed_offset = SEED + total_episodes
            )
            val_mean_reward = float(np.mean(val_rewards)) if val_rewards else 0.0
            val_win_rate = float(np.mean(val_wins_)) if val_wins_ else 0.0

            val_win_rates.append(val_win_rate)
            val_avg_rewards.append(val_mean_reward)

            logging.info(
                f"[Validation after Epoch {epoch_i + 1}] "
                f"ValWinRate={val_win_rate:.2f}, ValAvgRew={val_mean_reward:.2f}"
            )
        # -----------------------------------------------------

        # Example checkpoint
        if (epoch_i + 1) % (CHECKPOINT_ITER) == 0:
            agent.saveModel(MODEL_NAME, total_episodes)

    logging.info("Finished SAC training!")

    # Return all arrays if desired
    return (all_rewards,
            all_wins,
            all_critic_losses,
            all_actor_losses,
            all_alpha_losses,
            all_episode_steps,
            val_win_rates,
            val_avg_rewards)
