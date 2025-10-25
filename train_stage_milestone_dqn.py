"""
Train with Staged Milestone Rewards + DQN Agent
Uses actual learning agents (not random!) with the staged reward system
"""

import numpy as np
import torch
from reward_staged_milestone import StagedMilestoneRewardFunction, StagedMilestoneRewardConfig
from training_config import get_config
from game_environment import FireWaterEnv
from example_dqn import DQNAgent
from collections import deque
import os


def train_dqn_with_staged_rewards(use_wandb=False, wandb_project="firewater-staged-dqn", resume_episode=None):
    """Train DQN agents using staged milestone reward function
    
    Args:
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        resume_episode: Episode number to resume from (e.g., 400 to load ep400 checkpoint)
    """

    print("=" * 60)
    print("TRAINING DQN WITH STAGED MILESTONE REWARDS")
    if resume_episode:
        print(f"RESUMING FROM EPISODE {resume_episode}")
    print("=" * 60)
    print("\nReward Structure:")
    print("  Stage 0: Navigate to plates -> Progress + beta bonus")
    print("  Stage 1: Navigate to exits  -> Progress + gamma bonus")
    print("  Stage 2: Done (success)")
    print("\nAgent Type: DQN with Dueling Architecture")
    print("Learning: ENABLED (not random!)")
    print("=" * 60 + "\n")

    # ========================================================================
    # CONFIGURE TRAINING
    # ========================================================================

    # Training parameters
    num_episodes = 10000
    max_steps_per_episode = 3000
    learning_rate = 3e-4
    batch_size = 64

    # Logging - IMPROVED: More frequent for better visibility
    log_frequency = 10          # Log every 10 episodes (was 100)
    save_frequency = 100        # Save every 100 episodes (was 500)
    eval_frequency = 500        # Eval every 500 episodes (was 1000)
    print_every_episode = True  # Print progress every single episode

    print("Training Configuration:")
    print(f"  Episodes: {num_episodes}")
    print(f"  Max steps: {max_steps_per_episode}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Log frequency: Every {log_frequency} episodes")
    print(f"  Save frequency: Every {save_frequency} episodes")
    print(f"  Progress updates: Every episode\n")

    # ========================================================================
    # CONFIGURE STAGED MILESTONE REWARDS
    # ========================================================================

    reward_config = StagedMilestoneRewardConfig()

    # TUNABLE PARAMETERS
    reward_config.beta = 50.0       # Bonus for reaching plates
    reward_config.gamma = 200.0     # Bonus for reaching exits
    reward_config.diam_0 = 1000.0   # Distance normalization (stage 0)
    reward_config.diam_1 = 1000.0   # Distance normalization (stage 1)
    reward_config.rho_p = 20.0      # Plate detection radius
    reward_config.rho_e = 20.0      # Exit detection radius
    reward_config.step_penalty = -0.1

    print("Reward Configuration:")
    print(f"  beta (plate bonus): {reward_config.beta}")
    print(f"  gamma (finish bonus): {reward_config.gamma}")
    print(f"  diam_0: {reward_config.diam_0}")
    print(f"  diam_1: {reward_config.diam_1}")
    print(f"  rho_p: {reward_config.rho_p}")
    print(f"  rho_e: {reward_config.rho_e}")
    print(f"  Step penalty: {reward_config.step_penalty}\n")

    # ========================================================================
    # INITIALIZE W&B LOGGING (OPTIONAL)
    # ========================================================================

    if use_wandb:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    "num_episodes": num_episodes,
                    "max_steps": max_steps_per_episode,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "beta": reward_config.beta,
                    "gamma": reward_config.gamma,
                    "diam_0": reward_config.diam_0,
                    "diam_1": reward_config.diam_1,
                    "step_penalty": reward_config.step_penalty,
                    "agent": "DQN",
                    "reward_function": "staged_milestone",
                    "log_frequency": log_frequency,
                    "save_frequency": save_frequency
                }
            )
            print("Weights & Biases logging enabled")
            print(f"  Project: {wandb_project}")
            print(f"  Run: {wandb.run.name}")
            print(f"  URL: {wandb.run.url}\n")
        except ImportError:
            print("Warning: wandb not installed. Disabling W&B logging.")
            print("  Install with: pip install wandb\n")
            use_wandb = False
        except Exception as e:
            print(f"Warning: wandb init failed: {e}")
            print("  Continuing without W&B logging\n")
            use_wandb = False
    else:
        print("Weights & Biases logging: DISABLED")
        print("  (Enable with --wandb flag)\n")

    # ========================================================================
    # CREATE ENVIRONMENT WITH STAGED REWARDS
    # ========================================================================

    env = FireWaterEnv(max_steps=max_steps_per_episode)
    reward_fn = StagedMilestoneRewardFunction(reward_config)

    # Override environment's reward calculation
    def staged_rewards_wrapper(fire_won, water_won, fire_action, water_action):
        return reward_fn.calculate_rewards(
            env.fire_agent,
            env.water_agent,
            fire_action,
            water_action,
            env.fire_died,
            env.water_died,
            fire_won,
            water_won,
            env.bridge_activated,
            env.gate_activated,
            env.level.get_exits(),
            env.level.get_hazards()
        )

    env._calculate_rewards = staged_rewards_wrapper
    env.reward_fn = reward_fn

    # ========================================================================
    # CREATE DQN AGENTS (REAL LEARNING AGENTS!)
    # ========================================================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("\nCreating DQN agents...")
    fire_agent = DQNAgent(
        state_dim=52,
        action_dim=6,
        learning_rate=learning_rate,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=batch_size,
        device=device
    )

    water_agent = DQNAgent(
        state_dim=52,
        action_dim=6,
        learning_rate=learning_rate,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        batch_size=batch_size,
        device=device
    )

    print(f"  Fire agent: DQN (epsilon={fire_agent.epsilon:.3f})")
    print(f"  Water agent: DQN (epsilon={water_agent.epsilon:.3f})")
    print("  Learning: ENABLED\n")

    # ========================================================================
    # LOAD CHECKPOINT IF RESUMING
    # ========================================================================

    start_episode = 0
    if resume_episode is not None:
        print("=" * 60)
        print(f"LOADING CHECKPOINT FROM EPISODE {resume_episode}")
        print("=" * 60)
        
        fire_checkpoint_path = f"checkpoints/fire_staged_dqn_ep{resume_episode}.pth"
        water_checkpoint_path = f"checkpoints/water_staged_dqn_ep{resume_episode}.pth"
        
        # Check if checkpoint files exist
        if not os.path.exists(fire_checkpoint_path):
            print(f"ERROR: Fire checkpoint not found: {fire_checkpoint_path}")
            print("Available checkpoints:")
            if os.path.exists("checkpoints"):
                checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("fire_staged_dqn_ep")]
                for cp in sorted(checkpoints):
                    print(f"  - {cp}")
            return
        
        if not os.path.exists(water_checkpoint_path):
            print(f"ERROR: Water checkpoint not found: {water_checkpoint_path}")
            print("Available checkpoints:")
            if os.path.exists("checkpoints"):
                checkpoints = [f for f in os.listdir("checkpoints") if f.startswith("water_staged_dqn_ep")]
                for cp in sorted(checkpoints):
                    print(f"  - {cp}")
            return
        
        # Load checkpoints
        print(f"Loading fire agent from: {fire_checkpoint_path}")
        fire_agent.load(fire_checkpoint_path)
        
        print(f"Loading water agent from: {water_checkpoint_path}")
        water_agent.load(water_checkpoint_path)
        
        # Set starting episode
        start_episode = resume_episode
        
        print("\nCheckpoint loaded successfully!")
        print(f"  Fire epsilon: {fire_agent.epsilon:.3f}")
        print(f"  Water epsilon: {water_agent.epsilon:.3f}")
        print(f"  Fire steps: {fire_agent.steps}")
        print(f"  Fire updates: {fire_agent.updates}")
        print(f"  Resuming from episode: {start_episode + 1}")
        print("=" * 60 + "\n")

    # ========================================================================
    # TRAINING LOOP
    # ========================================================================

    print("Starting training...")
    print("=" * 60)
    print("Progress: [Episode] Reward | Length | Success | Stage | Time")
    print("=" * 60 + "\n")

    # Tracking
    episode_rewards = deque(maxlen=100)
    success_tracker = deque(maxlen=100)
    stage_1_reached = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    episode_times = deque(maxlen=100)

    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    # Training timing
    import time
    from datetime import timedelta
    training_start = time.time()
    last_log_time = training_start

    for episode in range(start_episode, num_episodes):
        episode_start = time.time()
        fire_obs, water_obs = env.reset()
        reward_fn.reset()  # Reset stage tracking

        episode_reward = 0
        episode_length = 0
        reached_stage_1 = False

        done = False
        while not done:
            # Select actions (with exploration)
            fire_action = fire_agent.select_action(fire_obs, training=True)
            water_action = water_agent.select_action(water_obs, training=True)

            # Step environment
            (fire_next_obs, water_next_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
                fire_action, water_action
            )

            # Track stage progression
            if reward_fn.stage >= 1:
                reached_stage_1 = True

            # Store experiences in replay buffer
            fire_agent.store_experience(fire_obs, fire_action, fire_reward, fire_next_obs, fire_done)
            water_agent.store_experience(water_obs, water_action, water_reward, water_next_obs, water_done)

            # Update agents (THIS IS WHERE LEARNING HAPPENS!)
            fire_agent.update()
            water_agent.update()

            # Track metrics
            episode_reward += fire_reward + water_reward
            episode_length += 1

            # Update observations
            fire_obs = fire_next_obs
            water_obs = water_next_obs
            done = fire_done or water_done

        # Episode complete - track metrics
        episode_time = time.time() - episode_start
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_times.append(episode_time)
        success_tracker.append(1 if info.get('both_won', False) else 0)
        stage_1_reached.append(1 if reached_stage_1 else 0)

        # Print progress every episode
        if print_every_episode:
            status = "WIN" if info.get('both_won', False) else "   "
            stage_str = f"S{reward_fn.stage}"
            print(f"[{episode+1:5d}] R:{episode_reward:7.2f} | L:{episode_length:4d} | {status} | {stage_str} | {episode_time:.1f}s")

        # Detailed logging
        if (episode + 1) % log_frequency == 0:
            current_time = time.time()
            time_since_last_log = current_time - last_log_time
            last_log_time = current_time
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_time = np.mean(episode_times)
            success_rate = np.mean(success_tracker)
            stage_1_rate = np.mean(stage_1_reached)

            elapsed = current_time - training_start
            episodes_per_sec = (episode + 1) / elapsed
            remaining_episodes = num_episodes - (episode + 1)
            eta_seconds = remaining_episodes / episodes_per_sec if episodes_per_sec > 0 else 0
            eta = timedelta(seconds=int(eta_seconds))

            print("\n" + "-" * 60)
            print(f"SUMMARY @ Episode {episode + 1}/{num_episodes}")
            print("-" * 60)
            print(f"  Avg Reward:        {avg_reward:8.2f}")
            print(f"  Avg Length:        {avg_length:8.0f} steps")
            print(f"  Success Rate:      {success_rate:7.1%}")
            print(f"  Stage 1 Rate:      {stage_1_rate:7.1%}")
            print(f"  Fire Epsilon:      {fire_agent.epsilon:8.3f}")
            print(f"  Water Epsilon:     {water_agent.epsilon:8.3f}")
            print(f"  Buffer Size:       {len(fire_agent.replay_buffer):8d}")
            print(f"  Avg Episode Time:  {avg_time:8.2f}s")
            print(f"  Time Elapsed:      {timedelta(seconds=int(elapsed))}")
            print(f"  ETA:               {eta}")
            print("-" * 60 + "\n")

            # Log to W&B
            if use_wandb:
                wandb.log({
                    "episode": episode + 1,
                    "avg_reward": avg_reward,
                    "avg_episode_length": avg_length,
                    "success_rate": success_rate,
                    "stage_1_reached_rate": stage_1_rate,
                    "fire_epsilon": fire_agent.epsilon,
                    "water_epsilon": water_agent.epsilon,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "episode_time": episode_time,
                    "avg_episode_time": avg_time,
                    "buffer_size": len(fire_agent.replay_buffer),
                    "stage": reward_fn.stage,
                    "elapsed_time": elapsed,
                    "eta_seconds": eta_seconds
                })

        # Save checkpoints
        if (episode + 1) % save_frequency == 0:
            checkpoint_start = time.time()
            fire_agent.save(f"checkpoints/fire_staged_dqn_ep{episode+1}.pth")
            water_agent.save(f"checkpoints/water_staged_dqn_ep{episode+1}.pth")
            checkpoint_time = time.time() - checkpoint_start
            print(f"\n*** CHECKPOINT SAVED @ Episode {episode + 1} ({checkpoint_time:.1f}s) ***\n")

    # ========================================================================
    # TRAINING COMPLETE
    # ========================================================================

    total_training_time = time.time() - training_start

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Episodes:     {num_episodes}")
    print(f"Total Time:         {timedelta(seconds=int(total_training_time))}")
    print(f"Final Success Rate: {np.mean(success_tracker):.2%}")
    print(f"Final Stage 1 Rate: {np.mean(stage_1_reached):.2%}")
    print(f"Final Epsilon:      {fire_agent.epsilon:.3f}")
    print(f"Buffer Size:        {len(fire_agent.replay_buffer)}")
    print("\nModels saved to: checkpoints/")
    print("  - fire_staged_dqn_ep*.pth")
    print("  - water_staged_dqn_ep*.pth")

    # Save final models
    fire_agent.save("checkpoints/fire_final.pth")
    water_agent.save("checkpoints/water_final.pth")
    print("\nFinal models:")
    print("  - checkpoints/fire_final.pth")
    print("  - checkpoints/water_final.pth")

    print("\nTo visualize trained agents:")
    print("  python visualize_staged_agents.py checkpoints/fire_final.pth")
    print("=" * 60)

    # Finish W&B logging
    if use_wandb:
        wandb.finish()
        print("\nW&B run finished")


if __name__ == "__main__":
    import sys
    import argparse

    # Check for PyTorch
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch not installed!")
        print("\nInstall with:")
        print("  pip install torch")
        sys.exit(1)

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train DQN agents with staged milestone rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start fresh training
  python train_stage_milestone_dqn.py
  
  # Start training with W&B logging
  python train_stage_milestone_dqn.py --wandb
  
  # Resume from episode 400
  python train_stage_milestone_dqn.py --resume 400
  
  # Resume with W&B logging
  python train_stage_milestone_dqn.py --resume 400 --wandb
        """
    )
    
    parser.add_argument(
        "--wandb", "-w",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--resume", "-r",
        type=int,
        metavar="EPISODE",
        help="Resume training from specified episode checkpoint (e.g., 400 to load ep400 checkpoint)"
    )
    
    parser.add_argument(
        "--project",
        type=str,
        default="firewater-staged-dqn",
        help="W&B project name (default: firewater-staged-dqn)"
    )
    
    args = parser.parse_args()
    
    # Display configuration
    print("\n" + "=" * 60)
    print("TRAINING CONFIGURATION")
    print("=" * 60)
    print(f"  W&B Logging: {'ENABLED' if args.wandb else 'DISABLED'}")
    if args.wandb:
        print(f"  W&B Project: {args.project}")
    print(f"  Resume Mode: {'YES (Episode ' + str(args.resume) + ')' if args.resume else 'NO (Fresh Start)'}")
    print("=" * 60 + "\n")

    # Run training
    train_dqn_with_staged_rewards(
        use_wandb=args.wandb,
        wandb_project=args.project,
        resume_episode=args.resume
    )
