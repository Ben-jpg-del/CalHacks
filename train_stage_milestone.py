"""
Train with Staged Milestone Rewards
Uses the mathematical staged reward specification
"""

import numpy as np
from reward_staged_milestone import StagedMilestoneRewardFunction, StagedMilestoneRewardConfig
from train_modular import ModularTrainer
from training_config import get_config
from game_environment import FireWaterEnv


def train_with_staged_rewards():
    """Train agents using staged milestone reward function"""
    
    print("=" * 60)
    print("TRAINING WITH STAGED MILESTONE REWARDS")
    print("=" * 60)
    print("\nReward Structure:")
    print("  Stage 0: Navigate to plates -> Progress + beta bonus")
    print("  Stage 1: Navigate to exits  -> Progress + gamma bonus")
    print("  Stage 2: Done (success)")
    print("\nReward Components:")
    print("  r_prog(t) = max(D_t - D_{t+1}, 0)  // Only positive when closer")
    print("  r_plates(t) = beta * IND[stage 0->1]   // Plate activation")
    print("  r_finish(t) = gamma * IND[stage 1->2]   // Mission complete")
    print("=" * 60 + "\n")
    
    # ========================================================================
    # CONFIGURE TRAINING
    # ========================================================================
    
    training_config = get_config('standard')
    
    # Training parameters
    training_config.num_episodes = 10000
    training_config.max_steps_per_episode = 3000
    training_config.learning_rate = 3e-4
    training_config.batch_size = 64
    
    # Logging
    training_config.log_frequency = 100
    training_config.save_frequency = 500
    training_config.eval_frequency = 1000
    training_config.use_wandb = False  # Set True to enable W&B
    
    # ========================================================================
    # CONFIGURE STAGED MILESTONE REWARDS
    # ========================================================================
    
    reward_config = StagedMilestoneRewardConfig()
    
    # TUNABLE PARAMETERS - Adjust these for your task!
    reward_config.beta = 50.0       # β: Bonus for reaching plates
    reward_config.gamma = 200.0     # Γ: Bonus for reaching exits
    reward_config.diam_0 = 1000.0   # Distance normalization (stage 0)
    reward_config.diam_1 = 1000.0   # Distance normalization (stage 1)
    reward_config.rho_p = 20.0      # Plate detection radius (pixels)
    reward_config.rho_e = 20.0      # Exit detection radius (pixels)
    reward_config.step_penalty = -0.1  # Small step penalty
    
    print("Reward Configuration:")
    print(f"  beta (plate bonus): {reward_config.beta}")
    print(f"  gamma (finish bonus): {reward_config.gamma}")
    print(f"  diam_0: {reward_config.diam_0}")
    print(f"  diam_1: {reward_config.diam_1}")
    print(f"  rho_p: {reward_config.rho_p}")
    print(f"  rho_e: {reward_config.rho_e}")
    print(f"  Step penalty: {reward_config.step_penalty}\n")
    
    # ========================================================================
    # CREATE CUSTOM ENVIRONMENT WITH STAGED REWARDS
    # ========================================================================
    
    # Create environment
    env = FireWaterEnv(max_steps=training_config.max_steps_per_episode)
    
    # Inject staged milestone reward function
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
    # CREATE AGENTS
    # ========================================================================
    
    # Option 1: Use random agents (for testing)
    print("Using random agents for testing...")
    print("(Replace with your RL agents for real training)\n")
    
    from train_modular import RandomAgent
    fire_agent = RandomAgent()
    water_agent = RandomAgent()
    
    # Option 2: Use your custom agents
    # from my_agent import MyAgent
    # fire_agent = MyAgent(device='cuda')
    # water_agent = MyAgent(device='cuda')
    
    # Option 3: Use example DQN agents
    # from example_custom_agent import SimpleQAgent
    # import torch
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # fire_agent = SimpleQAgent(device=device)
    # water_agent = SimpleQAgent(device=device)
    
    # ========================================================================
    # TRAIN WITH MODULAR TRAINER
    # ========================================================================
    
    # Note: We can't use ModularTrainer directly because we need custom env
    # Instead, we'll write a custom training loop
    
    print("Starting training...")
    print("=" * 60 + "\n")
    
    from collections import deque
    import numpy as np
    
    episode_rewards = deque(maxlen=100)
    success_tracker = deque(maxlen=100)
    stage_1_reached = deque(maxlen=100)
    
    for episode in range(training_config.num_episodes):
        fire_obs, water_obs = env.reset()
        reward_fn.reset()  # Reset stage tracking
        
        episode_reward = 0
        episode_length = 0
        reached_stage_1 = False
        
        done = False
        while not done:
            # Select actions
            fire_action = fire_agent.select_action(fire_obs)
            water_action = water_agent.select_action(water_obs)
            
            # Step environment
            (fire_next_obs, water_next_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
                fire_action, water_action
            )
            
            # Track stage progression
            if reward_fn.stage >= 1:
                reached_stage_1 = True
            
            # Update agents
            if hasattr(fire_agent, 'update'):
                fire_agent.update(fire_obs, fire_action, fire_reward, fire_next_obs, fire_done)
            if hasattr(water_agent, 'update'):
                water_agent.update(water_obs, water_action, water_reward, water_next_obs, water_done)
            
            # Track metrics
            episode_reward += fire_reward + water_reward
            episode_length += 1
            
            # Update observations
            fire_obs = fire_next_obs
            water_obs = water_next_obs
            done = fire_done or water_done
        
        # Episode complete - track metrics
        episode_rewards.append(episode_reward)
        success_tracker.append(1 if info.get('both_won', False) else 0)
        stage_1_reached.append(1 if reached_stage_1 else 0)
        
        # Logging
        if (episode + 1) % training_config.log_frequency == 0:
            avg_reward = np.mean(episode_rewards)
            success_rate = np.mean(success_tracker)
            stage_1_rate = np.mean(stage_1_reached)
            
            print(f"Episode {episode + 1}/{training_config.num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Stage 1 Reached: {stage_1_rate:.2%}")
            print(f"  Avg Length: {episode_length}")
            print()
        
        # Save checkpoints
        if (episode + 1) % training_config.save_frequency == 0:
            if hasattr(fire_agent, 'save'):
                fire_agent.save(f"checkpoints/fire_staged_ep{episode+1}.pth")
            if hasattr(water_agent, 'save'):
                water_agent.save(f"checkpoints/water_staged_ep{episode+1}.pth")
            print(f"[Checkpoint saved at episode {episode + 1}]\n")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final success rate: {np.mean(success_tracker):.2%}")
    print(f"Final stage 1 rate: {np.mean(stage_1_reached):.2%}")


def test_staged_rewards():
    """Test the staged reward function on a single episode"""
    
    print("\n" + "=" * 60)
    print("TESTING STAGED MILESTONE REWARDS")
    print("=" * 60 + "\n")
    
    from game_environment import FireWaterEnv
    
    # Create environment
    env = FireWaterEnv(max_steps=100)
    
    # Create reward function
    reward_config = StagedMilestoneRewardConfig()
    reward_config.beta = 50.0
    reward_config.gamma = 200.0
    reward_fn = StagedMilestoneRewardFunction(reward_config)
    
    # Override rewards
    def staged_rewards_wrapper(fire_won, water_won, fire_action, water_action):
        return reward_fn.calculate_rewards(
            env.fire_agent, env.water_agent,
            fire_action, water_action,
            env.fire_died, env.water_died,
            fire_won, water_won,
            env.bridge_activated, env.gate_activated,
            env.level.get_exits(), env.level.get_hazards()
        )
    
    env._calculate_rewards = staged_rewards_wrapper
    env.reward_fn = reward_fn
    
    # Run test episode
    fire_obs, water_obs = env.reset()
    reward_fn.reset()
    
    total_reward = 0
    step = 0
    done = False
    
    print("Running test episode with random actions...\n")
    
    while not done and step < 50:
        fire_action = np.random.randint(0, 6)
        water_action = np.random.randint(0, 6)
        
        (fire_obs, water_obs), (fire_reward, water_reward), dones, info = env.step(
            fire_action, water_action
        )
        
        total_reward += fire_reward + water_reward
        step += 1
        
        # Print stage transitions
        if step % 10 == 0:
            print(f"Step {step}:")
            print(f"  Stage: {reward_fn.stage}")
            print(f"  Reward: {fire_reward + water_reward:.3f}")
            print(f"  Bridge: {env.bridge_up}, Gate: {env.gate_open}")
        
        done = dones[0] or dones[1]
    
    print(f"\nTest complete!")
    print(f"  Total steps: {step}")
    print(f"  Total reward: {total_reward:.2f}")
    print(f"  Final stage: {reward_fn.stage}")
    print(f"  Success: {info.get('both_won', False)}")


def tune_reward_parameters():
    """Helper function to tune reward parameters"""
    
    print("\n" + "=" * 60)
    print("REWARD PARAMETER TUNING GUIDE")
    print("=" * 60 + "\n")
    
    print("Key Parameters to Tune:")
    print("-" * 60)
    
    print("\n1. Milestone Bonuses:")
    print("   beta = Plate activation bonus")
    print("     - Too low: Agents may not prioritize plates")
    print("     - Too high: Dominates progress rewards")
    print("     - Recommended: 50-100")
    print()
    print("   gamma = Finish bonus")
    print("     - Should be larger than beta")
    print("     - Encourages completing the task")
    print("     - Recommended: 200-500")
    
    print("\n2. Distance Normalization:")
    print("   diam_0 = Max distance in stage 0 (to plates)")
    print("     - Larger -> smaller progress rewards")
    print("     - Set to ~max distance in your level")
    print("     - Recommended: 1000-1500")
    print()
    print("   diam_1 = Max distance in stage 1 (to exits)")
    print("     - Similar logic as diam_0")
    print("     - Recommended: 1000-1500")
    
    print("\n3. Detection Radii:")
    print("   rho_p = Plate detection radius")
    print("     - Smaller -> must be more precise")
    print("     - Larger -> easier to activate")
    print("     - Recommended: 15-30 pixels")
    print()
    print("   rho_e = Exit detection radius")
    print("     - Similar to rho_p")
    print("     - Recommended: 15-30 pixels")
    
    print("\n4. Step Penalty:")
    print("   step_penalty = Small negative per step")
    print("     - Encourages efficiency")
    print("     - Too large: Agents rush, make mistakes")
    print("     - Recommended: -0.05 to -0.2")
    
    print("\n" + "=" * 60)
    print("Recommended Starting Configuration:")
    print("=" * 60)
    print("""
    reward_config = StagedMilestoneRewardConfig()
    reward_config.beta = 50.0
    reward_config.gamma = 200.0
    reward_config.diam_0 = 1000.0
    reward_config.diam_1 = 1000.0
    reward_config.rho_p = 20.0
    reward_config.rho_e = 20.0
    reward_config.step_penalty = -0.1
    """)


if __name__ == "__main__":
    import sys
    import os
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "train":
            train_with_staged_rewards()
        elif mode == "test":
            test_staged_rewards()
        elif mode == "tune":
            tune_reward_parameters()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: train, test, tune")
    else:
        print("Usage:")
        print("  python train_staged_milestone.py train  # Train with staged rewards")
        print("  python train_staged_milestone.py test   # Test reward function")
        print("  python train_staged_milestone.py tune   # Show tuning guide")
        print()
        
        # Default: show tuning guide
        tune_reward_parameters()