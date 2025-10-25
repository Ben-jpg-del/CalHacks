"""
Quick test to verify DQN training works with staged milestone rewards
Runs just 2 episodes to check for errors
"""

import numpy as np
import torch
from reward_staged_milestone import StagedMilestoneRewardFunction, StagedMilestoneRewardConfig
from game_environment import FireWaterEnv
from example_dqn import DQNAgent

print("=" * 60)
print("TESTING DQN TRAINING SETUP")
print("=" * 60)

# Create environment
print("\n[1/5] Creating environment...")
env = FireWaterEnv(max_steps=100)
print("  [OK] Environment created")

# Create reward function
print("\n[2/5] Creating staged milestone reward function...")
reward_config = StagedMilestoneRewardConfig()
reward_fn = StagedMilestoneRewardFunction(reward_config)

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
print("  [OK] Reward function injected")

# Create DQN agents
print("\n[3/5] Creating DQN agents...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

fire_agent = DQNAgent(state_dim=52, action_dim=6, device=device)
water_agent = DQNAgent(state_dim=52, action_dim=6, device=device)
print("  [OK] DQN agents created")

# Run 2 test episodes
print("\n[4/5] Running 2 test episodes...")
for episode in range(2):
    fire_obs, water_obs = env.reset()
    reward_fn.reset()

    episode_reward = 0
    steps = 0
    done = False

    while not done and steps < 50:
        # Select actions
        fire_action = fire_agent.select_action(fire_obs, training=True)
        water_action = water_agent.select_action(water_obs, training=True)

        # Step
        (fire_next_obs, water_next_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
            fire_action, water_action
        )

        # Store and update
        fire_agent.store_experience(fire_obs, fire_action, fire_reward, fire_next_obs, fire_done)
        water_agent.store_experience(water_obs, water_action, water_reward, water_next_obs, water_done)

        fire_agent.update()
        water_agent.update()

        # Track
        episode_reward += fire_reward + water_reward
        steps += 1

        fire_obs = fire_next_obs
        water_obs = water_next_obs
        done = fire_done or water_done

    print(f"  Episode {episode + 1}: {steps} steps, reward={episode_reward:.2f}")

print("  [OK] Test episodes completed")

# Test save/load
print("\n[5/5] Testing save/load...")
import os
os.makedirs("test_checkpoints", exist_ok=True)
fire_agent.save("test_checkpoints/test_fire.pth")
water_agent.save("test_checkpoints/test_water.pth")
print("  [OK] Agents saved")

test_agent = DQNAgent(device=device)
test_agent.load("test_checkpoints/test_fire.pth")
print("  [OK] Agent loaded")

# Cleanup
import shutil
shutil.rmtree("test_checkpoints")
print("  [OK] Cleanup complete")

print("\n" + "=" * 60)
print("[SUCCESS] ALL TESTS PASSED!")
print("=" * 60)
print("\nYou're ready to train!")
print("\nCommands:")
print("  python train_stage_milestone_dqn.py         # Without W&B")
print("  python train_stage_milestone_dqn.py --wandb # With W&B")
