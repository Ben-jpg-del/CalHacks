"""
Visualize Trained Agents with Staged Milestone Rewards
Shows your trained DQN agents playing in Pygame
"""

import pygame as pg
import numpy as np
import torch
import sys
from game_environment import FireWaterEnv
from reward_staged_milestone import StagedMilestoneRewardFunction, StagedMilestoneRewardConfig
from example_dqn import DQNAgent
from visualize import Visualizer
import time


def visualize_trained_agents(
    fire_model_path: str,
    water_model_path: str = None,
    num_episodes: int = 10,
    fps: int = 60
):
    """
    Visualize trained DQN agents with staged milestone rewards

    Args:
        fire_model_path: Path to fire agent checkpoint
        water_model_path: Path to water agent checkpoint (if None, uses fire_model_path)
        num_episodes: Number of episodes to visualize
        fps: Frames per second
    """

    print("=" * 60)
    print("VISUALIZING TRAINED AGENTS")
    print("=" * 60)
    print(f"Fire agent: {fire_model_path}")
    print(f"Water agent: {water_model_path or fire_model_path}")
    print(f"Episodes: {num_episodes}")
    print(f"FPS: {fps}")
    print("=" * 60 + "\n")

    # ========================================================================
    # SETUP ENVIRONMENT WITH STAGED REWARDS
    # ========================================================================

    env = FireWaterEnv(max_steps=3000)
    reward_config = StagedMilestoneRewardConfig()
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
    # LOAD TRAINED AGENTS
    # ========================================================================

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("Loading trained agents...")

    # Create agents
    fire_agent = DQNAgent(
        state_dim=52,
        action_dim=6,
        device=device
    )
    water_agent = DQNAgent(
        state_dim=52,
        action_dim=6,
        device=device
    )

    # Load weights
    try:
        fire_agent.load(fire_model_path)
        print(f"  [OK] Fire agent loaded from {fire_model_path}")
    except Exception as e:
        print(f"  [ERROR] Could not load fire agent: {e}")
        print("  [INFO] Using untrained agent instead")

    try:
        water_model_path = water_model_path or fire_model_path.replace('fire', 'water')
        water_agent.load(water_model_path)
        print(f"  [OK] Water agent loaded from {water_model_path}")
    except Exception as e:
        print(f"  [ERROR] Could not load water agent: {e}")
        print("  [INFO] Using untrained agent instead")

    # Set to evaluation mode (no exploration)
    fire_agent.epsilon = 0.0
    water_agent.epsilon = 0.0

    print("\nAgents ready! Starting visualization...\n")

    # ========================================================================
    # VISUALIZATION LOOP
    # ========================================================================

    vis = Visualizer()
    success_tracker = []

    for episode in range(num_episodes):
        fire_obs, water_obs = env.reset()
        reward_fn.reset()

        episode_reward = 0
        episode_length = 0
        done = False

        print(f"Episode {episode + 1}/{num_episodes} - ", end='', flush=True)

        while not done:
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        pg.quit()
                        print("\nVisualization ended by user")
                        return
                    if event.key == pg.K_r:
                        print("Manual reset")
                        done = True
                        break

            if done:
                break

            # Get actions from trained agents (NO EXPLORATION)
            fire_action = fire_agent.select_action(fire_obs, training=False)
            water_action = water_agent.select_action(water_obs, training=False)

            # Step environment
            (fire_obs, water_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
                fire_action, water_action
            )

            episode_reward += fire_reward + water_reward
            episode_length += 1

            # Render with metrics
            metrics = {
                'episode': episode + 1,
                'total_episodes': num_episodes,
                'success_rate': np.mean(success_tracker) if success_tracker else 0,
                'stage': reward_fn.stage,
                'stage_name': ['Plates', 'Exits', 'Done'][reward_fn.stage],
                'reward': episode_reward,
                'length': episode_length
            }

            vis.render(env, metrics)
            vis.tick(fps)

            done = fire_done or water_done

        # Episode complete
        success = info.get('both_won', False)
        success_tracker.append(1 if success else 0)

        if success:
            print(f"SUCCESS! (Reward: {episode_reward:.1f}, Length: {episode_length})")
        else:
            if info.get('fire_died'):
                print(f"FAILED - Fire died (Reward: {episode_reward:.1f}, Length: {episode_length})")
            elif info.get('water_died'):
                print(f"FAILED - Water died (Reward: {episode_reward:.1f}, Length: {episode_length})")
            else:
                print(f"TIMEOUT (Reward: {episode_reward:.1f}, Length: {episode_length})")

        # Pause to see result
        time.sleep(1.5)

    # ========================================================================
    # SUMMARY
    # ========================================================================

    pg.quit()

    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Success rate: {np.mean(success_tracker):.2%}")
    print(f"Successes: {sum(success_tracker)}/{num_episodes}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize_staged_agents.py <fire_model_path> [water_model_path] [num_episodes]")
        print()
        print("Examples:")
        print("  python visualize_staged_agents.py checkpoints/fire_staged_dqn_ep5000.pth")
        print("  python visualize_staged_agents.py checkpoints/fire_staged_dqn_ep5000.pth checkpoints/water_staged_dqn_ep5000.pth")
        print("  python visualize_staged_agents.py checkpoints/fire_staged_dqn_ep10000.pth none 20")
        sys.exit(1)

    fire_model = sys.argv[1]
    water_model = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != 'none' else None
    num_eps = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    visualize_trained_agents(fire_model, water_model, num_eps)
