"""
Fast Headless Training - Train RL agents without visualization
This script runs 10-100x faster than training with rendering
"""

import numpy as np
import time
from collections import deque
from game_environment import FireWaterEnv
from map_config import LevelConfig

# Optional: Uncomment if you have wandb installed
# import wandb


class RandomAgent:
    """Simple random agent for testing"""
    def __init__(self, action_space_size=6):
        self.action_space_size = action_space_size
    
    def select_action(self, observation):
        """Select random action"""
        return np.random.randint(0, self.action_space_size)
    
    def update(self, obs, action, reward, next_obs, done):
        """Placeholder for learning update"""
        pass


def train_headless(
    num_episodes: int = 10000,
    max_steps_per_episode: int = 3000,
    log_frequency: int = 100,
    use_wandb: bool = False,
    save_frequency: int = 1000
):
    """
    Fast training loop without visualization
    
    Args:
        num_episodes: Number of episodes to train
        max_steps_per_episode: Maximum steps per episode
        log_frequency: How often to print/log metrics
        use_wandb: Whether to use Weights & Biases for logging
        save_frequency: How often to save checkpoints
    """
    
    print("=" * 60)
    print("FAST HEADLESS TRAINING")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print(f"Visualization: DISABLED (for maximum speed)")
    print("=" * 60 + "\n")
    
    # Initialize environment
    env = FireWaterEnv(max_steps=max_steps_per_episode)
    
    # Initialize agents (replace with your RL agents)
    fire_agent = RandomAgent(action_space_size=6)
    water_agent = RandomAgent(action_space_size=6)
    
    # Tracking metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    success_tracker = deque(maxlen=100)
    cooperation_tracker = deque(maxlen=100)
    
    # Initialize W&B if requested
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="firewater-fast-training",
                config={
                    "num_episodes": num_episodes,
                    "max_steps": max_steps_per_episode,
                    "agent_type": "random"  # Update with your agent type
                }
            )
            print("Weights & Biases logging enabled\n")
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.\n")
            use_wandb = False
    
    # Training loop
    start_time = time.time()
    
    for episode in range(num_episodes):
        # Reset environment
        fire_obs, water_obs = env.reset()
        
        episode_fire_reward = 0
        episode_water_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select actions
            fire_action = fire_agent.select_action(fire_obs)
            water_action = water_agent.select_action(water_obs)
            
            # Step environment
            (fire_next_obs, water_next_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
                fire_action, water_action
            )
            
            # Update agents (implement your learning algorithm here)
            fire_agent.update(fire_obs, fire_action, fire_reward, fire_next_obs, fire_done)
            water_agent.update(water_obs, water_action, water_reward, water_next_obs, water_done)
            
            # Accumulate rewards
            episode_fire_reward += fire_reward
            episode_water_reward += water_reward
            episode_length += 1
            
            # Update observations
            fire_obs = fire_next_obs
            water_obs = water_next_obs
            
            # Check if episode is done
            done = fire_done or water_done
        
        # Track episode metrics
        total_reward = episode_fire_reward + episode_water_reward
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        success_tracker.append(1 if info['both_won'] else 0)
        cooperation_tracker.append(info['cooperation_events'])
        
        # Logging
        if (episode + 1) % log_frequency == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            success_rate = np.mean(success_tracker)
            avg_coop = np.mean(cooperation_tracker)
            
            elapsed = time.time() - start_time
            episodes_per_sec = (episode + 1) / elapsed
            
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Success Rate: {success_rate:.2%}")
            print(f"  Avg Cooperation: {avg_coop:.2f}")
            print(f"  Speed: {episodes_per_sec:.2f} episodes/sec")
            print(f"  Elapsed: {elapsed:.1f}s\n")
            
            # Log to W&B
            if use_wandb:
                wandb.log({
                    "episode": episode + 1,
                    "avg_reward": avg_reward,
                    "avg_length": avg_length,
                    "success_rate": success_rate,
                    "avg_cooperation": avg_coop,
                    "episodes_per_sec": episodes_per_sec
                })
        
        # Save checkpoint
        if (episode + 1) % save_frequency == 0:
            print(f"[Checkpoint] Episode {episode + 1} - Saving models...")
            # TODO: Implement model saving
            # torch.save(fire_agent.state_dict(), f"fire_agent_{episode+1}.pth")
            # torch.save(water_agent.state_dict(), f"water_agent_{episode+1}.pth")
    
    # Training complete
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total episodes: {num_episodes}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Average speed: {num_episodes/total_time:.2f} episodes/sec")
    print(f"Final success rate: {np.mean(success_tracker):.2%}")
    print("=" * 60)
    
    if use_wandb:
        wandb.finish()


def train_with_curriculum(
    stages: list = None,
    episodes_per_stage: int = 2000
):
    """
    Train with curriculum learning (progressively harder tasks)
    
    Args:
        stages: List of stage configurations
        episodes_per_stage: Episodes to train per stage
    """
    if stages is None:
        # Default curriculum: increase episode length
        stages = [
            {"max_steps": 500, "name": "Stage 1: Quick exploration"},
            {"max_steps": 1000, "name": "Stage 2: Medium episodes"},
            {"max_steps": 2000, "name": "Stage 3: Long episodes"},
            {"max_steps": 3000, "name": "Stage 4: Full episodes"}
        ]
    
    print("=" * 60)
    print("CURRICULUM LEARNING")
    print("=" * 60)
    
    for i, stage in enumerate(stages):
        print(f"\n{stage['name']}")
        print("-" * 60)
        
        # Create environment with stage settings
        env = FireWaterEnv(max_steps=stage['max_steps'])
        
        # Train for this stage
        train_headless(
            num_episodes=episodes_per_stage,
            max_steps_per_episode=stage['max_steps'],
            log_frequency=100,
            use_wandb=False
        )
    
    print("\n" + "=" * 60)
    print("CURRICULUM COMPLETE")
    print("=" * 60)


def benchmark_speed():
    """Benchmark environment speed"""
    print("Benchmarking environment speed...")
    
    env = FireWaterEnv()
    
    num_steps = 10000
    start_time = time.time()
    
    env.reset()
    for _ in range(num_steps):
        fire_action = np.random.randint(0, 6)
        water_action = np.random.randint(0, 6)
        _, _, (done_f, done_w), _ = env.step(fire_action, water_action)
        if done_f or done_w:
            env.reset()
    
    elapsed = time.time() - start_time
    steps_per_sec = num_steps / elapsed
    
    print(f"\nBenchmark Results:")
    print(f"  Steps: {num_steps}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {steps_per_sec:.0f} steps/sec")
    print(f"  Speed: {steps_per_sec*60:.0f} steps/min")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "benchmark":
            benchmark_speed()
        elif mode == "curriculum":
            train_with_curriculum()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: benchmark, curriculum")
    else:
        # Default: fast training
        train_headless(
            num_episodes=5000,
            log_frequency=100,
            use_wandb=False
        )
