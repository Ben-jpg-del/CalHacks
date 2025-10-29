"""
GPU-Accelerated Training Script
Trains thousands of parallel environments on GPU for massive speedup
Compatible with existing checkpoint format for visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import time
from typing import Optional, Dict, List
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import TorchFireWaterEnv
from map_config import LevelLibrary
from map_1 import LevelLibrary as Map1Library
from map_2 import LevelLibrary as Map2Library


class GPUDQNAgent(nn.Module):
    """
    DQN Agent optimized for GPU training
    All operations happen on GPU for maximum speed
    """

    def __init__(
        self,
        state_dim: int = 52,
        action_dim: int = 6,
        hidden_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.action_dim = action_dim

        # Q-Network
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        ).to(device)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """Forward pass - returns Q-values"""
        return self.network(states)

    def select_actions(
        self,
        states: torch.Tensor,
        epsilon: float = 0.0
    ) -> torch.Tensor:
        """
        Select actions using epsilon-greedy

        Args:
            states: [batch_size, state_dim]
            epsilon: Exploration rate

        Returns:
            actions: [batch_size]
        """
        batch_size = states.size(0)

        if epsilon > 0 and np.random.random() < epsilon:
            # Random actions
            return torch.randint(
                0,
                self.action_dim,
                (batch_size,),
                device=self.device
            )
        else:
            # Greedy actions
            with torch.no_grad():
                q_values = self.forward(states)
                return q_values.argmax(dim=1)


class GPUReplayBuffer:
    """
    Replay buffer that stores experiences on GPU
    Much faster than CPU-based buffers
    """

    def __init__(
        self,
        capacity: int,
        state_dim: int,
        device: str = 'cuda'
    ):
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate GPU memory
        self.states = torch.zeros(
            (capacity, state_dim),
            device=device,
            dtype=torch.float32
        )
        self.actions = torch.zeros(
            (capacity,),
            device=device,
            dtype=torch.long
        )
        self.rewards = torch.zeros(
            (capacity,),
            device=device,
            dtype=torch.float32
        )
        self.next_states = torch.zeros(
            (capacity, state_dim),
            device=device,
            dtype=torch.float32
        )
        self.dones = torch.zeros(
            (capacity,),
            device=device,
            dtype=torch.float32
        )

    def push(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ):
        """
        Add batch of experiences

        Args:
            All tensors are [batch_size, ...]
        """
        batch_size = states.size(0)

        # Handle wraparound
        if self.position + batch_size <= self.capacity:
            # No wraparound needed
            idx_start = self.position
            idx_end = self.position + batch_size

            self.states[idx_start:idx_end] = states
            self.actions[idx_start:idx_end] = actions
            self.rewards[idx_start:idx_end] = rewards
            self.next_states[idx_start:idx_end] = next_states
            self.dones[idx_start:idx_end] = dones

            self.position = idx_end % self.capacity
        else:
            # Need to wrap around
            first_part = self.capacity - self.position
            second_part = batch_size - first_part

            # First part
            self.states[self.position:] = states[:first_part]
            self.actions[self.position:] = actions[:first_part]
            self.rewards[self.position:] = rewards[:first_part]
            self.next_states[self.position:] = next_states[:first_part]
            self.dones[self.position:] = dones[:first_part]

            # Second part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]

            self.position = second_part

        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """Sample random batch"""
        indices = torch.randint(
            0,
            self.size,
            (batch_size,),
            device=self.device
        )

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size


def train_gpu(
    num_envs: int = 1024,
    num_episodes: int = 5000,
    batch_size: int = 1024,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    target_update_freq: int = 1000,
    buffer_capacity: int = 1000000,
    save_dir: str = 'checkpoints_gpu',
    save_freq: int = 100,
    log_freq: int = 10,
    device: str = 'cuda',
    use_wandb: bool = False,
    wandb_project: str = 'firewater-gpu',
    map_distribution: Optional[Dict[str, float]] = None
):
    """
    GPU-accelerated training

    Args:
        num_envs: Number of parallel environments (can be 1000+!)
        num_episodes: Number of episodes to train
        batch_size: Batch size for neural network updates
        learning_rate: Learning rate
        gamma: Discount factor
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        epsilon_decay: Epsilon decay per episode
        target_update_freq: Steps between target network updates
        buffer_capacity: Replay buffer size
        save_dir: Directory for checkpoints
        save_freq: Save every N episodes
        log_freq: Log every N episodes
        device: Device to use
        use_wandb: Enable wandb logging
        wandb_project: Wandb project name
        map_distribution: Distribution of maps to train on
    """

    print("=" * 80)
    print("GPU-ACCELERATED TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Parallel environments: {num_envs}")
    print(f"Batch size: {batch_size}")
    print("=" * 80 + "\n")

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            config={
                'num_envs': num_envs,
                'num_episodes': num_episodes,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'gamma': gamma,
                'device': device,
            }
        )

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Prepare level configs
    if map_distribution is None:
        map_distribution = {
            'tutorial': 0.5,
            'tower': 0.3,
            'map2': 0.2
        }

    # Create level configs for each environment
    map_names = list(map_distribution.keys())
    map_weights = list(map_distribution.values())
    map_choices = np.random.choice(
        map_names,
        size=num_envs,
        p=map_weights
    )

    level_configs = []
    for map_name in map_choices:
        if map_name == 'tutorial':
            level_configs.append(LevelLibrary.get_tutorial_level())
        elif map_name == 'tower':
            level_configs.append(Map1Library.get_tower_level())
        elif map_name == 'map2':
            level_configs.append(Map2Library.get_map2_level())

    print(f"Map distribution:")
    unique, counts = np.unique(map_choices, return_counts=True)
    for name, count in zip(unique, counts):
        print(f"  {name}: {count} ({count/num_envs*100:.1f}%)")
    print()

    # Create environment
    env = TorchFireWaterEnv(
        num_envs=num_envs,
        level_configs=level_configs,
        device=device
    )

    # Create agents
    fire_agent = GPUDQNAgent(device=device)
    water_agent = GPUDQNAgent(device=device)

    # Target networks
    fire_target = GPUDQNAgent(device=device)
    water_target = GPUDQNAgent(device=device)
    fire_target.load_state_dict(fire_agent.state_dict())
    water_target.load_state_dict(water_agent.state_dict())
    fire_target.eval()
    water_target.eval()

    # Optimizers
    fire_optimizer = optim.Adam(fire_agent.parameters(), lr=learning_rate)
    water_optimizer = optim.Adam(water_agent.parameters(), lr=learning_rate)

    # Replay buffers
    fire_buffer = GPUReplayBuffer(buffer_capacity, env.obs_dim, device)
    water_buffer = GPUReplayBuffer(buffer_capacity, env.obs_dim, device)

    # Training state
    epsilon = epsilon_start
    total_steps = 0

    # Metrics
    episode_rewards = deque(maxlen=100)
    success_rates = deque(maxlen=100)

    print("🚀 Starting GPU training...\n")
    print("NOTE: First episode may take a few minutes to initialize...")
    print("You should see progress updates every few seconds.\n")
    import sys
    sys.stdout.flush()  # Force output to show immediately

    start_time = time.time()

    for episode in range(num_episodes):
        episode_start = time.time()
        # Reset environments
        fire_obs, water_obs = env.reset()

        episode_reward = 0
        steps = 0
        max_steps = 3000

        while steps < max_steps:
            # Select actions
            fire_actions = fire_agent.select_actions(fire_obs, epsilon)
            water_actions = water_agent.select_actions(water_obs, epsilon)

            # Step environments
            (fire_obs_next, water_obs_next), \
            (fire_rewards, water_rewards), \
            (fire_dones, water_dones), \
            infos = env.step(fire_actions, water_actions)

            # Store transitions
            fire_buffer.push(
                fire_obs,
                fire_actions,
                fire_rewards,
                fire_obs_next,
                fire_dones
            )
            water_buffer.push(
                water_obs,
                water_actions,
                water_rewards,
                water_obs_next,
                water_dones
            )

            # Train if enough samples
            if len(fire_buffer) >= batch_size:
                # Sample and train fire agent
                states, actions, rewards, next_states, dones = fire_buffer.sample(batch_size)

                # Compute Q-values
                current_q = fire_agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                # Compute target Q-values (Double DQN)
                with torch.no_grad():
                    next_actions = fire_agent(next_states).argmax(dim=1)
                    next_q = fire_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards + (1 - dones) * gamma * next_q

                # Compute loss
                fire_loss = F.mse_loss(current_q, target_q)

                # Optimize
                fire_optimizer.zero_grad()
                fire_loss.backward()
                torch.nn.utils.clip_grad_norm_(fire_agent.parameters(), 10.0)
                fire_optimizer.step()

                # Same for water agent
                states, actions, rewards, next_states, dones = water_buffer.sample(batch_size)

                current_q = water_agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = water_agent(next_states).argmax(dim=1)
                    next_q = water_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
                    target_q = rewards + (1 - dones) * gamma * next_q

                water_loss = F.mse_loss(current_q, target_q)

                water_optimizer.zero_grad()
                water_loss.backward()
                torch.nn.utils.clip_grad_norm_(water_agent.parameters(), 10.0)
                water_optimizer.step()

            # Update observations
            fire_obs = fire_obs_next
            water_obs = water_obs_next

            # Track metrics
            episode_reward += (fire_rewards + water_rewards).mean().item()
            steps += 1
            total_steps += num_envs

            # Update target networks
            if total_steps % target_update_freq == 0:
                fire_target.load_state_dict(fire_agent.state_dict())
                water_target.load_state_dict(water_agent.state_dict())

            # Check if all done
            if fire_dones.all() and water_dones.all():
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Track metrics
        episode_rewards.append(episode_reward / steps if steps > 0 else 0)
        successes = sum(info['both_won'] for info in infos)
        success_rates.append(successes / num_envs * 100)

        # Print quick progress after EVERY episode
        episode_time = time.time() - episode_start
        print(f"Ep {episode + 1}/{num_episodes} | "
              f"Steps: {steps} | "
              f"Reward: {episode_reward/steps if steps > 0 else 0:.2f} | "
              f"Time: {episode_time:.1f}s | "
              f"Success: {success_rates[-1]:.0f}%", flush=True)

        # Detailed logging every N episodes
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards)
            avg_success = np.mean(success_rates)
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed

            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{num_episodes} - DETAILED STATS")
            print(f"{'='*60}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Success Rate: {avg_success:.1f}%")
            print(f"  Epsilon: {epsilon:.3f}")
            print(f"  Steps/sec: {steps_per_sec:.0f}")
            print(f"  Buffer: {len(fire_buffer)}")
            print(f"  Total Time: {elapsed/60:.1f}m")
            print(f"  ETA: {(elapsed/60)/(episode+1)*(num_episodes-episode-1):.1f}m")
            print(f"{'='*60}\n", flush=True)

            if use_wandb:
                wandb.log({
                    'episode': episode + 1,
                    'avg_reward': avg_reward,
                    'success_rate': avg_success,
                    'epsilon': epsilon,
                    'steps_per_second': steps_per_sec,
                    'buffer_size': len(fire_buffer),
                }, step=episode + 1)

        # Save checkpoints
        if (episode + 1) % save_freq == 0:
            checkpoint_dir = os.path.join(save_dir, f'checkpoint_ep{episode+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Save in format compatible with original training
            torch.save({
                'policy_net': fire_agent.network.state_dict(),
                'epsilon': epsilon,
            }, os.path.join(checkpoint_dir, 'fire_agent.pth'))

            torch.save({
                'policy_net': water_agent.network.state_dict(),
                'epsilon': epsilon,
            }, os.path.join(checkpoint_dir, 'water_agent.pth'))

            print(f"💾 Saved checkpoint to {checkpoint_dir}\n")

    # Save final models
    final_dir = os.path.join(save_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)

    torch.save({
        'policy_net': fire_agent.network.state_dict(),
        'epsilon': epsilon,
    }, os.path.join(final_dir, 'fire_agent.pth'))

    torch.save({
        'policy_net': water_agent.network.state_dict(),
        'epsilon': epsilon,
    }, os.path.join(final_dir, 'water_agent.pth'))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Final success rate: {np.mean(success_rates):.1f}%")
    print(f"Total steps: {total_steps:,}")
    print(f"Average steps/sec: {total_steps/(time.time()-start_time):.0f}")
    print("=" * 80)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GPU-Accelerated Training")
    parser.add_argument('--envs', type=int, default=1024, help='Number of parallel environments')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    parser.add_argument('--save-dir', type=str, default='checkpoints_gpu')

    args = parser.parse_args()

    train_gpu(
        num_envs=args.envs,
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        device=args.device,
        use_wandb=args.wandb,
        save_dir=args.save_dir
    )
