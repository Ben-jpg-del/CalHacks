"""
Parallel Multi-Map Training Script
Optimized for GPU (L4) with PyTorch parallelization
Trains a generalized policy across multiple maps simultaneously
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import time
from typing import Optional, Dict
import os
import wandb

from parallel_multi_map_env import ParallelMultiMapEnv, CurriculumMultiMapEnv, MapRegistry
from example_dqn import DQNetwork, ReplayBuffer
from reward_functions import get_reward_function, RewardConfig


class ParallelDQNAgent:
    """
    DQN Agent optimized for parallel environment training
    Uses batched operations for efficiency on GPU
    """

    def __init__(
        self,
        state_dim: int = 52,
        action_dim: int = 6,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.9995,
        target_update_freq: int = 500,
        batch_size: int = 256,  # Larger batch for parallel training
        buffer_capacity: int = 500000,  # Larger buffer for multi-map
        device: str = 'cuda'
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device
        self.update_count = 0

        # Networks
        self.policy_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer
        self.memory = ReplayBuffer(buffer_capacity)

    def select_actions(self, states: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Select actions for batch of states (parallelized)

        Args:
            states: Tensor of shape [num_envs, state_dim]
            training: Whether in training mode (use epsilon-greedy)

        Returns:
            actions: Tensor of shape [num_envs]
        """
        if training and np.random.random() < self.epsilon:
            # Random actions for exploration
            return torch.randint(0, self.action_dim, (states.shape[0],), device=self.device)
        else:
            # Greedy actions
            with torch.no_grad():
                q_values = self.policy_net(states)
                return q_values.argmax(dim=1)

    def store_transitions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor
    ):
        """
        Store batch of transitions in replay buffer

        Args:
            All tensors of shape [num_envs, ...]
        """
        # Convert to CPU numpy
        states_np = states.cpu().numpy()
        actions_np = actions.cpu().numpy()
        rewards_np = rewards.cpu().numpy()
        next_states_np = next_states.cpu().numpy()
        dones_np = dones.cpu().numpy()

        # Store each transition
        for i in range(states.shape[0]):
            self.memory.push(
                states_np[i],
                actions_np[i],
                rewards_np[i],
                next_states_np[i],
                dones_np[i]
            )

    def train_step(self) -> Optional[float]:
        """
        Perform one training step (if enough samples in buffer)

        Returns:
            Loss value or None
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Move to device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q values (Double DQN)
        with torch.no_grad():
            # Use policy net to select actions
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # Use target net to evaluate
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss (Huber loss for stability)
        loss = nn.SmoothL1Loss()(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, filepath: str):
        """Save agent checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, filepath)

    def load(self, filepath: str):
        """Load agent checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.update_count = checkpoint['update_count']


def train_parallel_multimap(
    num_episodes: int = 5000,
    num_envs_per_map: int = 8,
    map_distribution: Optional[Dict[str, float]] = None,
    use_curriculum: bool = True,
    curriculum_schedule: Optional[Dict[int, Dict[str, float]]] = None,
    reward_type: str = 'dense',
    save_dir: str = 'checkpoints_multimap',
    save_freq: int = 100,
    log_freq: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    resume_from: Optional[str] = None,
    use_wandb: bool = False,
    wandb_project: str = 'firewater-multimap',
    wandb_run_name: Optional[str] = None
):
    """
    Train agents with parallel multi-map environments

    Args:
        num_episodes: Total episodes to train
        num_envs_per_map: Number of parallel environments per map
        map_distribution: Custom map distribution (if not using curriculum)
        use_curriculum: Whether to use curriculum learning
        curriculum_schedule: Custom curriculum schedule
        reward_type: Reward function type ('sparse', 'dense', 'cooperation', etc.)
        save_dir: Directory to save checkpoints
        save_freq: Save checkpoint every N episodes
        log_freq: Log metrics every N episodes
        device: PyTorch device
        resume_from: Path to checkpoint directory to resume from
        use_wandb: Whether to use Weights & Biases logging
        wandb_project: W&B project name
        wandb_run_name: W&B run name (optional)
    """
    print("=" * 80)
    print("PARALLEL MULTI-MAP TRAINING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Available maps: {MapRegistry.get_map_names()}")
    print(f"Num environments per map: {num_envs_per_map}")
    print(f"Curriculum learning: {use_curriculum}")
    print(f"Reward type: {reward_type}")
    print(f"W&B Logging: {use_wandb}")
    print("=" * 80 + "\n")

    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                'num_episodes': num_episodes,
                'num_envs_per_map': num_envs_per_map,
                'use_curriculum': use_curriculum,
                'curriculum_schedule': curriculum_schedule,
                'reward_type': reward_type,
                'device': device,
                'map_distribution': map_distribution,
                'available_maps': MapRegistry.get_map_names(),
            },
            resume='allow' if resume_from else False
        )
        print(f"✅ W&B initialized: {wandb.run.name}\n")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create reward function
    reward_config = RewardConfig()
    reward_fn = get_reward_function(reward_type, reward_config)

    # Create environment
    if use_curriculum:
        if curriculum_schedule is None:
            # Default curriculum: start easy, gradually add harder maps
            curriculum_schedule = {
                0: {'tutorial': 1.0},
                500: {'tutorial': 0.7, 'tower': 0.3},
                1000: {'tutorial': 0.5, 'tower': 0.5},
                2000: {'tutorial': 0.3, 'tower': 0.7},
            }

        env = CurriculumMultiMapEnv(
            num_envs_per_map=num_envs_per_map,
            curriculum_schedule=curriculum_schedule,
            reward_function=reward_fn,
            device=device
        )
    else:
        env = ParallelMultiMapEnv(
            num_envs_per_map=num_envs_per_map,
            map_distribution=map_distribution,
            reward_function=reward_fn,
            device=device
        )

    # Create agents
    fire_agent = ParallelDQNAgent(
        state_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device
    )
    water_agent = ParallelDQNAgent(
        state_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=device
    )

    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from:
        fire_checkpoint = os.path.join(resume_from, 'fire_agent.pth')
        water_checkpoint = os.path.join(resume_from, 'water_agent.pth')

        if os.path.exists(fire_checkpoint) and os.path.exists(water_checkpoint):
            print(f"📂 Loading checkpoints from {resume_from}")
            fire_agent.load(fire_checkpoint)
            water_agent.load(water_checkpoint)

            # Try to get episode number from directory name
            try:
                start_episode = int(resume_from.split('_ep')[-1])
                print(f"🔄 Resuming from episode {start_episode}\n")
            except:
                print("⚠️  Could not determine start episode, starting from 0\n")
        else:
            print(f"⚠️  Checkpoint files not found in {resume_from}, starting fresh\n")

    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    success_rate_tracker = deque(maxlen=100)
    map_success_rates = {map_name: deque(maxlen=100) for map_name in MapRegistry.get_map_names()}

    print("🚀 Starting training...\n")
    start_time = time.time()

    for episode in range(start_episode, num_episodes):
        # Update curriculum if using
        if use_curriculum:
            env.update_curriculum(episode)

        # Reset environments
        fire_obs, water_obs = env.reset()

        episode_reward = 0
        episode_steps = 0
        done_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        max_steps = 3000  # Prevent episodes from running too long

        while not done_mask.all() and episode_steps < max_steps:
            # Select actions for all environments
            fire_actions = fire_agent.select_actions(fire_obs, training=True)
            water_actions = water_agent.select_actions(water_obs, training=True)

            # Step all environments
            (fire_obs_next, water_obs_next), \
            (fire_rewards, water_rewards), \
            (fire_dones, water_dones), \
            infos = env.step(fire_actions, water_actions)

            # Store transitions
            fire_agent.store_transitions(fire_obs, fire_actions, fire_rewards, fire_obs_next, fire_dones)
            water_agent.store_transitions(water_obs, water_actions, water_rewards, water_obs_next, water_dones)

            # Train agents
            fire_loss = fire_agent.train_step()
            water_loss = water_agent.train_step()

            # Update observations
            fire_obs = fire_obs_next
            water_obs = water_obs_next

            # Track episode stats (only for non-done environments)
            active_mask = ~done_mask
            episode_reward += (fire_rewards + water_rewards)[active_mask].sum().item()
            episode_steps += active_mask.sum().item()

            # Update done mask
            done_mask = done_mask | (fire_dones.bool() | water_dones.bool())

        # Track success rates
        successes = [info.get('both_won', False) for info in infos]
        success_rate_tracker.append(np.mean(successes))

        # Track per-map success rates
        for info in infos:
            map_name = info['map_name']
            if map_name in map_success_rates:
                map_success_rates[map_name].append(info.get('both_won', False))

        episode_rewards.append(episode_reward / env.num_envs)
        episode_lengths.append(episode_steps / env.num_envs)

        # Logging
        if (episode + 1) % log_freq == 0:
            avg_reward = np.mean(episode_rewards)
            avg_length = np.mean(episode_lengths)
            avg_success = np.mean(success_rate_tracker) * 100
            elapsed = time.time() - start_time

            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Length: {avg_length:.1f}")
            print(f"  Success Rate: {avg_success:.1f}%")
            print(f"  Fire Epsilon: {fire_agent.epsilon:.3f}")
            print(f"  Water Epsilon: {water_agent.epsilon:.3f}")

            # Per-map success rates
            per_map_metrics = {}
            for map_name, successes in map_success_rates.items():
                if len(successes) > 0:
                    map_success = np.mean(successes) * 100
                    print(f"  {map_name.capitalize()} Success: {map_success:.1f}%")
                    per_map_metrics[f'success_rate/{map_name}'] = map_success

            print(f"  Time: {elapsed/60:.1f}m")
            print(f"  Buffer: {len(fire_agent.memory)}")
            print()

            # Log to wandb
            if use_wandb:
                log_dict = {
                    'episode': episode + 1,
                    'avg_reward': avg_reward,
                    'avg_episode_length': avg_length,
                    'success_rate': avg_success,
                    'fire_epsilon': fire_agent.epsilon,
                    'water_epsilon': water_agent.epsilon,
                    'buffer_size': len(fire_agent.memory),
                    'elapsed_time_minutes': elapsed / 60,
                }
                # Add per-map metrics
                log_dict.update(per_map_metrics)
                wandb.log(log_dict, step=episode + 1)

        # Save checkpoints
        if (episode + 1) % save_freq == 0:
            checkpoint_dir = os.path.join(save_dir, f'checkpoint_ep{episode+1}')
            os.makedirs(checkpoint_dir, exist_ok=True)

            fire_agent.save(os.path.join(checkpoint_dir, 'fire_agent.pth'))
            water_agent.save(os.path.join(checkpoint_dir, 'water_agent.pth'))

            print(f"💾 Saved checkpoint to {checkpoint_dir}\n")

    # Save final models
    final_dir = os.path.join(save_dir, 'final')
    os.makedirs(final_dir, exist_ok=True)
    fire_agent.save(os.path.join(final_dir, 'fire_agent.pth'))
    water_agent.save(os.path.join(final_dir, 'water_agent.pth'))

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Final success rate: {np.mean(success_rate_tracker)*100:.1f}%")
    for map_name, successes in map_success_rates.items():
        if len(successes) > 0:
            print(f"  {map_name.capitalize()}: {np.mean(successes)*100:.1f}%")
    print(f"Models saved to: {save_dir}")
    print("=" * 80)

    # Log final metrics to wandb
    if use_wandb:
        final_metrics = {
            'final/total_time_hours': (time.time() - start_time) / 3600,
            'final/success_rate': np.mean(success_rate_tracker) * 100,
        }
        for map_name, successes in map_success_rates.items():
            if len(successes) > 0:
                final_metrics[f'final/success_rate_{map_name}'] = np.mean(successes) * 100

        wandb.log(final_metrics)

        # Save model artifacts
        artifact = wandb.Artifact(
            name=f'firewater-agents-{wandb.run.id}',
            type='model',
            description='Final trained Fire and Water agents'
        )
        artifact.add_file(os.path.join(final_dir, 'fire_agent.pth'))
        artifact.add_file(os.path.join(final_dir, 'water_agent.pth'))
        wandb.log_artifact(artifact)

        wandb.finish()
        print("✅ W&B run finished and artifacts saved\n")

    env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Multi-Map Training")
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes')
    parser.add_argument('--envs-per-map', type=int, default=8, help='Environments per map')
    parser.add_argument('--no-curriculum', action='store_true', help='Disable curriculum learning')
    parser.add_argument('--reward', type=str, default='dense', choices=['sparse', 'dense', 'cooperation', 'safety'])
    parser.add_argument('--save-dir', type=str, default='checkpoints_multimap')
    parser.add_argument('--resume-from', type=str, default=None, help='Resume from checkpoint directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='firewater-multimap', help='W&B project name')
    parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')

    args = parser.parse_args()

    train_parallel_multimap(
        num_episodes=args.episodes,
        num_envs_per_map=args.envs_per_map,
        use_curriculum=not args.no_curriculum,
        reward_type=args.reward,
        save_dir=args.save_dir,
        device=args.device,
        resume_from=args.resume_from,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
