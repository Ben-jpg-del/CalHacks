"""
Modular Training Skeleton - Easy to Customize!
This is your main training script with all components plugged in.

QUICK START:
1. Edit training_config.py to set your hyperparameters
2. Edit reward_functions.py to customize your rewards
3. Run this script!
"""

import numpy as np
import torch
import os
from collections import deque
from typing import Optional

# Import our modular components
from training_config import TrainingConfig, get_config
from reward_functions import RewardConfig, get_reward_function
from gym_wrapper import FireWaterGymEnv, MultiAgentFireWaterGymEnv
from game_environment import FireWaterEnv


class ModularTrainer:
    """
    Modular training system
    Plug in your own agents, reward functions, and configurations!
    """
    
    def __init__(
        self,
        training_config: TrainingConfig,
        reward_config: Optional[RewardConfig] = None,
        fire_agent = None,
        water_agent = None
    ):
        """
        Initialize trainer
        
        Args:
            training_config: Training hyperparameters
            reward_config: Reward function configuration
            fire_agent: RL agent for fire character (None = random)
            water_agent: RL agent for water character (None = random)
        """
        self.config = training_config
        self.reward_config = reward_config or RewardConfig()
        
        # Create environment with custom reward function
        self.env = self._create_environment()
        
        # Agents (default to random if not provided)
        self.fire_agent = fire_agent or RandomAgent(self.config.action_dim)
        self.water_agent = water_agent or RandomAgent(self.config.action_dim)
        
        # Tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.success_tracker = deque(maxlen=100)
        self.cooperation_tracker = deque(maxlen=100)
        
        # Logging
        self.wandb_run = None
        if self.config.use_wandb:
            self._init_wandb()
    
    def _create_environment(self):
        """Create environment with custom reward function"""
        # Create base environment
        env = FireWaterEnv(max_steps=self.config.max_steps_per_episode)
        
        # Inject custom reward function
        reward_fn = get_reward_function(
            self.config.reward_function,
            self.reward_config
        )
        
        # Override the reward calculation
        original_calculate_rewards = env._calculate_rewards
        
        def custom_calculate_rewards(fire_won, water_won, fire_action, water_action):
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
        
        env._calculate_rewards = custom_calculate_rewards
        env.reward_fn = reward_fn
        
        return env
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.to_dict()
            )
            print(f"W&B logging enabled: {self.wandb_run.url}")
        except ImportError:
            print("Warning: wandb not installed. Logging disabled.")
            self.config.use_wandb = False
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("MODULAR RL TRAINING")
        print("=" * 60)
        print(f"Configuration:")
        print(f"  Episodes: {self.config.num_episodes}")
        print(f"  Max steps: {self.config.max_steps_per_episode}")
        print(f"  Reward function: {self.config.reward_function}")
        print(f"  Agent type: {self.config.agent_type}")
        print(f"  Device: {self.config.device}")
        print(f"  Parallel envs: {self.config.num_parallel_envs}")
        print("=" * 60 + "\n")
        
        for episode in range(self.config.num_episodes):
            # Run one episode
            metrics = self._run_episode(episode)
            
            # Track metrics
            self.episode_rewards.append(metrics['total_reward'])
            self.episode_lengths.append(metrics['length'])
            self.success_tracker.append(1 if metrics['success'] else 0)
            self.cooperation_tracker.append(metrics['cooperation_events'])
            
            # Logging
            if (episode + 1) % self.config.log_frequency == 0:
                self._log_metrics(episode + 1)
            
            # Checkpointing
            if (episode + 1) % self.config.save_frequency == 0:
                self._save_checkpoint(episode + 1)
            
            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                self._evaluate(episode + 1)
        
        # Training complete
        self._finish_training()
    
    def _run_episode(self, episode: int) -> dict:
        """
        Run one training episode
        
        Returns:
            Dictionary with episode metrics
        """
        # Reset environment and reward function
        fire_obs, water_obs = self.env.reset()
        if hasattr(self.env, 'reward_fn'):
            self.env.reward_fn.reset()
        
        episode_fire_reward = 0
        episode_water_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            # Select actions
            fire_action = self.fire_agent.select_action(fire_obs)
            water_action = self.water_agent.select_action(water_obs)
            
            # Step environment
            (fire_next_obs, water_next_obs), (fire_reward, water_reward), (fire_done, water_done), info = self.env.step(
                fire_action, water_action
            )
            
            # Update agents
            if hasattr(self.fire_agent, 'update'):
                self.fire_agent.update(fire_obs, fire_action, fire_reward, fire_next_obs, fire_done)
            if hasattr(self.water_agent, 'update'):
                self.water_agent.update(water_obs, water_action, water_reward, water_next_obs, water_done)
            
            # Track rewards
            episode_fire_reward += fire_reward
            episode_water_reward += water_reward
            episode_length += 1
            
            # Update observations
            fire_obs = fire_next_obs
            water_obs = water_next_obs
            
            # Check done
            done = fire_done or water_done
        
        # Compile metrics
        return {
            'fire_reward': episode_fire_reward,
            'water_reward': episode_water_reward,
            'total_reward': episode_fire_reward + episode_water_reward,
            'length': episode_length,
            'success': info.get('both_won', False),
            'cooperation_events': info.get('cooperation_events', 0),
            'fire_died': info.get('fire_died', False),
            'water_died': info.get('water_died', False)
        }
    
    def _log_metrics(self, episode: int):
        """Log training metrics"""
        avg_reward = np.mean(self.episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        success_rate = np.mean(self.success_tracker)
        avg_coop = np.mean(self.cooperation_tracker)
        
        print(f"Episode {episode}/{self.config.num_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Length: {avg_length:.1f}")
        print(f"  Success Rate: {success_rate:.2%}")
        print(f"  Avg Cooperation: {avg_coop:.2f}")
        
        # Log to W&B
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.log({
                "episode": episode,
                "avg_reward": avg_reward,
                "avg_length": avg_length,
                "success_rate": success_rate,
                "avg_cooperation": avg_coop
            })
    
    def _save_checkpoint(self, episode: int):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.save_dir, f"checkpoint_ep{episode}.pth")
        
        # Save agents if they have save method
        if hasattr(self.fire_agent, 'save'):
            fire_path = os.path.join(self.config.save_dir, f"fire_ep{episode}.pth")
            self.fire_agent.save(fire_path)
        
        if hasattr(self.water_agent, 'save'):
            water_path = os.path.join(self.config.save_dir, f"water_ep{episode}.pth")
            self.water_agent.save(water_path)
        
        print(f"  [Checkpoint saved at episode {episode}]")
    
    def _evaluate(self, episode: int):
        """Run evaluation episodes"""
        print(f"\n  [Evaluating at episode {episode}...]")
        
        eval_rewards = []
        eval_successes = []
        
        for _ in range(self.config.num_eval_episodes):
            fire_obs, water_obs = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                # Use greedy policy for evaluation
                fire_action = self.fire_agent.select_action(fire_obs, training=False)
                water_action = self.water_agent.select_action(water_obs, training=False)
                
                (fire_obs, water_obs), rewards, dones, info = self.env.step(fire_action, water_action)
                episode_reward += sum(rewards)
                done = dones[0] or dones[1]
            
            eval_rewards.append(episode_reward)
            eval_successes.append(1 if info.get('both_won', False) else 0)
        
        eval_avg_reward = np.mean(eval_rewards)
        eval_success_rate = np.mean(eval_successes)
        
        print(f"  Eval Avg Reward: {eval_avg_reward:.2f}")
        print(f"  Eval Success Rate: {eval_success_rate:.2%}\n")
        
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.log({
                "eval_episode": episode,
                "eval_avg_reward": eval_avg_reward,
                "eval_success_rate": eval_success_rate
            })
    
    def _finish_training(self):
        """Clean up after training"""
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Final success rate: {np.mean(self.success_tracker):.2%}")
        print(f"Final avg reward: {np.mean(self.episode_rewards):.2f}")
        
        # Save final models
        if hasattr(self.fire_agent, 'save'):
            self.fire_agent.save(os.path.join(self.config.save_dir, "fire_final.pth"))
        if hasattr(self.water_agent, 'save'):
            self.water_agent.save(os.path.join(self.config.save_dir, "water_final.pth"))
        
        if self.config.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        print("=" * 60)


class RandomAgent:
    """Simple random agent for testing"""
    def __init__(self, action_dim=6):
        self.action_dim = action_dim
    
    def select_action(self, observation, training=True):
        return np.random.randint(0, self.action_dim)
    
    def update(self, obs, action, reward, next_obs, done):
        pass


# ========== EASY CUSTOMIZATION FUNCTIONS ==========

def train_with_defaults():
    """Train with default settings (good starting point)"""
    config = get_config('standard')
    reward_config = RewardConfig()
    
    trainer = ModularTrainer(config, reward_config)
    trainer.train()


def train_with_custom_rewards():
    """Example: Train with custom reward configuration"""
    config = get_config('standard')
    
    # Customize rewards
    reward_config = RewardConfig()
    reward_config.both_win_reward = 500.0      # Much higher win reward
    reward_config.step_penalty = -0.05         # Encourage faster completion
    reward_config.distance_weight = 0.01       # Stronger distance shaping
    reward_config.cooperation_bonus = 20.0     # Emphasize cooperation
    
    # Use cooperation-focused reward function
    config.reward_function = "cooperation"
    
    trainer = ModularTrainer(config, reward_config)
    trainer.train()


def train_fast_debug():
    """Example: Fast training for debugging"""
    config = get_config('debug')
    reward_config = RewardConfig()
    
    trainer = ModularTrainer(config, reward_config)
    trainer.train()


def train_with_custom_config():
    """Example: Train with completely custom configuration"""
    # Start with standard config
    config = get_config('standard')
    
    # Customize everything!
    config.num_episodes = 20000
    config.learning_rate = 1e-4
    config.batch_size = 128
    config.hidden_dim = 512
    config.reward_function = "dense"
    config.use_wandb = True
    config.wandb_project = "my-custom-training"
    
    # Custom reward settings
    reward_config = RewardConfig()
    reward_config.both_win_reward = 1000.0
    reward_config.death_penalty = -200.0
    
    # Add your own RL agents here
    # fire_agent = YourCustomAgent()
    # water_agent = YourCustomAgent()
    
    trainer = ModularTrainer(config, reward_config)
    trainer.train()


def train_from_config_file(config_path: str):
    """Load configuration from file and train"""
    config = TrainingConfig.load(config_path)
    reward_config = RewardConfig()
    
    trainer = ModularTrainer(config, reward_config)
    trainer.train()


# ========== MAIN ENTRY POINT ==========

if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("MODULAR RL TRAINING SKELETON")
    print("=" * 60)
    print("\nAvailable training modes:")
    print("  1. defaults       - Standard training")
    print("  2. custom-rewards - Custom reward configuration")
    print("  3. debug          - Fast debug training")
    print("  4. custom-config  - Fully custom configuration")
    print("  5. from-file      - Load config from JSON file")
    print("=" * 60 + "\n")
    
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        
        if mode == "defaults":
            train_with_defaults()
        elif mode == "custom-rewards":
            train_with_custom_rewards()
        elif mode == "debug":
            train_fast_debug()
        elif mode == "custom-config":
            train_with_custom_config()
        elif mode == "from-file":
            if len(sys.argv) < 3:
                print("Usage: python train_modular.py from-file <config.json>")
            else:
                train_from_config_file(sys.argv[2])
        else:
            print(f"Unknown mode: {mode}")
    else:
        # Default: standard training
        print("Starting standard training...")
        print("(Use: python train_modular.py <mode> to choose different modes)\n")
        train_with_defaults()
