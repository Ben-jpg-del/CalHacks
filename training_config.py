"""
Training Configuration System
Easy-to-customize training parameters and hyperparameters
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import json
import os


@dataclass
class TrainingConfig:
    """
    Master configuration for training - CUSTOMIZE ALL THESE PARAMETERS!
    
    This is your single source of truth for all training settings.
    Edit these values instead of hardcoding them throughout your code.
    """
    
    # ========== BASIC TRAINING SETTINGS ==========
    num_episodes: int = 10000           # Total episodes to train
    max_steps_per_episode: int = 3000   # Max steps before timeout
    random_seed: Optional[int] = None   # Random seed (None = random)
    
    # ========== AGENT SETTINGS ==========
    agent_type: str = "dqn"             # Agent type: "dqn", "ppo", "sac", "custom"
    state_dim: int = 52                 # Observation dimension
    action_dim: int = 6                 # Action dimension
    
    # ========== NETWORK ARCHITECTURE ==========
    hidden_dim: int = 256               # Hidden layer size
    num_layers: int = 2                 # Number of hidden layers
    activation: str = "relu"            # Activation: "relu", "tanh", "elu"
    use_dueling: bool = True            # Use dueling architecture (DQN)
    use_noisy_layers: bool = False      # Use noisy nets for exploration
    
    # ========== LEARNING HYPERPARAMETERS ==========
    learning_rate: float = 3e-4         # Learning rate
    batch_size: int = 64                # Batch size for updates
    gamma: float = 0.99                 # Discount factor
    tau: float = 0.005                  # Soft update coefficient (target network)
    
    # ========== EXPLORATION ==========
    epsilon_start: float = 1.0          # Initial exploration rate
    epsilon_end: float = 0.01           # Final exploration rate
    epsilon_decay: float = 0.995        # Decay rate per episode
    epsilon_decay_steps: int = 10000    # Decay over N episodes
    
    # ========== EXPERIENCE REPLAY ==========
    buffer_size: int = 100000           # Replay buffer capacity
    min_buffer_size: int = 1000         # Min samples before training
    prioritized_replay: bool = False    # Use prioritized experience replay
    priority_alpha: float = 0.6         # Priority exponent
    priority_beta: float = 0.4          # Importance sampling exponent
    
    # ========== TARGET NETWORK ==========
    target_update_frequency: int = 100  # Update target network every N steps
    use_soft_update: bool = True        # Use soft update vs hard update
    
    # ========== REWARD SETTINGS ==========
    reward_function: str = "dense"      # Reward type: "sparse", "dense", "safety", "cooperation", "curriculum"
    reward_scale: float = 1.0           # Scale all rewards
    reward_clip: Optional[float] = None # Clip rewards to [-clip, clip]
    
    # ========== TRAINING FEATURES ==========
    use_curriculum: bool = False        # Use curriculum learning
    use_her: bool = False               # Use Hindsight Experience Replay
    use_multi_step: bool = False        # Use multi-step returns
    n_step: int = 3                     # N for n-step returns
    
    # ========== VECTORIZATION ==========
    num_parallel_envs: int = 1          # Number of parallel environments
    use_gpu: bool = True                # Use GPU if available
    device: str = "auto"                # Device: "cpu", "cuda", "auto"
    
    # ========== LOGGING & CHECKPOINTING ==========
    log_frequency: int = 100            # Log metrics every N episodes
    save_frequency: int = 500           # Save model every N episodes
    eval_frequency: int = 1000          # Evaluate every N episodes
    num_eval_episodes: int = 10         # Episodes for evaluation
    
    use_wandb: bool = False             # Use Weights & Biases logging
    wandb_project: str = "firewater-rl" # W&B project name
    wandb_entity: Optional[str] = None  # W&B entity (username/team)
    
    save_dir: str = "checkpoints"       # Directory for saved models
    log_dir: str = "logs"               # Directory for logs
    
    # ========== VISUALIZATION ==========
    render_training: bool = False       # Render during training (SLOW!)
    render_eval: bool = True            # Render during evaluation
    render_fps: int = 60                # FPS for rendering
    
    # ========== ADVANCED ==========
    gradient_clip: Optional[float] = 10.0   # Gradient clipping value
    weight_decay: float = 0.0           # L2 regularization
    use_layer_norm: bool = False        # Use layer normalization
    use_batch_norm: bool = False        # Use batch normalization
    
    # Partner agent settings (for multi-agent)
    train_both_agents: bool = True      # Train both agents vs one
    share_network: bool = False         # Share network between agents
    
    # Custom parameters (add your own!)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration"""
        # Auto-detect device
        if self.device == "auto":
            import torch
            self.device = "cuda" if torch.cuda.is_available() and self.use_gpu else "cpu"
        
        # Create directories
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        config_dict = {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls(**config_dict)
    
    def update(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_params[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


# ========== PRESET CONFIGURATIONS ==========

def get_fast_debug_config() -> TrainingConfig:
    """Fast config for debugging (small networks, few episodes)"""
    config = TrainingConfig()
    config.num_episodes = 100
    config.max_steps_per_episode = 500
    config.hidden_dim = 64
    config.buffer_size = 1000
    config.log_frequency = 10
    config.save_frequency = 50
    config.render_training = False
    return config


def get_standard_config() -> TrainingConfig:
    """Standard config for typical training"""
    config = TrainingConfig()
    config.num_episodes = 10000
    config.max_steps_per_episode = 3000
    config.hidden_dim = 256
    config.buffer_size = 100000
    config.log_frequency = 100
    config.save_frequency = 500
    config.use_wandb = False
    return config


def get_performance_config() -> TrainingConfig:
    """High-performance config with parallel environments"""
    config = TrainingConfig()
    config.num_episodes = 50000
    config.max_steps_per_episode = 3000
    config.hidden_dim = 512
    config.num_layers = 3
    config.buffer_size = 500000
    config.batch_size = 256
    config.num_parallel_envs = 16
    config.use_gpu = True
    config.log_frequency = 100
    config.save_frequency = 1000
    config.render_training = False
    config.use_wandb = True
    return config


def get_curriculum_config() -> TrainingConfig:
    """Config with curriculum learning"""
    config = TrainingConfig()
    config.num_episodes = 20000
    config.use_curriculum = True
    config.reward_function = "curriculum"
    config.epsilon_decay_steps = 5000
    config.log_frequency = 100
    config.save_frequency = 500
    return config


def get_cooperation_config() -> TrainingConfig:
    """Config optimized for cooperation"""
    config = TrainingConfig()
    config.num_episodes = 15000
    config.reward_function = "cooperation"
    config.train_both_agents = True
    config.share_network = False
    config.log_frequency = 100
    config.save_frequency = 500
    return config


# ========== CONFIG REGISTRY ==========
PRESET_CONFIGS = {
    'debug': get_fast_debug_config,
    'standard': get_standard_config,
    'performance': get_performance_config,
    'curriculum': get_curriculum_config,
    'cooperation': get_cooperation_config
}


def get_config(preset: str = 'standard', **overrides) -> TrainingConfig:
    """
    Get configuration with optional overrides
    
    Args:
        preset: Preset name ('debug', 'standard', 'performance', etc.)
        **overrides: Parameters to override
    
    Returns:
        TrainingConfig instance
    
    Example:
        config = get_config('standard', learning_rate=1e-4, batch_size=128)
    """
    if preset not in PRESET_CONFIGS:
        raise ValueError(f"Unknown preset: {preset}. Choose from {list(PRESET_CONFIGS.keys())}")
    
    config = PRESET_CONFIGS[preset]()
    config.update(**overrides)
    
    return config


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    print("=" * 60)
    print("TRAINING CONFIGURATION EXAMPLES")
    print("=" * 60)
    
    # Example 1: Standard config
    print("\n1. Standard Config:")
    config = get_standard_config()
    print(f"   Episodes: {config.num_episodes}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Device: {config.device}")
    
    # Example 2: Debug config with overrides
    print("\n2. Debug Config (with overrides):")
    config = get_config('debug', learning_rate=1e-3, batch_size=32)
    print(f"   Episodes: {config.num_episodes}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.batch_size}")
    
    # Example 3: Performance config
    print("\n3. Performance Config:")
    config = get_performance_config()
    print(f"   Episodes: {config.num_episodes}")
    print(f"   Parallel envs: {config.num_parallel_envs}")
    print(f"   Hidden dim: {config.hidden_dim}")
    print(f"   Buffer size: {config.buffer_size}")
    
    # Example 4: Custom config
    print("\n4. Custom Config:")
    config = TrainingConfig()
    config.num_episodes = 20000
    config.learning_rate = 1e-4
    config.reward_function = "cooperation"
    config.use_wandb = True
    config.custom_params['my_param'] = 42
    print(f"   Episodes: {config.num_episodes}")
    print(f"   Reward function: {config.reward_function}")
    print(f"   Custom params: {config.custom_params}")
    
    # Example 5: Save and load
    print("\n5. Save and Load:")
    config = get_standard_config()
    config.save("test_config.json")
    loaded_config = TrainingConfig.load("test_config.json")
    print(f"   Saved and loaded successfully!")
    print(f"   Episodes: {loaded_config.num_episodes}")
    
    # Cleanup
    os.remove("test_config.json")
    
    print("\n" + "=" * 60)
    print("All available presets:")
    for name in PRESET_CONFIGS.keys():
        print(f"  - {name}")
    print("=" * 60)
