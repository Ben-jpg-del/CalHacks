"""
🎯 GYM-BASED RL TRAINING GUIDE
Complete guide for using the modular training system
"""

# ============================================================================
# PART 1: QUICK START (3 Steps)
# ============================================================================

"""
Step 1: Install dependencies
----------------------------
pip install numpy pygame torch gymnasium wandb

Step 2: Choose your training configuration
------------------------------------------
python train_modular.py defaults          # Standard training
python train_modular.py custom-rewards    # Custom rewards
python train_modular.py debug             # Fast debug mode

Step 3: Watch your agents train!
--------------------------------
That's it! The system handles everything else.
"""

# ============================================================================
# PART 2: CUSTOMIZING REWARDS (Most Common Task)
# ============================================================================

"""
Example 1: Simple Reward Changes
---------------------------------
Edit reward_functions.py or create inline:
"""

from reward_functions import RewardConfig

# Create custom reward config
reward_config = RewardConfig()

# Customize values
reward_config.both_win_reward = 500.0      # Increase win reward
reward_config.death_penalty = -200.0       # Harsher death penalty
reward_config.step_penalty = -0.05         # Encourage speed
reward_config.distance_weight = 0.01       # Stronger goal shaping
reward_config.cooperation_bonus = 20.0     # Emphasize teamwork

# Use in training
from train_modular import ModularTrainer
from training_config import get_config

config = get_config('standard')
config.reward_function = "dense"  # or "sparse", "safety", "cooperation"

trainer = ModularTrainer(config, reward_config)
trainer.train()

"""
Example 2: Create Your Own Reward Function
-------------------------------------------
Edit reward_functions.py and add new class:
"""

from reward_functions import BaseRewardFunction

class MyCustomReward(BaseRewardFunction):
    def calculate_rewards(self, fire_agent, water_agent, fire_action, water_action,
                         fire_died, water_died, fire_won, water_won,
                         bridge_activated, gate_activated, exits, hazards):
        fire_reward = 0.0
        water_reward = 0.0
        
        # Your custom logic here!
        if fire_won and water_won:
            fire_reward = 1000.0
            water_reward = 1000.0
        
        # Add distance-based shaping
        fire_exit = exits['fire']
        fire_dist = ((fire_agent.rect.centerx - fire_exit.centerx)**2 + 
                     (fire_agent.rect.centery - fire_exit.centery)**2)**0.5
        fire_reward -= fire_dist * 0.001
        
        return fire_reward, water_reward

# Register your reward function
from reward_functions import REWARD_FUNCTIONS
REWARD_FUNCTIONS['my_custom'] = MyCustomReward

# Use it
config.reward_function = "my_custom"

# ============================================================================
# PART 3: CUSTOMIZING TRAINING PARAMETERS
# ============================================================================

"""
Example 1: Edit Hyperparameters
--------------------------------
"""

from training_config import TrainingConfig

config = TrainingConfig()

# Basic settings
config.num_episodes = 50000
config.max_steps_per_episode = 5000

# Learning settings
config.learning_rate = 1e-4
config.batch_size = 128
config.gamma = 0.99

# Network architecture
config.hidden_dim = 512
config.num_layers = 3
config.use_dueling = True

# Exploration
config.epsilon_start = 1.0
config.epsilon_end = 0.01
config.epsilon_decay = 0.995

# Logging
config.log_frequency = 100
config.save_frequency = 500
config.use_wandb = True

"""
Example 2: Use Preset Configurations
-------------------------------------
"""

from training_config import get_config

# Available presets:
config = get_config('standard')      # Balanced settings
config = get_config('debug')         # Fast for testing
config = get_config('performance')   # High-performance
config = get_config('curriculum')    # With curriculum learning
config = get_config('cooperation')   # Emphasizes teamwork

# Override specific parameters
config = get_config('standard', 
                    learning_rate=1e-4,
                    batch_size=256,
                    use_wandb=True)

"""
Example 3: Save and Load Configurations
----------------------------------------
"""

# Save config
config.save("my_config.json")

# Load config
from training_config import TrainingConfig
config = TrainingConfig.load("my_config.json")

# Use in training
from train_modular import train_from_config_file
train_from_config_file("my_config.json")

# ============================================================================
# PART 4: USING WITH STABLE-BASELINES3
# ============================================================================

"""
Example: Train with Stable-Baselines3 PPO
------------------------------------------
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_wrapper import FireWaterGymEnv

# Create environment
def make_env():
    return FireWaterGymEnv(agent_type="fire")

# Vectorize environment
env = DummyVecEnv([make_env for _ in range(4)])

# Create PPO agent
model = PPO(
    "MlpPolicy", 
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)

# Train
model.learn(total_timesteps=1000000)

# Save
model.save("ppo_firewater")

# Load and evaluate
model = PPO.load("ppo_firewater")
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)

"""
Example: Multi-Agent with Stable-Baselines3
--------------------------------------------
"""

from gym_wrapper import MultiAgentFireWaterGymEnv

# Multi-agent wrapper
env = MultiAgentFireWaterGymEnv()

# Train both agents
# (Note: SB3 doesn't natively support multi-agent, so you'd need to 
#  alternate training or use a library like RLlib)

# ============================================================================
# PART 5: IMPLEMENTING YOUR OWN RL AGENT
# ============================================================================

"""
Template: Custom RL Agent
--------------------------
"""

class MyRLAgent:
    def __init__(self, state_dim=52, action_dim=6):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Initialize your model here
        # self.model = YourNeuralNetwork()
        # self.optimizer = torch.optim.Adam(self.model.parameters())
    
    def select_action(self, observation, training=True):
        """
        Select action based on observation
        
        Args:
            observation: State vector (numpy array, shape (52,))
            training: If True, use exploration; if False, use greedy policy
        
        Returns:
            action: Integer from 0-5
        """
        # Your action selection logic here
        # Example: epsilon-greedy
        if training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.action_dim)
        else:
            # Use your model to predict best action
            # with torch.no_grad():
            #     q_values = self.model(torch.FloatTensor(observation))
            #     return q_values.argmax().item()
            pass
    
    def update(self, obs, action, reward, next_obs, done):
        """
        Update agent based on experience
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            done: Whether episode ended
        """
        # Your learning update here
        # 1. Store experience in replay buffer
        # 2. Sample batch from buffer
        # 3. Compute loss
        # 4. Backpropagate
        # 5. Update target network
        pass
    
    def save(self, path):
        """Save model checkpoint"""
        # torch.save(self.model.state_dict(), path)
        pass
    
    def load(self, path):
        """Load model checkpoint"""
        # self.model.load_state_dict(torch.load(path))
        pass

"""
Using Your Custom Agent
-----------------------
"""

from train_modular import ModularTrainer
from training_config import get_config

# Create your agents
fire_agent = MyRLAgent()
water_agent = MyRLAgent()

# Create trainer with your agents
config = get_config('standard')
trainer = ModularTrainer(config, fire_agent=fire_agent, water_agent=water_agent)
trainer.train()

# ============================================================================
# PART 6: ADVANCED FEATURES
# ============================================================================

"""
Example 1: Parallel Environments for Speed
-------------------------------------------
"""

from game_environment import FireWaterEnv
import numpy as np

class VectorizedEnv:
    def __init__(self, num_envs=16):
        self.envs = [FireWaterEnv() for _ in range(num_envs)]
        self.num_envs = num_envs
    
    def reset(self):
        results = [env.reset() for env in self.envs]
        fire_obs = np.stack([r[0] for r in results])
        water_obs = np.stack([r[1] for r in results])
        return fire_obs, water_obs
    
    def step(self, fire_actions, water_actions):
        results = []
        for env, f_act, w_act in zip(self.envs, fire_actions, water_actions):
            results.append(env.step(f_act, w_act))
        
        fire_obs = np.stack([r[0][0] for r in results])
        water_obs = np.stack([r[0][1] for r in results])
        rewards = np.stack([r[1] for r in results])
        dones = np.stack([r[2] for r in results])
        infos = [r[3] for r in results]
        
        return (fire_obs, water_obs), rewards, dones, infos

# Use vectorized env for 10-20x speedup
vec_env = VectorizedEnv(num_envs=16)
fire_obs, water_obs = vec_env.reset()  # Shape: (16, 52)

"""
Example 2: Curriculum Learning
-------------------------------
"""

from training_config import get_config

config = get_config('curriculum')

# The curriculum automatically adjusts difficulty
# Early: shorter episodes, sparse rewards
# Later: longer episodes, dense rewards

trainer = ModularTrainer(config)
trainer.train()

"""
Example 3: Custom Evaluation
-----------------------------
"""

def evaluate_agents(fire_agent, water_agent, num_episodes=100):
    from game_environment import FireWaterEnv
    
    env = FireWaterEnv()
    successes = 0
    total_reward = 0
    
    for _ in range(num_episodes):
        fire_obs, water_obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            fire_action = fire_agent.select_action(fire_obs, training=False)
            water_action = water_agent.select_action(water_obs, training=False)
            
            (fire_obs, water_obs), rewards, dones, info = env.step(
                fire_action, water_action
            )
            
            episode_reward += sum(rewards)
            done = dones[0] or dones[1]
        
        if info['both_won']:
            successes += 1
        total_reward += episode_reward
    
    return {
        'success_rate': successes / num_episodes,
        'avg_reward': total_reward / num_episodes
    }

# Use it
results = evaluate_agents(fire_agent, water_agent)
print(f"Success Rate: {results['success_rate']:.2%}")

# ============================================================================
# PART 7: COMMON PATTERNS
# ============================================================================

"""
Pattern 1: Quick Experiment
----------------------------
"""

# Try different reward functions
for reward_fn in ['sparse', 'dense', 'safety', 'cooperation']:
    config = get_config('debug')
    config.reward_function = reward_fn
    config.save_dir = f"checkpoints_{reward_fn}"
    
    trainer = ModularTrainer(config)
    trainer.train()

"""
Pattern 2: Hyperparameter Sweep
--------------------------------
"""

learning_rates = [1e-3, 3e-4, 1e-4]
batch_sizes = [32, 64, 128]

for lr in learning_rates:
    for bs in batch_sizes:
        config = get_config('standard')
        config.learning_rate = lr
        config.batch_size = bs
        config.save_dir = f"checkpoints_lr{lr}_bs{bs}"
        
        trainer = ModularTrainer(config)
        trainer.train()

"""
Pattern 3: Resume Training
--------------------------
"""

# Save checkpoint during training
# (automatically done by ModularTrainer)

# Load and continue
fire_agent = MyRLAgent()
fire_agent.load("checkpoints/fire_ep5000.pth")

config = get_config('standard')
config.num_episodes = 10000  # Train for 5000 more

trainer = ModularTrainer(config, fire_agent=fire_agent)
trainer.train()

# ============================================================================
# PART 8: TROUBLESHOOTING
# ============================================================================

"""
Problem: Agents not learning
----------------------------
Solutions:
1. Start with dense rewards (config.reward_function = "dense")
2. Increase distance_weight in RewardConfig
3. Use smaller learning rate (config.learning_rate = 1e-4)
4. Increase exploration (config.epsilon_decay = 0.999)

Problem: Training too slow
--------------------------
Solutions:
1. Disable rendering (config.render_training = False)
2. Use vectorized environments (VectorizedEnv)
3. Reduce max_steps_per_episode
4. Use debug config for testing

Problem: Agents die immediately
-------------------------------
Solutions:
1. Use safety reward function (config.reward_function = "safety")
2. Increase death_penalty (reward_config.death_penalty = -500)
3. Add near_hazard_penalty
4. Start with curriculum learning

Problem: No cooperation
-----------------------
Solutions:
1. Use cooperation reward function
2. Increase cooperation_bonus
3. Add proximity_bonus and synchronization_bonus
4. Train both agents simultaneously

Problem: High variance in results
----------------------------------
Solutions:
1. Increase batch_size
2. Use gradient clipping (config.gradient_clip = 1.0)
3. Reduce learning rate
4. Use target networks with slower updates
"""

# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
Training Modes:
--------------
python train_modular.py defaults          # Standard
python train_modular.py custom-rewards    # Custom rewards
python train_modular.py debug             # Fast debug
python train_modular.py custom-config     # Full customization
python train_modular.py from-file config.json  # From file

Reward Functions:
----------------
- sparse: Only win/lose
- dense: Distance shaping
- safety: Avoid hazards
- cooperation: Emphasize teamwork
- curriculum: Adaptive difficulty

Config Presets:
--------------
- standard: Balanced settings
- debug: Fast for testing
- performance: High-performance
- curriculum: Progressive learning
- cooperation: Team-focused

Key Files:
---------
- training_config.py: All hyperparameters
- reward_functions.py: Reward logic
- gym_wrapper.py: Gym compatibility
- train_modular.py: Main training script
- example_dqn.py: Full DQN example
"""

print(__doc__)
