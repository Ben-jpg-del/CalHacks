# Fire & Water RL Training Package

A modular, efficient reinforcement learning environment for cooperative multi-agent training. The architecture separates game logic from visualization, enabling 10-100x faster training.

## Demo

https://github.com/user-attachments/assets/935f8d54-e3ae-4155-b117-8b1f2d5c1abb

## Architecture

The package is organized into independent, reusable modules:

```
firewater_rl_package/
├── physics_engine.py      # Pure Python physics (no Pygame)
├── map_config.py          # Level definitions and geometry
├── game_environment.py    # Gym-like RL environment
├── train_fast.py          # Headless training script
├── visualize.py           # Pygame visualization
├── example_dqn.py         # Example RL agent implementation
└── README.md              # This file
```

## Key Features

- **Decoupled Design**: Physics engine works without Pygame
- **Fast Training**: 10-100x speedup by disabling visualization
- **Gym-like Interface**: Standard `reset()` and `step()` API
- **Modular**: Easy to swap physics, maps, or RL algorithms
- **Flexible**: Use with any RL framework (PyTorch, TensorFlow, JAX)

## Installation

### Requirements
```bash
pip install numpy pygame
```

### Optional (for faster training)
```bash
pip install torch wandb  # PyTorch + experiment tracking
```

## Quick Start

### 1. Human Play Mode
Test the game mechanics manually:
```bash
python visualize.py human
```

**Controls:**
- Fire Agent (Red): A/D = move, W = jump
- Water Agent (Blue): Arrow keys = move/jump
- R = reset, ESC = quit

### 2. Visualize Random Agents
Watch random agents play:
```bash
python visualize.py random 10  # 10 episodes
```

### 3. Fast Headless Training
Train without visualization (maximum speed):
```bash
python train_fast.py
```

### 4. Benchmark Performance
Test environment speed:
```bash
python train_fast.py benchmark
```

Expected output: **10,000+ steps/second** on modern hardware

## Implementing Your Own RL Agent

The package provides a clean interface for implementing any RL algorithm:

### Example: DQN Agent

```python
from game_environment import FireWaterEnv
import torch
import torch.nn as nn

# 1. Define your neural network
class DQN(nn.Module):
    def __init__(self, state_dim=52, action_dim=6):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# 2. Create your agent class
class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.epsilon = 0.1
    
    def select_action(self, observation):
        if np.random.random() < self.epsilon:
            return np.random.randint(0, 6)
        else:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(observation)
                q_values = self.model(obs_tensor)
                return q_values.argmax().item()
    
    def update(self, obs, action, reward, next_obs, done):
        # Implement your learning update here
        pass

# 3. Train your agent
env = FireWaterEnv()
fire_agent = DQNAgent()
water_agent = DQNAgent()

for episode in range(10000):
    fire_obs, water_obs = env.reset()
    done = False
    
    while not done:
        fire_action = fire_agent.select_action(fire_obs)
        water_action = water_agent.select_action(water_obs)
        
        (fire_next_obs, water_next_obs), rewards, dones, info = env.step(
            fire_action, water_action
        )
        
        fire_agent.update(fire_obs, fire_action, rewards[0], fire_next_obs, dones[0])
        water_agent.update(water_obs, water_action, rewards[1], water_next_obs, dones[1])
        
        fire_obs, water_obs = fire_next_obs, water_next_obs
        done = dones[0] or dones[1]
```

See `example_dqn.py` for a complete implementation.

## Environment Details

### Action Space
Each agent has 6 discrete actions:
- 0: Idle
- 1: Move left
- 2: Move right
- 3: Jump
- 4: Move left + jump
- 5: Move right + jump

### Observation Space
52-dimensional state vector:
- **[0-4]**: Agent state (position, velocity, grounded)
- **[5-9]**: Partner state
- **[10-11]**: Switch states (bridge, gate)
- **[12-29]**: Radial clearance (18 rays)
- **[30-31]**: Distance/angle to exit
- **[32-51]**: Reserved for future features

### Reward Structure
The default reward function includes:
- **+100**: Both agents reach exits (win)
- **-100**: Agent dies in hazard
- **+10**: Activate cooperation mechanism
- **-0.01**: Small step penalty (encourages efficiency)

**Customize rewards** in `game_environment.py` → `_calculate_rewards()`

## Training Tips

### 1. Start with Curriculum Learning
```bash
python train_fast.py curriculum
```

Progressively increase episode length to help agents learn.

### 2. Tune Reward Shaping
The default rewards are basic. Consider adding:
- Distance-based shaping (guide agents toward goals)
- Velocity penalties (reduce jittery movement)
- Cooperation bonuses (reward helping partner)

### 3. Use Replay Buffers
For stable learning, implement experience replay:
```python
from collections import deque
replay_buffer = deque(maxlen=100000)
```

### 4. Monitor Training
Enable Weights & Biases logging:
```python
train_headless(use_wandb=True)
```

## Performance Comparison

| Mode | Speed | Use Case |
|------|-------|----------|
| Headless Training | 10,000+ steps/sec | Rapid prototyping, hyperparameter search |
| Visualized Training | 100-500 steps/sec | Debugging, watching agent behavior |
| Human Play | 60 FPS | Testing game mechanics |

## Advanced Usage

### Custom Levels
```python
from map_config import LevelConfig, Rect

# Create custom level
level = LevelConfig("Custom")
level.fire_start = (100, 100, 28, 36)
level.water_start = (200, 200, 28, 36)
level.base_solids = [
    Rect(0, 300, 960, 20),  # Floor
    # Add more platforms...
]

# Use in environment
env = FireWaterEnv(level=level)
```

### Multi-GPU Training
```python
# Distribute environments across GPUs
import torch.multiprocessing as mp

def train_worker(gpu_id):
    torch.cuda.set_device(gpu_id)
    env = FireWaterEnv()
    # Train on this GPU...

if __name__ == "__main__":
    mp.spawn(train_worker, nprocs=4)
```

### Parallel Environments
```python
# Run multiple environments in parallel
envs = [FireWaterEnv() for _ in range(16)]

observations = [env.reset() for env in envs]
# Batch process actions...
```

## Contributing

To add new features:

1. **New level**: Edit `map_config.py`
2. **New physics**: Modify `physics_engine.py`
3. **New reward**: Update `game_environment.py`
4. **New visualization**: Extend `visualize.py`
