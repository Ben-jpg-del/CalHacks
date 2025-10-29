# Parallel Multi-Map Training Guide

## Overview

This guide explains how to use the new **parallel multi-map training system** to train a generalized policy that works across all maps.

## Key Features

✅ **Parallel Training**: Run multiple environments simultaneously for faster training
✅ **Multi-Map Learning**: Train on all maps at once for better generalization
✅ **Curriculum Learning**: Start easy, gradually increase difficulty
✅ **L4 GPU Optimized**: Batched operations for maximum GPU utilization
✅ **Easy to Extend**: Simple system to add new maps
✅ **Automatic Checkpointing**: Save and resume training anytime

## Architecture

### 1. Map Registry (`parallel_multi_map_env.py`)

Central registry for all maps:

```python
from parallel_multi_map_env import MapRegistry

# Get all available maps
all_maps = MapRegistry.get_all_maps()
# Returns: {'tutorial': LevelConfig, 'tower': LevelConfig, ...}

# Get specific map
tutorial = MapRegistry.get_map('tutorial')

# List all map names
names = MapRegistry.get_map_names()
# Returns: ['tutorial', 'tower']
```

### 2. Parallel Multi-Map Environment

Runs multiple environments across different maps:

```python
from parallel_multi_map_env import ParallelMultiMapEnv

# Create environment with 4 envs per map
env = ParallelMultiMapEnv(
    num_envs_per_map=4,
    map_distribution={'tutorial': 0.5, 'tower': 0.5},
    device='cuda'
)

# Returns batch tensors
fire_obs, water_obs = env.reset()
# Shape: [num_envs, obs_dim]

# Step all environments in parallel
(fire_obs_next, water_obs_next), \
(fire_rewards, water_rewards), \
(fire_dones, water_dones), \
infos = env.step(fire_actions, water_actions)
```

### 3. Curriculum Learning (Optional)

Automatically adjusts map difficulty based on training progress:

```python
from parallel_multi_map_env import CurriculumMultiMapEnv

curriculum_schedule = {
    0: {'tutorial': 1.0},                      # Start: 100% tutorial
    1000: {'tutorial': 0.7, 'tower': 0.3},    # Episode 1000: Add tower
    2000: {'tutorial': 0.5, 'tower': 0.5},    # Episode 2000: Equal split
}

env = CurriculumMultiMapEnv(
    num_envs_per_map=4,
    curriculum_schedule=curriculum_schedule,
    device='cuda'
)

# Call this at end of each episode
env.update_curriculum(episode_number)
```

## Usage

### Method 1: Jupyter Notebook (Recommended for Colab)

Use the provided notebook: `Parallel_MultiMap_Training_Colab.ipynb`

1. Open in Google Colab
2. Select L4 GPU runtime
3. Configure training parameters
4. Run all cells

### Method 2: Python Script (Local Training)

```bash
# Basic training
python train_parallel_multimap.py

# Custom configuration
python train_parallel_multimap.py \
  --episodes 5000 \
  --envs-per-map 8 \
  --reward dense \
  --device cuda

# Resume from checkpoint
python train_parallel_multimap.py \
  --resume-from checkpoints_multimap/checkpoint_ep1000

# Without curriculum learning
python train_parallel_multimap.py --no-curriculum
```

### Method 3: Programmatic Usage

```python
from train_parallel_multimap import train_parallel_multimap

train_parallel_multimap(
    num_episodes=5000,
    num_envs_per_map=8,
    use_curriculum=True,
    reward_type='dense',
    save_dir='checkpoints_multimap',
    device='cuda'
)
```

## Adding New Maps

### Step 1: Create Map File

Create a new file (e.g., `map_3.py`):

```python
"""
My Custom Map
"""

from physics_engine import Rect
from typing import List, Dict

class LevelConfig:
    def __init__(self, level_name: str = "My Custom Map"):
        self.name = level_name
        self.width = 960
        self.height = 540

        # Agent spawn positions
        self.fire_start = (100, 100, 28, 36)
        self.water_start = (200, 200, 28, 36)

        # Define platforms
        self.base_solids = [
            # Boundary walls
            Rect(-10, 0, 10, self.height),
            Rect(self.width, 0, 10, self.height),
            Rect(0, -10, self.width, 10),
            Rect(0, self.height, self.width, 10),

            # Your custom platforms
            Rect(0, 500, self.width, 40),  # Floor
            # Add more platforms...
        ]

        # Interactive elements
        self.bridge = Rect(300, 400, 200, 20)
        self.gate = Rect(500, 300, 20, 100)

        # Pressure plates
        self.plate_a = Rect(250, 484, 40, 16)
        self.plate_b = Rect(700, 484, 40, 16)

        # Hazards
        self.water_pool = Rect(400, 450, 100, 50)
        self.lava_pool = Rect(600, 450, 100, 50)

        # Exits
        self.exit_water = Rect(100, 400, 36, 36)
        self.exit_fire = Rect(800, 400, 36, 36)

    def get_solids(self, bridge_up: bool = False, gate_open: bool = False) -> List[Rect]:
        solids = list(self.base_solids)
        if bridge_up:
            solids.append(self.bridge)
        if not gate_open:
            solids.append(self.gate)
        return solids

    def get_hazards(self) -> Dict[str, Rect]:
        hazards = {}
        if self.water_pool:
            hazards['water_pool'] = self.water_pool
        if self.lava_pool:
            hazards['lava_pool'] = self.lava_pool
        return hazards

    def get_exits(self) -> Dict[str, Rect]:
        return {
            'water': self.exit_water,
            'fire': self.exit_fire
        }

    def get_switches(self) -> Dict[str, Rect]:
        return {
            'plate_a': self.plate_a,
            'plate_b': self.plate_b
        }

class LevelLibrary:
    @staticmethod
    def get_my_custom_map() -> LevelConfig:
        return LevelConfig("My Custom Map")
```

### Step 2: Register in MapRegistry

Edit `parallel_multi_map_env.py`:

```python
# At top of file, add import
from map_3 import LevelLibrary as Map3Library

# In MapRegistry class, update get_all_maps():
class MapRegistry:
    @staticmethod
    def get_all_maps():
        return {
            'tutorial': LevelLibrary.get_tutorial_level(),
            'tower': Map1Library.get_tower_level(),
            'my_custom_map': Map3Library.get_my_custom_map(),  # Add this
        }
```

### Step 3: Update Training Configuration

Now your map is automatically available! Update curriculum schedule:

```python
CURRICULUM_SCHEDULE = {
    0: {'tutorial': 1.0},
    500: {'tutorial': 0.7, 'tower': 0.3},
    1000: {'tutorial': 0.5, 'tower': 0.3, 'my_custom_map': 0.2},
    2000: {'tutorial': 0.3, 'tower': 0.3, 'my_custom_map': 0.4},
}
```

That's it! The system handles everything else automatically.

## Training Configuration

### Recommended Settings for L4 GPU

```python
# Configuration optimized for L4 GPU (24GB VRAM)
NUM_EPISODES = 5000
NUM_ENVS_PER_MAP = 8-12        # L4 can handle 8-12 per map
BATCH_SIZE = 256               # Larger batch for better GPU utilization
BUFFER_CAPACITY = 500000       # Large buffer for diverse experiences
LEARNING_RATE = 3e-4           # Standard for DQN
SAVE_FREQ = 100                # Save every 100 episodes
```

### Curriculum Schedule Examples

**Conservative (Slow progression):**
```python
{
    0: {'tutorial': 1.0},
    1000: {'tutorial': 0.8, 'tower': 0.2},
    2000: {'tutorial': 0.6, 'tower': 0.4},
    3000: {'tutorial': 0.4, 'tower': 0.6},
    4000: {'tutorial': 0.2, 'tower': 0.8},
}
```

**Aggressive (Fast progression):**
```python
{
    0: {'tutorial': 1.0},
    300: {'tutorial': 0.5, 'tower': 0.5},
    600: {'tower': 1.0},
}
```

**Multi-Map (3+ maps):**
```python
{
    0: {'tutorial': 1.0},
    500: {'tutorial': 0.6, 'tower': 0.4},
    1000: {'tutorial': 0.4, 'tower': 0.4, 'map3': 0.2},
    2000: {'tutorial': 0.3, 'tower': 0.3, 'map3': 0.4},
}
```

## Monitoring Training

### Console Output

Training prints progress every N episodes:

```
Episode 100/5000
  Avg Reward: 45.23
  Avg Length: 234.5
  Success Rate: 12.3%
  Fire Epsilon: 0.950
  Water Epsilon: 0.950
  Tutorial Success: 15.2%
  Tower Success: 8.7%
  Time: 5.2m
  Buffer: 25600
```

### Per-Map Success Tracking

The system automatically tracks success rate for each map separately, helping you:
- Identify which maps are harder
- Adjust curriculum schedule
- Monitor generalization performance

## Checkpoints

### Structure

```
checkpoints_multimap/
├── checkpoint_ep100/
│   ├── fire_agent.pth
│   └── water_agent.pth
├── checkpoint_ep200/
│   ├── fire_agent.pth
│   └── water_agent.pth
├── ...
└── final/
    ├── fire_agent.pth
    └── water_agent.pth
```

### Loading Checkpoints

```python
from train_parallel_multimap import ParallelDQNAgent

# Load agents
fire_agent = ParallelDQNAgent(device='cuda')
water_agent = ParallelDQNAgent(device='cuda')

fire_agent.load('checkpoints_multimap/final/fire_agent.pth')
water_agent.load('checkpoints_multimap/final/water_agent.pth')

# Set to evaluation mode
fire_agent.epsilon = 0.0
water_agent.epsilon = 0.0
```

### Resuming Training

```bash
python train_parallel_multimap.py \
  --resume-from checkpoints_multimap/checkpoint_ep1000 \
  --episodes 5000
```

This will continue training from episode 1000 to 5000.

## Evaluation

### Evaluate on All Maps

```python
from parallel_multi_map_env import MapRegistry
from game_environment import FireWaterEnv
import torch

# Load agents
fire_agent = ParallelDQNAgent(device='cuda')
water_agent = ParallelDQNAgent(device='cuda')
fire_agent.load('checkpoints_multimap/final/fire_agent.pth')
water_agent.load('checkpoints_multimap/final/water_agent.pth')
fire_agent.epsilon = 0.0
water_agent.epsilon = 0.0

# Test on each map
for map_name in MapRegistry.get_map_names():
    level = MapRegistry.get_map(map_name)
    env = FireWaterEnv(level=level)

    successes = 0
    for _ in range(100):
        # Run episode and track success
        # ... (see notebook for full code)

    print(f"{map_name}: {successes}% success")
```

## Visualization

The generalized policy can be visualized on any map:

```bash
# Tutorial map
python visualize.py trained \
  checkpoints_multimap/final/fire_agent.pth \
  checkpoints_multimap/final/water_agent.pth \
  --map tutorial

# Tower map
python visualize.py trained \
  checkpoints_multimap/final/fire_agent.pth \
  checkpoints_multimap/final/water_agent.pth \
  --map tower

# Your custom map
python visualize.py trained \
  checkpoints_multimap/final/fire_agent.pth \
  checkpoints_multimap/final/water_agent.pth \
  --map my_custom_map
```

## Performance Optimization Tips

### 1. GPU Utilization
- Use 8-12 environments per map on L4 GPU
- Increase batch size to 256-512
- Ensure device='cuda' everywhere

### 2. Training Speed
- Start with curriculum learning (easier maps first)
- Use dense rewards for faster learning
- Increase num_envs_per_map for more parallelism

### 3. Generalization
- Train on multiple maps simultaneously
- Use diverse map distribution
- Train for longer (5000+ episodes)

### 4. Stability
- Use target network updates (every 500 steps)
- Gradient clipping (max norm 10.0)
- Larger replay buffer (500k)

## Troubleshooting

### OOM (Out of Memory)
- Reduce `num_envs_per_map`
- Reduce `batch_size`
- Reduce `buffer_capacity`

### Poor Generalization
- Train on all maps simultaneously (not sequentially)
- Increase training episodes
- Use curriculum learning
- Balance map distribution

### Slow Training
- Increase `num_envs_per_map`
- Use L4 GPU instead of T4
- Reduce logging frequency
- Increase batch size

### Maps Not Found
- Check MapRegistry imports in `parallel_multi_map_env.py`
- Ensure map file exists
- Verify map name in get_all_maps()

## Comparison: Single vs Multi-Map Training

| Aspect | Single Map | Multi-Map Parallel |
|--------|------------|-------------------|
| Training Time | Baseline | 2-3x faster |
| Generalization | Poor | Excellent |
| GPU Utilization | Low (1 env) | High (8-16+ envs) |
| Easy to Add Maps | No (retrain) | Yes (automatic) |
| Curriculum Learning | No | Yes |
| Overfitting Risk | High | Low |

## Best Practices

1. **Start with curriculum learning** - Easier maps first
2. **Monitor per-map success** - Identify weak areas
3. **Save frequently** - Every 100 episodes
4. **Evaluate on all maps** - Test generalization
5. **Balance map distribution** - Don't favor one map too much
6. **Use dense rewards** - Faster learning signal
7. **Train longer** - Generalization takes time (5000+ episodes)
8. **Add maps incrementally** - Start with 2, gradually add more

## FAQ

**Q: Can I train on just one map?**
A: Yes, set `map_distribution={'tutorial': 1.0}` or use the old training scripts.

**Q: How many maps can I add?**
A: Unlimited! Just register in MapRegistry. The system scales automatically.

**Q: Does this work with custom reward functions?**
A: Yes, pass `reward_function` to the environment constructor.

**Q: Can I change map distribution during training?**
A: Yes, use `CurriculumMultiMapEnv` with a schedule.

**Q: What if I don't have a GPU?**
A: It works on CPU, but will be slower. Reduce `num_envs_per_map` to 2-4.

**Q: Can I resume training with different map distribution?**
A: Yes, but the agents were trained on the old distribution. Best to continue with same or gradually shift.

## Files Overview

- `parallel_multi_map_env.py` - Multi-map environment & curriculum learning
- `train_parallel_multimap.py` - Training script with parallel execution
- `Parallel_MultiMap_Training_Colab.ipynb` - Jupyter notebook for Colab
- `PARALLEL_MULTIMAP_GUIDE.md` - This guide

## Next Steps

1. **Create your first custom map** (follow "Adding New Maps" section)
2. **Train with default settings** (use notebook or script)
3. **Evaluate generalization** (test on all maps)
4. **Iterate on curriculum** (adjust difficulty progression)
5. **Add more maps** (scale to 3, 4, 5+ maps)

Happy training! 🚀
