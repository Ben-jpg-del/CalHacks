# Map Selection Guide

This guide explains how to select different maps for training and visualization.

## Available Maps

| Map Name | Description | File | Platforms | Hazards |
|----------|-------------|------|-----------|---------|
| `tutorial` | Default horizontal platforming level | map_config.py | 11 | Water pool |
| `tower` | Custom sketch level (based on user design) | map_1.py | 11 | Water pool, Lava pool |

## Training with Different Maps

### Using train_stage_milestone_dqn.py

The training script now supports map selection via the `--map` argument:

```bash
# Train on tutorial map (default)
python train_stage_milestone_dqn.py

# Train on custom tower map
python train_stage_milestone_dqn.py --map tower

# Train on tower map with W&B logging
python train_stage_milestone_dqn.py --map tower --wandb

# Resume training on tower map from episode 400
python train_stage_milestone_dqn.py --map tower --resume 400

# Full example: Resume on tower with W&B
python train_stage_milestone_dqn.py --map tower --resume 400 --wandb --project my-project
```

### Command-Line Arguments

- `--map {tutorial,tower}` or `-m {tutorial,tower}` - Select map (default: tutorial)
- `--wandb` or `-w` - Enable Weights & Biases logging
- `--resume EPISODE` or `-r EPISODE` - Resume from checkpoint
- `--project NAME` - W&B project name (default: firewater-staged-dqn)

### Get Help

```bash
python train_stage_milestone_dqn.py --help
```

## Visualization with Different Maps

### Using visualize.py

The visualization script also supports map selection:

```bash
# Play tutorial map (default)
python visualize.py human

# Play custom tower map
python visualize.py human --map tower

# Watch random agents on tower map
python visualize.py random 10 --map tower

# List all available maps
python visualize.py --list-maps

# Slow motion visualization
python visualize.py human --map tower --fps 30
```

### Available Modes

- `human` - Manual keyboard control
- `random` - Random agent policy
- `trained` - Visualize trained agents

## Map Characteristics

### Tutorial Map

**Layout:** Horizontal platforming
- Spawn: Fire and water start on opposite sides
- Goal: Navigate across platforms to reach exits
- Cooperation: Water activates bridge, fire opens gate
- Hazards: Water pool in center (kills fire)
- Difficulty: Medium

### Tower Map (Custom Sketch Level)

**Layout:** Multi-level horizontal
- Spawn: Both agents start together on bottom left
- Goal: Navigate up 3 levels to reach exits on top right
- Cooperation:
  - Water activates bridge to connect middle platforms
  - Fire opens gate to unlock exit platform
- Hazards:
  - Water pool on middle platform (kills fire)
  - Lava pool on bottom-right platform (kills water)
- Difficulty: Medium-Hard

## Creating Your Own Maps

To add a new map:

1. Create a new map file (e.g., `map_2.py`) following the structure in `map_1.py`
2. Update `train_stage_milestone_dqn.py`:
   - Import your map library
   - Add map selection logic in the `train_dqn_with_staged_rewards()` function
   - Add to the `--map` choices in the argument parser
3. Update `visualize.py`:
   - Import your map library
   - Add to `get_level_from_name()` function
   - Add to `list_available_maps()` function

## Examples

### Training Examples

```bash
# Quick test on tutorial map
python train_stage_milestone_dqn.py

# Full training run on custom map
python train_stage_milestone_dqn.py --map tower --wandb

# Continue training on tower map
python train_stage_milestone_dqn.py --map tower --resume 500
```

### Visualization Examples

```bash
# Play tutorial level
python visualize.py human

# Play custom level
python visualize.py human --map tower

# Debug with slow motion
python visualize.py human --map tower --fps 20

# Watch random agents struggle
python visualize.py random 5 --map tower
```

## Troubleshooting

### PyTorch Not Installed

Training requires PyTorch:
```bash
pip install torch
```

### Map Not Loading

- Ensure `map_1.py` exists in the same directory
- Check that the file has a `LevelLibrary` class with `get_tower_level()` method
- Verify the map name is spelled correctly (case-insensitive)

### Pygame Not Installed

Visualization requires Pygame:
```bash
pip install pygame
```

## Map Statistics Comparison

| Metric | Tutorial | Tower |
|--------|----------|-------|
| Width x Height | 960x540 | 960x540 |
| Base Platforms | 11 | 11 |
| Dynamic Platforms | 1 bridge, 1 gate | 1 bridge, 1 gate |
| Pressure Plates | 2 | 2 |
| Hazards | 1 (water) | 2 (water, lava) |
| Spawn Location | Opposite sides | Together (bottom-left) |
| Exit Location | Opposite sides | Together (top-right) |
| Vertical Levels | 2-3 | 3 |

---

For more information about map structure, see `map_config.py` and `map_1.py`.
