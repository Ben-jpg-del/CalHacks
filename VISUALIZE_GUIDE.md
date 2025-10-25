# Visualization Guide

This guide explains how to use the updated `visualize.py` script to play or visualize agents on different maps.

## Quick Start

### 1. Play the Tutorial Level (Default)
```bash
python visualize.py human
```

### 2. Play the Custom Tower Level
```bash
python visualize.py human --map tower
```

### 3. List All Available Maps
```bash
python visualize.py --list-maps
```

## Available Maps

| Map Name | Description | Layout |
|----------|-------------|--------|
| `tutorial` | Default tutorial level | Horizontal platforming with cooperation puzzles |
| `tower` | The Tower Ascent | Vertical climbing challenge (from map_1.py) |
| `empty` | Empty sandbox | Just boundary walls for testing |

## Usage Examples

### Human Play Mode

Play the game yourself with keyboard controls:

```bash
# Play tutorial level (default)
python visualize.py human

# Play tower level
python visualize.py human --map tower

# Play empty level
python visualize.py human --map empty
```

**Controls:**
- **Fire Agent (Red):** A/D = move, W = jump
- **Water Agent (Blue):** Arrow keys = move/jump
- **R** = Reset level
- **ESC** = Quit

### Random Agents Mode

Watch random agents play for testing/debugging:

```bash
# Watch 5 episodes on tutorial level
python visualize.py random 5

# Watch 10 episodes on tower level
python visualize.py random 10 --map tower

# Watch 20 episodes on tower level at 30 FPS
python visualize.py random 20 --map tower --fps 30
```

### Trained Agent Mode

Visualize your trained RL agents:

```bash
# Visualize trained agent on tutorial level
python visualize.py trained checkpoints/model.pth

# Visualize trained agent on tower level
python visualize.py trained checkpoints/model.pth --map tower
```

**Note:** You'll need to implement agent loading in the `visualize_trained_agent()` function.

## Command-Line Arguments

### Positional Arguments

- `mode` - Visualization mode (optional, default: `human`)
  - `human` - Manual keyboard control
  - `random` - Random agent policy
  - `trained` - Load and visualize trained agents

- `episodes_or_model` - Context-dependent argument
  - For `random` mode: Number of episodes to run (default: 5)
  - For `trained` mode: Path to model checkpoint file

### Optional Arguments

- `--map MAP` or `-m MAP` - Select which map to use
  - Options: `tutorial`, `tower`, `empty`
  - Default: `tutorial`

- `--list-maps` - Display all available maps and exit

- `--fps FPS` - Set frame rate for visualization
  - Default: 60
  - Lower values slow down the visualization

- `--help` or `-h` - Show help message

## Map Selection Details

### Tutorial Level
- **File:** `map_config.py`
- **Description:** Default horizontal platforming level
- **Features:**
  - Water must activate bridge for fire
  - Fire must open gate for water
  - Water pool hazard (kills fire)
  - Basic cooperation mechanics

### Tower Level (Custom)
- **File:** `map_1.py`
- **Description:** "The Tower Ascent" - Vertical climbing puzzle
- **Features:**
  - 5 vertical levels to climb
  - 29 platforms
  - Both water pool and lava pool hazards
  - Bridge and gate cooperation mechanics
  - More challenging than tutorial

### Empty Level
- **File:** `map_config.py`
- **Description:** Minimal level for testing
- **Features:**
  - Just boundary walls
  - No platforms, hazards, or objectives
  - Useful for physics testing

## Adding Your Own Maps

To add a new map to the visualization system:

1. **Create your map file** (e.g., `map_2.py`) following the structure in `map_1.py`

2. **Import it in visualize.py:**
   ```python
   try:
       from map_2 import LevelLibrary as Map2Library
       MAP2_AVAILABLE = True
   except ImportError:
       MAP2_AVAILABLE = False
   ```

3. **Add it to `get_level_from_name()` function:**
   ```python
   elif level_name == "yourmap" and MAP2_AVAILABLE:
       return Map2Library.get_your_level()
   ```

4. **Add it to `list_available_maps()` function:**
   ```python
   if MAP2_AVAILABLE:
       print("  yourmap   - Description of your map")
   ```

## Troubleshooting

### "map_1.py not found" Warning

If you see this warning, the custom tower level is not available. Make sure:
- `map_1.py` exists in the same directory as `visualize.py`
- The file has the correct structure with a `LevelLibrary` class

### Pygame Not Installed

```bash
pip install pygame
```

### Game Runs Too Fast/Slow

Adjust the FPS:
```bash
python visualize.py human --map tower --fps 30  # Slower
python visualize.py human --map tower --fps 120 # Faster
```

## Integration with Training

The training script (`train_stage_milestone_dqn.py`) has also been updated to use custom maps. See the training documentation for details on training agents on different maps.

## Examples

```bash
# Quick test on tower level
python visualize.py human --map tower

# Watch random agents struggle on tower level
python visualize.py random 20 --map tower --fps 30

# Test physics on empty level
python visualize.py human --map empty

# Show all available maps
python visualize.py --list-maps
```

## Tips

- **Learning a new map:** Start with human play mode to understand the layout
- **Testing agents:** Use random mode to verify the map works correctly
- **Debugging:** Use empty map to test basic physics without obstacles
- **Slow motion:** Lower FPS to see agent behavior more clearly

---

For more information about creating custom maps, see `map_config.py` and `map_1.py`.
