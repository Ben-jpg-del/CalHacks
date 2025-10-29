# Map 2 Integration - Complete Guide

## ✅ Changes Made (Already Done for You!)

I've updated the following files to integrate your `map_2.py`:

### 1. `parallel_multi_map_env.py`
- ✅ Added import: `from map_2 import LevelLibrary as Map2Library`
- ✅ Added to MapRegistry: `'map2': Map2Library.get_map2_level()`

### 2. `visualize.py`
- ✅ Added import with error handling for map_2
- ✅ Updated `get_level_from_name()` to support "map2"
- ✅ Updated `list_available_maps()` to show map2

## 📝 What You Need in Your `map_2.py`

Your `map_2.py` file needs to follow this structure:

```python
"""
Your Custom Map 2
Description of your map
"""

from physics_engine import Rect
from typing import List, Dict


class LevelConfig:
    """Level configuration for Map 2"""

    def __init__(self, level_name: str = "My Map 2"):
        self.name = level_name
        self.width = 960
        self.height = 540

        # Agent spawn positions (x, y, width, height)
        self.fire_start = (100, 450, 28, 36)    # Red agent start
        self.water_start = (50, 450, 28, 36)    # Blue agent start

        # Define all static geometry (platforms, walls, etc.)
        self.base_solids = [
            # Boundary walls (10px thick)
            Rect(-10, 0, 10, self.height),          # Left wall
            Rect(self.width, 0, 10, self.height),   # Right wall
            Rect(0, -10, self.width, 10),           # Top wall
            Rect(0, self.height, self.width, 10),   # Bottom wall

            # Your custom platforms
            Rect(0, 500, self.width, 40),           # Main floor
            # Add more platforms here...
            # Rect(x, y, width, height),
        ]

        # Interactive elements
        self.bridge = Rect(300, 400, 200, 20)       # Bridge (activated by water on plate_a)
        self.gate = Rect(500, 300, 20, 100)         # Gate (opens when fire hits plate_b)

        # Pressure plates
        self.plate_a = Rect(250, 484, 40, 16)       # Water activates -> bridge appears
        self.plate_b = Rect(700, 484, 40, 16)       # Fire activates -> gate opens

        # Hazards
        self.water_pool = Rect(400, 450, 100, 50)   # Kills fire agent
        self.lava_pool = Rect(600, 450, 100, 50)    # Kills water agent

        # Exit zones
        self.exit_water = Rect(100, 400, 36, 36)    # Water agent goal (blue)
        self.exit_fire = Rect(800, 400, 36, 36)     # Fire agent goal (red)

    def get_solids(self, bridge_up: bool = False, gate_open: bool = False) -> List[Rect]:
        """Get list of active solid obstacles based on game state"""
        solids = list(self.base_solids)

        if bridge_up:
            solids.append(self.bridge)

        if not gate_open:
            solids.append(self.gate)

        return solids

    def get_hazards(self) -> Dict[str, Rect]:
        """Get dictionary of hazard zones"""
        hazards = {}
        if self.water_pool:
            hazards['water_pool'] = self.water_pool
        if self.lava_pool:
            hazards['lava_pool'] = self.lava_pool
        return hazards

    def get_exits(self) -> Dict[str, Rect]:
        """Get dictionary of exit zones"""
        return {
            'water': self.exit_water,
            'fire': self.exit_fire
        }

    def get_switches(self) -> Dict[str, Rect]:
        """Get dictionary of pressure plate zones"""
        return {
            'plate_a': self.plate_a,  # Water activates bridge
            'plate_b': self.plate_b   # Fire opens gate
        }


class LevelLibrary:
    """Collection of levels for Map 2"""

    @staticmethod
    def get_map2_level() -> LevelConfig:
        """Get Map 2 level configuration"""
        return LevelConfig("My Map 2")
```

## 🎮 How to Use Map 2

### 1. Visualization (Local)

```bash
# Human play mode
python visualize.py human --map map2

# Random agents
python visualize.py random 10 --map map2

# Trained agents
python visualize.py trained \
  checkpoints/fire_agent.pth \
  checkpoints/water_agent.pth \
  --map map2
```

### 2. Training (Jupyter Notebook)

In `Parallel_MultiMap_Training_Colab.ipynb`, update the curriculum schedule:

```python
# In Cell 4 (Configure Training)

CURRICULUM_SCHEDULE = {
    0: {'tutorial': 1.0},                                      # Start: 100% tutorial
    500: {'tutorial': 0.6, 'tower': 0.4},                     # Add tower
    1000: {'tutorial': 0.4, 'tower': 0.3, 'map2': 0.3},      # Add map2 (your new map!)
    2000: {'tutorial': 0.3, 'tower': 0.3, 'map2': 0.4},      # Focus on map2
}
```

Then run training as normal! The system will automatically:
- Train on all 3 maps simultaneously
- Track success rate for each map separately
- Create a generalized policy

### 3. Training (Python Script)

```bash
python train_parallel_multimap.py --episodes 5000 --envs-per-map 8
```

The script will automatically detect map2 and include it in training!

## 📊 Evaluation

The evaluation will automatically test on all available maps including map2:

```python
# In notebook Cell 9 (Evaluate Trained Agent)
# It automatically finds and tests map2!

# Output will show:
# Tutorial: 85% success
# Tower:    72% success
# Map2:     68% success    ← Your new map!
# Overall:  75% success
```

## 🔍 Verify Integration

Check if map2 is properly integrated:

```bash
# List available maps
python visualize.py --list-maps

# Should show:
# tutorial  - Default tutorial level
# tower     - The Tower Ascent
# map2      - Your custom map 2    ← Should appear!
# empty     - Empty level
```

## 📐 Design Tips for map_2.py

### Platform Layout
- Use `Rect(x, y, width, height)` for each platform
- Coordinate system: (0,0) is top-left, (960, 540) is bottom-right
- Make sure agents can jump between platforms (max jump height ~120px)

### Agent Spawn Points
- Fire and water start together (cooperation!)
- Start on a stable platform
- Format: `(x, y, 28, 36)` - width=28, height=36 are standard

### Pressure Plates
- Small rectangles (40x16 is standard)
- Place on platforms where agents can reach
- plate_a → bridge (typically for water)
- plate_b → gate (typically for fire)

### Hazards
- water_pool: Kills fire agent
- lava_pool: Kills water agent
- Place strategically to require cooperation

### Exits
- Size: 36x36 (standard)
- Place so both need cooperation to win
- Can be on same platform or different ones

## 🎯 Example Map Ideas

### Idea 1: Maze Layout
- Complex path with multiple dead ends
- Pressure plates open shortcuts
- Hazards block certain paths for each agent

### Idea 2: Multi-Level Tower
- Vertical ascent with 4-5 levels
- Each level has a pressure plate
- Agents must help each other climb

### Idea 3: Mirror Layout
- Symmetric design (left/right)
- Agents on opposite sides
- Must synchronize actions

## ✅ Checklist

- [ ] Created `map_2.py` with proper structure
- [ ] Defined `LevelConfig` class
- [ ] Defined `LevelLibrary` class with `get_map2_level()` method
- [ ] Set spawn points, platforms, hazards, exits
- [ ] Verified map loads: `python visualize.py --list-maps`
- [ ] Tested visualization: `python visualize.py human --map map2`
- [ ] Updated Jupyter notebook curriculum to include map2
- [ ] Trained on all 3 maps
- [ ] Evaluated generalized policy

## 🚨 Common Issues

### ImportError: cannot import name 'LevelLibrary'
- Make sure your map_2.py has a `LevelLibrary` class
- Check that `get_map2_level()` method exists

### Map not showing in list
- Verify map_2.py is in the same directory
- Check for syntax errors in map_2.py
- Run: `python -c "from map_2 import LevelLibrary; print('OK')"`

### Agents fall through floor
- Check that floor platform exists: `Rect(0, 500, 960, 40)`
- Make sure spawn points are on solid ground

### Can't visualize
- Ensure method name is exactly `get_map2_level()` (not get_map2, get_level, etc.)
- Check that LevelLibrary class name matches

## 📚 Next Steps

1. **Design your map** - Sketch it on paper first
2. **Implement map_2.py** - Use the template above
3. **Test visualization** - Make sure it works
4. **Update notebook** - Add to curriculum
5. **Train!** - Run parallel multi-map training
6. **Evaluate** - See how well agents generalize

Need help? Check:
- `map_1.py` - Example of a complete custom map
- `map_config.py` - Original tutorial map structure
- `PARALLEL_MULTIMAP_GUIDE.md` - Complete documentation

Happy map building! 🗺️
