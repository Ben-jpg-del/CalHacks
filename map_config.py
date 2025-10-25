"""
Map Configuration - Level definitions without Pygame dependency
Defines all game geometry, spawn points, and interactive elements
"""

from physics_engine import Rect
from typing import List, Dict


class LevelConfig:
    """Level configuration without Pygame dependency"""
    
    def __init__(self, level_name: str = "Tutorial"):
        self.name = level_name
        self.width = 960
        self.height = 540
        
        # Agent spawn positions (x, y, width, height)
        self.fire_start = (420, 380, 28, 36)
        self.water_start = (60, 384, 28, 36)
        
        # Define all static geometry
        self.base_solids = [
            # Boundary walls (10px thick)
            Rect(-10, 0, 10, self.height),              # Left wall
            Rect(self.width, 0, 10, self.height),       # Right wall
            Rect(0, -10, self.width, 10),               # Top wall
            Rect(0, self.height, self.width, 10),       # Bottom wall
            
            # Platform geometry
            Rect(0, 500, self.width, 40),               # Main floor
            Rect(40, 420, 240, 20),                     # Left platform
            Rect(400, 420, 120, 20),                    # Middle-left platform
            Rect(700, 420, 260, 20),                    # Right platform
            Rect(40, 340, 240, 20),                     # Upper left platform
            Rect(35, 340, 5, 160),                      # Left barrier
            Rect(925, 0, 5, 420)                        # Right barrier (black)
        ]
        
        # Interactive elements
        self.bridge = Rect(520, 400, 180, 20)           # Activated by water on plate_a
        self.gate = Rect(280, 340, 20, 120)             # Opens when fire hits plate_b
        
        # Pressure plates
        self.plate_a = Rect(220, 404, 40, 16)           # Water activates -> bridge appears
        self.plate_b = Rect(840, 404, 40, 16)           # Fire activates -> gate opens
        
        # Hazards
        self.water_pool = Rect(520, 420, 180, 80)       # Kills fire agent
        self.lava_pool = None                            # Optional: kills water agent
        
        # Exit zones
        self.exit_water = Rect(60, 308, 36, 36)         # Water agent goal
        self.exit_fire = Rect(880, 388, 36, 36)         # Fire agent goal
    
    def get_solids(self, bridge_up: bool = False, gate_open: bool = False) -> List[Rect]:
        """
        Get list of active solid obstacles based on game state
        
        Args:
            bridge_up: Whether the bridge is active
            gate_open: Whether the gate is open
            
        Returns:
            List of solid Rect objects that block movement
        """
        solids = list(self.base_solids)
        
        # Add bridge if activated
        if bridge_up:
            solids.append(self.bridge)
        
        # Add gate if not opened
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
    """Collection of pre-defined levels"""
    
    @staticmethod
    def get_tutorial_level() -> LevelConfig:
        """Get the tutorial level configuration"""
        return LevelConfig("Tutorial")
    
    @staticmethod
    def get_custom_level(name: str, config: dict) -> LevelConfig:
        """
        Create a custom level from configuration dictionary
        
        Args:
            name: Level name
            config: Dictionary containing level parameters
            
        Example config:
            {
                'width': 960,
                'height': 540,
                'fire_start': (x, y, w, h),
                'water_start': (x, y, w, h),
                'base_solids': [Rect(...), ...],
                'bridge': Rect(...),
                'gate': Rect(...),
                ...
            }
        """
        level = LevelConfig(name)
        
        # Override default values with config
        for key, value in config.items():
            if hasattr(level, key):
                setattr(level, key, value)
        
        return level
    
    @staticmethod
    def create_empty_level(width: int = 960, height: int = 540) -> LevelConfig:
        """Create an empty level with just boundary walls"""
        level = LevelConfig("Empty")
        level.width = width
        level.height = height
        level.base_solids = [
            Rect(-10, 0, 10, height),              # Left wall
            Rect(width, 0, 10, height),            # Right wall
            Rect(0, -10, width, 10),               # Top wall
            Rect(0, height, width, 10),            # Bottom wall
        ]
        level.bridge = None
        level.gate = None
        level.plate_a = None
        level.plate_b = None
        level.water_pool = None
        level.lava_pool = None
        
        return level


# Example: How to create custom levels
if __name__ == "__main__":
    # Get tutorial level
    tutorial = LevelLibrary.get_tutorial_level()
    print(f"Tutorial level: {tutorial.name}")
    print(f"Dimensions: {tutorial.width}x{tutorial.height}")
    print(f"Number of base solids: {len(tutorial.base_solids)}")
    
    # Create empty level
    empty = LevelLibrary.create_empty_level()
    print(f"\nEmpty level: {empty.name}")
    print(f"Number of base solids: {len(empty.base_solids)}")
    
    # Create custom level
    custom_config = {
        'width': 1200,
        'height': 600,
        'fire_start': (100, 100, 28, 36),
        'water_start': (200, 200, 28, 36)
    }
    custom = LevelLibrary.get_custom_level("Custom Level", custom_config)
    print(f"\nCustom level: {custom.name}")
    print(f"Dimensions: {custom.width}x{custom.height}")
