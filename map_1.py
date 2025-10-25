"""
Custom Level: "The Tower Ascent"
A vertical cooperative puzzle where agents must climb a tower together,
navigating hazards and activating switches to help each other progress.
"""

from physics_engine import Rect
from typing import List, Dict


class LevelConfig:
    """Level configuration based on custom sketch - horizontal multi-level puzzle"""

    def __init__(self, level_name: str = "Custom Sketch Level"):
        self.name = level_name
        self.width = 960
        self.height = 540

        # Agent spawn positions (x, y, width, height) - bottom left area
        self.fire_start = (150, 454, 28, 36)       # Fire spawns bottom left (red box)
        self.water_start = (80, 454, 28, 36)       # Water spawns bottom left (blue box)

        # Define all static geometry based on the sketch
        self.base_solids = [
            # Boundary walls (10px thick)
            Rect(-10, 0, 10, self.height),              # Left wall
            Rect(self.width, 0, 10, self.height),       # Right wall
            Rect(0, -10, self.width, 10),               # Top wall
            Rect(0, self.height, self.width, 10),       # Bottom wall

            # Bottom level (floor with spawn area)
            Rect(0, 500, 380, 40),                      # Bottom left floor

            # Bottom middle level - long platform on right
            Rect(400, 420, 560, 20),                    # Long bottom-right platform

            # Middle level - left side platforms
            Rect(100, 340, 250, 20),                    # Middle-left long platform

            # Middle level - right side platform
            Rect(700, 340, 260, 20),                    # Middle-right platform

            # Top level - left side (lowered to be reachable)
            Rect(100, 220, 180, 20),                    # Top-left platform (with green/gate area)

            # Top level - right side (lowered to be reachable, extends far right)
            Rect(700, 220, 260, 20),                    # Top-right platform (with exits)

            # Barrier on right side
            Rect(955, 100, 5, 400),                     # Right barrier
        ]

        # Interactive elements
        # Bridge connects middle platforms (based on sketch's center connection)
        self.bridge = Rect(350, 340, 350, 20)          # Activated by water on plate_a

        # Gate blocks access to the winning platform (right side, blocks exits)
        self.gate = Rect(680, 220, 20, 120)            # Opens when fire hits plate_b

        # Pressure plates (small rectangles in sketch)
        self.plate_a = Rect(830, 324, 40, 16)          # Water activates (blue small rect) -> bridge appears
        self.plate_b = Rect(500, 404, 40, 16)          # Fire activates (red small rect) -> gate opens

        # Hazards based on large colored rectangles in sketch
        self.water_pool = Rect(500, 340, 180, 20)      # Water pool on middle-right (blue large rect - kills fire)
        self.lava_pool = Rect(400, 420, 100, 20)       # Lava pool on bottom-right (red large rect - kills water)

        # Exit zones - top right (adjusted to match lowered top level)
        self.exit_water = Rect(820, 188, 36, 36)       # Water exit top right (blue box)
        self.exit_fire = Rect(900, 188, 36, 36)        # Fire exit top right (red box)

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
    def get_tower_level() -> LevelConfig:
        """Get Custom Sketch level configuration"""
        return LevelConfig("Custom Sketch Level")

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


# Example: How to use this map
if __name__ == "__main__":
    # Get Custom Sketch level
    level = LevelLibrary.get_tower_level()
    print(f"Level: {level.name}")
    print(f"Dimensions: {level.width}x{level.height}")
    print(f"Fire start: {level.fire_start}")
    print(f"Water start: {level.water_start}")
    print(f"Number of base solids: {len(level.base_solids)}")
    print(f"Number of pressure plates: {len(level.get_switches())}")
    print(f"Pressure plates: {list(level.get_switches().keys())}")
    print(f"Hazards: {list(level.get_hazards().keys())}")

    print("\nLevel Description:")
    print("- Agents spawn together on bottom left")
    print("- Must navigate across 3 horizontal levels")
    print("- Water agent activates bridge to connect middle platforms")
    print("- Fire agent activates gate to open passage from top level")
    print("- Water pool on middle platform (kills fire)")
    print("- Lava pool on bottom-right platform (kills water)")
    print("- Both exits are on the top-right platform")

    # Test different game states
    print("\nPlatform counts in different states:")
    print(f"  Initial (no switches): {len(level.get_solids(False, False))} platforms")
    print(f"  Bridge active: {len(level.get_solids(True, False))} platforms")
    print(f"  Gate open: {len(level.get_solids(False, True))} platforms")
    print(f"  Both active: {len(level.get_solids(True, True))} platforms")
