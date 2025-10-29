"""
Custom Level: "The Crossing"
Diagonal cooperation puzzle where agents start on opposite sides and must cross paths.
- Fire (red) starts BOTTOM LEFT
- Water (blue) starts BOTTOM RIGHT
- Fire's plate is on MIDDLE RIGHT (water must go there)
- Water's plate is on MIDDLE LEFT (fire must go there)
- Fire's exit is TOP LEFT (blocked until water hits fire's plate)
- Water's exit is TOP RIGHT (blocked until fire hits water's plate)
- Hazards in center create risk during crossing
"""

from physics_engine import Rect
from typing import List, Dict


class LevelConfig:
    """Level configuration - diagonal crossing with cooperation"""

    def __init__(self, level_name: str = "The Crossing"):
        self.name = level_name
        self.width = 960
        self.height = 540

        # Agent spawn positions - OPPOSITE sides
        self.fire_start = (80, 454, 28, 36)        # Fire spawns BOTTOM LEFT
        self.water_start = (850, 454, 28, 36)      # Water spawns BOTTOM RIGHT

        # Define all static geometry - symmetric X-pattern
        self.base_solids = [
            # Boundary walls (10px thick)
            Rect(-10, 0, 10, self.height),              # Left wall
            Rect(self.width, 0, 10, self.height),       # Right wall
            Rect(0, -10, self.width, 10),               # Top wall
            Rect(0, self.height, self.width, 10),       # Bottom wall

            # Bottom level - spawn platforms (LEFT and RIGHT)
            Rect(0, 500, 250, 40),                      # Bottom LEFT floor (fire spawn)
            Rect(710, 500, 250, 40),                    # Bottom RIGHT floor (water spawn)

            # Middle level - LEFT side (water's plate here)
            Rect(40, 340, 60, 20),                      # Middle LEFT platform (shortened to 60px)

            # Middle level - RIGHT side (fire's plate here)
            Rect(620, 340, 400, 20),                    # Middle RIGHT platform (extended 100px to the left)

            # Center platforms - for crossing/traversing
            Rect(300, 460, 120, 20),                    # Center-left lower
            Rect(540, 460, 120, 20),                    # Center-right lower
            Rect(420, 480, 120, 20),                    # Center middle (moved down 100px, below hazards)

            # Top level - LEFT side (fire's exit here, blocked initially)
            Rect(0, 180, 240, 20),                      # Top LEFT platform (extended to left border)

            # Top level - RIGHT side (water's exit here, blocked initially)
            Rect(720, 180, 240, 20),                    # Top RIGHT platform (extended to right border)

            # Vertical connectors
            Rect(120, 340, 20, 120),                    # Left vertical connector
            # Right vertical connector removed to allow access to top right platform
        ]

        # Interactive elements - Bridges that APPEAR when plates are activated (like tutorial/map_1)
        # Gate bridge (RIGHT) - appears when fire steps on plate_b, closes off upper level
        self.gate = Rect(700, 0, 20, 180)              # Vertical wall from ceiling (y=0) down to top platform (y=180)

        # Bridge platform (LEFT) - appears when water steps on plate_a
        self.bridge = Rect(100, 340, 200, 20)          # Horizontal bridge 200px long, at middle level height

        # Pressure plates - OPPOSITE sides from spawn
        # NOTE: plate_a is activated by WATER, plate_b is activated by FIRE (game logic)
        self.plate_a = Rect(100, 324, 40, 16)          # WATER's plate on MIDDLE LEFT - water steps here -> creates bridge
        self.plate_b = Rect(820, 324, 40, 16)          # FIRE's plate on MIDDLE RIGHT - fire steps here -> creates gate bridge

        # Hazards in CENTER to make crossing dangerous
        self.water_pool = Rect(420, 380, 60, 20)       # Water pool CENTER (kills fire during crossing)
        self.lava_pool = Rect(480, 380, 60, 20)        # Lava pool CENTER (kills water during crossing)

        # Exit zones - DIAGONAL corners (opposite of spawn)
        self.exit_fire = Rect(100, 148, 36, 36)        # Fire exit TOP LEFT
        self.exit_water = Rect(824, 148, 36, 36)       # Water exit TOP RIGHT

    def get_solids(self, bridge_up: bool = False, gate_open: bool = False) -> List[Rect]:
        """
        Get list of active solid obstacles based on game state

        Args:
            bridge_up: Whether the bridge is active (water hit plate_a - ADDS green bridge as solid platform)
            gate_open: Whether the gate is open (fire hit plate_b - REMOVES green wall barrier)

        Returns:
            List of solid Rect objects that block movement
        """
        solids = list(self.base_solids)

        # LEFT: Bridge appears when activated (water steps on plate_a -> creates solid bridge)
        if bridge_up:
            solids.append(self.bridge)

        # RIGHT: Gate wall blocks by default, REMOVED when activated (fire steps on plate_b -> removes wall)
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
            'plate_a': self.plate_a,  # Water activates gate
            'plate_b': self.plate_b   # Fire activates bridge
        }


class LevelLibrary:
    """Collection of pre-defined levels"""

    @staticmethod
    def get_map2_level() -> LevelConfig:
        """Get The Crossing level configuration"""
        return LevelConfig("The Crossing")

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
    # Get The Crossing level
    level = LevelLibrary.get_map2_level()
    print(f"Level: {level.name}")
    print(f"Dimensions: {level.width}x{level.height}")
    print(f"Fire start: {level.fire_start}")
    print(f"Water start: {level.water_start}")
    print(f"Number of base solids: {len(level.base_solids)}")
    print(f"Number of pressure plates: {len(level.get_switches())}")
    print(f"Pressure plates: {list(level.get_switches().keys())}")
    print(f"Hazards: {list(level.get_hazards().keys())}")

    print("\nLevel Description:")
    print("- DIAGONAL CROSSING: Agents start on opposite sides")
    print("- Fire (RED) spawns BOTTOM LEFT")
    print("- Water (BLUE) spawns BOTTOM RIGHT")
    print("- Water's plate (BLUE) is on MIDDLE LEFT - water travels there")
    print("- Fire's plate (RED) is on MIDDLE RIGHT - fire travels there")
    print("- Fire's exit is TOP LEFT (150px wall blocks until water helps)")
    print("- Water's exit is TOP RIGHT (150px wall blocks until fire helps)")
    print("- Center hazards (lava + water) create danger during crossing")

    print("\nSolution Strategy:")
    print("  1. Fire starts bottom LEFT, Water starts bottom RIGHT")
    print("  2. WATER travels LEFT to plate_a (middle-left)")
    print("     -> Removes bridge wall -> Fire can access TOP LEFT exit")
    print("  3. FIRE travels RIGHT to plate_b (middle-right)")
    print("     -> Removes gate wall -> Water can access TOP RIGHT exit")
    print("  4. Fire climbs to TOP LEFT exit (wall now removed)")
    print("  5. Water climbs to TOP RIGHT exit (wall now removed)")
    print("  6. Must avoid center hazards during crossing!")

    print("\nKey Features:")
    print("  - X-pattern: diagonal movement from corners")
    print("  - Each agent MUST cross to opposite side to help")
    print("  - Symmetric design (balanced for both agents)")
    print("  - Center hazards punish unsafe crossing")
    print("  - True cooperation: neither can win alone")

    # Test different game states
    print("\nPlatform counts in different states:")
    print(f"  Initial (no switches): {len(level.get_solids(False, False))} platforms")
    print(f"  Bridge active: {len(level.get_solids(True, False))} platforms")
    print(f"  Gate open: {len(level.get_solids(False, True))} platforms")
    print(f"  Both active: {len(level.get_solids(True, True))} platforms")