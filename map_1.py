@staticmethod
def get_tutorial_level() -> LevelConfig:
        """Get the tutorial level configuration"""
        return LevelConfig("Tutorial")

@staticmethod
def get_map_1_level() -> LevelConfig:
        """
        Get Map 1 - A more complex level with multi-tiered platforms and challenges
        Features:
        - Multi-level vertical platforms
        - Narrow passages requiring timing
        - Multiple hazard zones (water and lava)
        - Sequential puzzle elements requiring coordination
        - Staircase-like geometry
        """
        level = LevelConfig("Map 1")

        # Same dimensions
        level.width = 960
        level.height = 540

        # Agent spawn positions - start on opposite sides at different heights
        level.fire_start = (50, 440, 28, 36)      # Fire starts bottom-left
        level.water_start = (880, 280, 28, 36)    # Water starts mid-right

        # More complex platform geometry with multiple tiers
        level.base_solids = [
            # Boundary walls (10px thick)
            Rect(-10, 0, 10, level.height),         # Left wall
            Rect(level.width, 0, 10, level.height), # Right wall
            Rect(0, -10, level.width, 10),          # Top wall
            Rect(0, level.height, level.width, 10), # Bottom wall

            # Ground level (bottom tier)
            Rect(0, 500, 300, 40),                  # Left ground
            Rect(500, 500, 460, 40),                # Right ground (gap in middle)

            # Lower tier platforms (staircase effect)
            Rect(0, 440, 150, 20),                  # Bottom-left step
            Rect(120, 380, 120, 20),                # Second step up
            Rect(220, 320, 120, 20),                # Third step up

            # Middle tier - central area
            Rect(320, 260, 160, 20),                # Central platform (high)
            Rect(460, 320, 100, 20),                # Mid-right platform

            # Upper tier platforms
            Rect(540, 200, 140, 20),                # Upper-right platform
            Rect(720, 260, 120, 20),                # Right-mid platform
            Rect(860, 300, 100, 20),                # Far right platform

            # Vertical barriers creating narrow passages
            Rect(340, 180, 20, 80),                 # Central barrier above platform
            Rect(580, 120, 20, 80),                 # Upper barrier (creates challenge)

            # Ceiling platforms (create enclosed areas)
            Rect(100, 200, 120, 15),                # Left ceiling piece
            Rect(600, 140, 100, 15),                # Upper ceiling piece

            # Small obstacles
            Rect(450, 500, 30, 40),                 # Small wall on ground (gap creator)
            Rect(390, 240, 30, 20),                 # Small block near central platform
        ]

        # Interactive elements - more complex puzzle
        level.bridge = Rect(300, 500, 200, 20)      # Bridge over gap (activated by water on plate_a)
        level.gate = Rect(560, 200, 20, 100)        # Vertical gate blocking path (opened by fire on plate_b)

        # Pressure plates - require more coordination to reach
        level.plate_a = Rect(240, 304, 40, 16)      # Water plate (on third step) -> activates bridge
        level.plate_b = Rect(660, 244, 40, 16)      # Fire plate (on right platform) -> opens gate

        # Hazards - both types present
        level.water_pool = Rect(300, 480, 200, 20)  # Water pool under bridge area (kills fire)
        level.lava_pool = Rect(560, 320, 100, 80)   # Lava pool on right side (kills water)

        # Exit zones - require navigating the complex geometry
        level.exit_water = Rect(620, 124, 36, 36)   # Water exit (upper area, past gate)
        level.exit_fire = Rect(800, 244, 36, 36)    # Fire exit (right platform area)

        return level

@staticmethod
def get_custom_level(name: str, config: dict)
