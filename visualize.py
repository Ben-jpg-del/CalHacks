"""
Visualization Script - Pygame-based rendering
Use this to visualize trained agents or play manually
"""

import pygame as pg
import numpy as np
from game_environment import FireWaterEnv
from map_config import LevelConfig, LevelLibrary
import sys
import time

# Import custom maps
try:
    from map_1 import LevelLibrary as Map1Library
    CUSTOM_MAPS_AVAILABLE = True
except ImportError:
    CUSTOM_MAPS_AVAILABLE = False
    print("Warning: map_1.py not found. Custom maps unavailable.")


# Colors
RED = (220, 60, 60)
BLUE = (60, 120, 255)
GREEN = (60, 200, 120)
GRAY = (170, 170, 170)
WHITE = (240, 240, 240)
BLACK = (15, 15, 20)
YELLOW = (255, 220, 60)
DARK_BLUE = (30, 60, 120)
DARK_RED = (120, 30, 30)


def get_level_from_name(level_name: str):
    """
    Get a level configuration from its name

    Args:
        level_name: Name of the level ("tutorial", "tower", "empty")

    Returns:
        LevelConfig object
    """
    level_name = level_name.lower()

    if level_name == "tutorial":
        return LevelLibrary.get_tutorial_level()
    elif level_name == "tower" and CUSTOM_MAPS_AVAILABLE:
        return Map1Library.get_tower_level()
    elif level_name == "empty":
        return LevelLibrary.create_empty_level()
    else:
        print(f"Unknown level: {level_name}")
        print("Available levels: tutorial, tower (if map_1.py exists), empty")
        print("Using tutorial level as default")
        return LevelLibrary.get_tutorial_level()


def list_available_maps():
    """Display all available maps"""
    print("\n" + "=" * 60)
    print("AVAILABLE MAPS")
    print("=" * 60)
    print("  tutorial  - Default tutorial level (horizontal layout)")
    if CUSTOM_MAPS_AVAILABLE:
        print("  tower     - The Tower Ascent (vertical climbing puzzle)")
    print("  empty     - Empty level with just boundary walls")
    print("=" * 60 + "\n")


class Visualizer:
    """Handles all Pygame rendering"""

    def __init__(self, width=960, height=540, scale=1.0):
        pg.init()
        self.width = width
        self.height = height
        self.scale = scale

        self.screen_width = int(width * scale)
        self.screen_height = int(height * scale)

        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        pg.display.set_caption("Fire & Water RL Visualization")

        self.clock = pg.time.Clock()
        self.font_large = pg.font.SysFont(None, 36)
        self.font_medium = pg.font.SysFont(None, 24)
        self.font_small = pg.font.SysFont(None, 18)

    def render(self, env: FireWaterEnv, metrics: dict = None):
        """Render the current game state"""
        self.screen.fill(BLACK)

        # Get state for rendering
        state = env.get_state_for_rendering()
        level = state['level']

        # Draw static geometry
        for solid in level.base_solids:
            self._draw_rect(solid, GRAY)

        # Draw bridge (if active)
        if state['bridge_up']:
            self._draw_rect(level.bridge, GREEN)

        # Draw gate (if closed)
        if not state['gate_open']:
            self._draw_rect(level.gate, GREEN)

        # Draw pressure plates
        switches = level.get_switches()
        plate_a_color = (100, 150, 255) if state['bridge_up'] else (50, 75, 150)
        plate_b_color = (255, 100, 100) if state['gate_open'] else (150, 50, 50)
        self._draw_rect(switches['plate_a'], plate_a_color)
        self._draw_rect(switches['plate_b'], plate_b_color)

        # Draw hazards
        hazards = level.get_hazards()
        if 'water_pool' in hazards:
            self._draw_rect(hazards['water_pool'], BLUE, alpha=128)
        if 'lava_pool' in hazards:
            self._draw_rect(hazards['lava_pool'], RED, alpha=128)

        # Draw exits
        exits = level.get_exits()
        self._draw_rect(exits['fire'], RED, border=2)
        self._draw_rect(exits['water'], BLUE, border=2)

        # Draw agents
        fire_rect = state['fire']['rect']
        water_rect = state['water']['rect']

        if not state['fire']['died']:
            pg.draw.rect(self.screen, RED, self._scale_rect(fire_rect))
        if not state['water']['died']:
            pg.draw.rect(self.screen, BLUE, self._scale_rect(water_rect))

        # Draw info overlay
        if metrics:
            self._draw_info(state, metrics)

        pg.display.flip()

    def _draw_rect(self, rect, color, alpha=255, border=0):
        """Draw a rectangle with optional alpha and border"""
        scaled = self._scale_rect((rect.x, rect.y, rect.width, rect.height))

        if alpha < 255:
            # Create surface with alpha
            surf = pg.Surface((scaled[2], scaled[3]))
            surf.set_alpha(alpha)
            surf.fill(color)
            self.screen.blit(surf, (scaled[0], scaled[1]))
        else:
            if border > 0:
                pg.draw.rect(self.screen, color, scaled, border)
            else:
                pg.draw.rect(self.screen, color, scaled)

    def _scale_rect(self, rect):
        """Scale rectangle coordinates"""
        if isinstance(rect, tuple):
            x, y, w, h = rect
        else:
            x, y, w, h = rect.x, rect.y, rect.width, rect.height
        return (int(x * self.scale), int(y * self.scale),
                int(w * self.scale), int(h * self.scale))

    def _draw_info(self, state, metrics):
        """Draw information overlay"""
        y = 10
        x = 10

        # Episode info
        if 'episode' in metrics:
            text = self.font_medium.render(
                f"Episode: {metrics['episode']}", True, YELLOW
            )
            self.screen.blit(text, (x, y))
            y += 30

        # Step count
        text = self.font_small.render(
            f"Steps: {state['step_count']}", True, WHITE
        )
        self.screen.blit(text, (x, y))
        y += 25

        # Success rate
        if 'success_rate' in metrics:
            text = self.font_small.render(
                f"Success Rate: {metrics['success_rate']:.1%}", True, GREEN
            )
            self.screen.blit(text, (x, y))
            y += 25

        # Cooperation events
        text = self.font_small.render(
            f"Bridge: {'UP' if state['bridge_up'] else 'DOWN'}", True, BLUE
        )
        self.screen.blit(text, (x, y))
        y += 20

        text = self.font_small.render(
            f"Gate: {'OPEN' if state['gate_open'] else 'CLOSED'}", True, RED
        )
        self.screen.blit(text, (x, y))
        y += 30

        # Controls
        text = self.font_small.render("Controls:", True, YELLOW)
        self.screen.blit(text, (x, y))
        y += 20

        text = self.font_small.render("R - Reset", True, WHITE)
        self.screen.blit(text, (x, y))
        y += 18

        text = self.font_small.render("ESC - Quit", True, WHITE)
        self.screen.blit(text, (x, y))

    def tick(self, fps=60):
        """Control frame rate"""
        self.clock.tick(fps)


def visualize_random_agents(num_episodes=10, fps=60, level_name="tutorial"):
    """Visualize random agents playing"""
    print("Visualizing random agents...")
    print("Press ESC to quit, R to reset\n")

    # Load the specified level
    level = get_level_from_name(level_name)
    print(f"Using level: {level.name}\n")

    env = FireWaterEnv(level=level)
    vis = Visualizer()

    for episode in range(num_episodes):
        fire_obs, water_obs = env.reset()
        done = False

        success_tracker = []

        while not done:
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    return
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        pg.quit()
                        return
                    if event.key == pg.K_r:
                        print("Manual reset")
                        done = True

            # Random actions
            fire_action = np.random.randint(0, 6)
            water_action = np.random.randint(0, 6)

            # Step
            (fire_obs, water_obs), rewards, dones, info = env.step(fire_action, water_action)

            # Render
            metrics = {
                'episode': episode + 1,
                'success_rate': np.mean(success_tracker) if success_tracker else 0
            }
            vis.render(env, metrics)
            vis.tick(fps)

            if dones[0] or dones[1]:
                done = True
                success_tracker.append(1 if info['both_won'] else 0)

                if info['both_won']:
                    print(f"Episode {episode + 1}: SUCCESS!")
                else:
                    print(f"Episode {episode + 1}: Failed")

                time.sleep(1)  # Pause to see result

    pg.quit()
    print("Visualization complete")


def human_play_mode(level_name="tutorial"):
    """
    Human-controlled gameplay
    Fire: A/D = move, W = jump
    Water: Arrow keys = move/jump
    """
    print("=" * 60)
    print("HUMAN PLAY MODE")
    print("=" * 60)
    print("Fire Agent (Red):")
    print("  A/D = Move left/right")
    print("  W = Jump")
    print("\nWater Agent (Blue):")
    print("  Arrow Keys = Move/Jump")
    print("\nPress R to reset, ESC to quit")
    print("=" * 60 + "\n")

    # Load the specified level
    level = get_level_from_name(level_name)
    print(f"Using level: {level.name}\n")

    env = FireWaterEnv(level=level)
    vis = Visualizer()

    running = True
    while running:
        fire_obs, water_obs = env.reset()
        episode_done = False

        while not episode_done and running:
            # Handle events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
                    episode_done = True
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE:
                        running = False
                        episode_done = True
                    if event.key == pg.K_r:
                        print("Reset level")
                        episode_done = True

            # Get keyboard input
            keys = pg.key.get_pressed()

            # Map keys to actions
            fire_action = 0  # idle
            water_action = 0  # idle

            # Fire controls (A/D/W)
            if keys[pg.K_w] and keys[pg.K_a]:
                fire_action = 4  # left + jump
            elif keys[pg.K_w] and keys[pg.K_d]:
                fire_action = 5  # right + jump
            elif keys[pg.K_w]:
                fire_action = 3  # jump
            elif keys[pg.K_a]:
                fire_action = 1  # left
            elif keys[pg.K_d]:
                fire_action = 2  # right

            # Water controls (arrows)
            if keys[pg.K_UP] and keys[pg.K_LEFT]:
                water_action = 4  # left + jump
            elif keys[pg.K_UP] and keys[pg.K_RIGHT]:
                water_action = 5  # right + jump
            elif keys[pg.K_UP]:
                water_action = 3  # jump
            elif keys[pg.K_LEFT]:
                water_action = 1  # left
            elif keys[pg.K_RIGHT]:
                water_action = 2  # right

            # Step environment
            (fire_obs, water_obs), rewards, dones, info = env.step(fire_action, water_action)

            # Render
            vis.render(env)
            vis.tick(60)

            if dones[0] or dones[1]:
                episode_done = True
                if info['both_won']:
                    print("\n*** YOU WIN! Both agents reached their exits! ***\n")
                elif info['fire_died']:
                    print("\nFire died in water!")
                elif info['water_died']:
                    print("\nWater died in lava!")
                time.sleep(1.5)

    pg.quit()
    print("Human play mode ended")


def visualize_trained_agent(model_path: str = None, num_episodes: int = 10, level_name="tutorial"):
    """
    Visualize a trained agent

    Args:
        model_path: Path to trained model weights
        num_episodes: Number of episodes to visualize
        level_name: Name of the level to use
    """
    print(f"Visualizing trained agent from: {model_path}")
    print("Note: Replace with your actual agent loading code\n")

    # TODO: Load your trained agents here
    # fire_agent = YourRLAgent()
    # water_agent = YourRLAgent()
    # fire_agent.load(model_path + "_fire.pth")
    # water_agent.load(model_path + "_water.pth")

    # For now, use random agents as placeholder
    print("Warning: Using random agents (implement agent loading)")
    visualize_random_agents(num_episodes, level_name=level_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fire & Water RL Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Human play mode with tutorial level
  python visualize.py human

  # Human play with tower level
  python visualize.py human --map tower

  # Random agents on tower level for 10 episodes
  python visualize.py random 10 --map tower

  # List all available maps
  python visualize.py --list-maps

  # Visualize trained agent on tower level
  python visualize.py trained checkpoints/model.pth --map tower
        """
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="human",
        choices=["human", "random", "trained"],
        help="Visualization mode (default: human)"
    )

    parser.add_argument(
        "episodes_or_model",
        nargs="?",
        default=None,
        help="Number of episodes (for random mode) or model path (for trained mode)"
    )

    parser.add_argument(
        "--map", "-m",
        type=str,
        default="tutorial",
        help="Map to use: tutorial, tower, empty (default: tutorial)"
    )

    parser.add_argument(
        "--list-maps",
        action="store_true",
        help="List all available maps and exit"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for visualization (default: 60)"
    )

    args = parser.parse_args()

    # Handle list maps command
    if args.list_maps:
        list_available_maps()
        sys.exit(0)

    # Execute the requested mode
    if args.mode == "human":
        human_play_mode(level_name=args.map)
    elif args.mode == "random":
        episodes = int(args.episodes_or_model) if args.episodes_or_model else 5
        visualize_random_agents(episodes, fps=args.fps, level_name=args.map)
    elif args.mode == "trained":
        model_path = args.episodes_or_model if args.episodes_or_model else None
        episodes = 10  # Default number of episodes for trained agent visualization
        visualize_trained_agent(model_path, episodes, level_name=args.map)
    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: human, random, trained")
        parser.print_help()
