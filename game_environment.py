"""
Game Environment - Gym-like interface for RL training
Completely decoupled from visualization/rendering
"""

import numpy as np
from typing import Tuple, Dict, Optional
from physics_engine import PhysicsEngine, Agent, Rect
from map_config import LevelConfig


class FireWaterEnv:
    """
    Gym-like environment for cooperative RL training
    
    Action Space: 
        0 = idle
        1 = left
        2 = right
        3 = jump
        4 = left + jump
        5 = right + jump
    
    Observation Space:
        52-dimensional state vector for each agent
    """
    
    def __init__(self, level: Optional[LevelConfig] = None, max_steps: int = 3000):
        """
        Initialize environment
        
        Args:
            level: Level configuration (uses tutorial level if None)
            max_steps: Maximum steps per episode
        """
        self.physics = PhysicsEngine()
        self.level = level if level is not None else LevelConfig()
        self.max_steps = max_steps
        self.reset()
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reset environment to initial state
        
        Returns:
            Tuple of (fire_observation, water_observation)
        """
        # Initialize agents at spawn positions
        self.fire_agent = Agent(
            rect=Rect(*self.level.fire_start),
            velocity=[0.0, 0.0],
            grounded=False,
            agent_type='fire'
        )
        self.water_agent = Agent(
            rect=Rect(*self.level.water_start),
            velocity=[0.0, 0.0],
            grounded=False,
            agent_type='water'
        )
        
        # Initialize game state
        self.bridge_up = False
        self.gate_open = False
        self.step_count = 0
        self.fire_died = False
        self.water_died = False
        
        # Track cooperation events
        self.bridge_activated = False
        self.gate_activated = False
        
        return self._get_observations()
    
    def step(self, fire_action: int, water_action: int) -> Tuple[
        Tuple[np.ndarray, np.ndarray],  # observations
        Tuple[float, float],              # rewards
        Tuple[bool, bool],                # dones
        Dict                              # info
    ]:
        """
        Execute one environment step
        
        Args:
            fire_action: Action for fire agent (0-5)
            water_action: Action for water agent (0-5)
            
        Returns:
            Tuple of (observations, rewards, dones, info)
        """
        self.step_count += 1
        
        # Get current state of environment
        solids = self.level.get_solids(self.bridge_up, self.gate_open)
        hazards = self.level.get_hazards()
        
        # Apply physics to both agents
        new_fire_agent = self.physics.apply_movement(
            self.fire_agent, fire_action, solids, hazards
        )
        new_water_agent = self.physics.apply_movement(
            self.water_agent, water_action, solids, hazards
        )
        
        # Check for agent deaths
        if new_fire_agent is None:
            self.fire_died = True
            # Keep old position for observation
        else:
            self.fire_agent = new_fire_agent
        
        if new_water_agent is None:
            self.water_died = True
        else:
            self.water_agent = new_water_agent
        
        # Update switches (permanent activation)
        switches = self.level.get_switches()
        
        # Water activates bridge
        if not self.bridge_activated and switches['plate_a']:
            if self.water_agent.rect.colliderect(switches['plate_a']):
                self.bridge_up = True
                self.bridge_activated = True
        
        # Fire opens gate
        if not self.gate_activated and switches['plate_b']:
            if self.fire_agent.rect.colliderect(switches['plate_b']):
                self.gate_open = True
                self.gate_activated = True
        
        # Check win conditions
        exits = self.level.get_exits()
        fire_won = exits['fire'].collidepoint(self.fire_agent.rect.center)
        water_won = exits['water'].collidepoint(self.water_agent.rect.center)
        
        # Calculate rewards
        fire_reward, water_reward = self._calculate_rewards(
            fire_won, water_won, fire_action, water_action
        )
        
        # Get observations
        observations = self._get_observations()
        
        # Check done conditions
        both_won = fire_won and water_won
        timeout = self.step_count >= self.max_steps
        fire_done = self.fire_died or both_won or timeout
        water_done = self.water_died or both_won or timeout
        
        # Compile info dictionary
        info = self._get_info(fire_won, water_won, both_won)
        
        return observations, (fire_reward, water_reward), (fire_done, water_done), info
    
    def _get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get state observations for both agents"""
        solids = self.level.get_solids(self.bridge_up, self.gate_open)
        switches = {
            'bridge_up': self.bridge_up, 
            'gate_open': self.gate_open
        }
        exits = self.level.get_exits()
        
        fire_obs = self.physics.get_state_vector(
            self.fire_agent, self.water_agent, switches, solids, exits
        )
        water_obs = self.physics.get_state_vector(
            self.water_agent, self.fire_agent, switches, solids, exits
        )
        
        return fire_obs, water_obs
    
    def _calculate_rewards(self, fire_won: bool, water_won: bool, 
                          fire_action: int, water_action: int) -> Tuple[float, float]:
        """
        Calculate rewards for both agents
        
        This is a basic reward structure - customize for your needs!
        """
        fire_reward = 0.0
        water_reward = 0.0
        
        # Death penalty
        if self.fire_died:
            fire_reward = -100.0
        if self.water_died:
            water_reward = -100.0
        
        # Win reward
        if fire_won and water_won:
            fire_reward += 100.0
            water_reward += 100.0
        
        # Small step penalty to encourage efficiency
        fire_reward -= 0.01
        water_reward -= 0.01
        
        # Cooperation rewards
        if self.bridge_activated and not hasattr(self, '_bridge_reward_given'):
            water_reward += 10.0  # Water activated bridge
            fire_reward += 5.0    # Fire benefits from bridge
            self._bridge_reward_given = True
        
        if self.gate_activated and not hasattr(self, '_gate_reward_given'):
            fire_reward += 10.0   # Fire opened gate
            water_reward += 5.0   # Water benefits from gate
            self._gate_reward_given = True
        
        # Distance-based shaping (optional - can help with exploration)
        exits = self.level.get_exits()
        
        # Fire distance to exit
        if not self.fire_died:
            fire_exit = exits['fire']
            fire_dist = np.sqrt(
                (self.fire_agent.rect.centerx - fire_exit.centerx)**2 +
                (self.fire_agent.rect.centery - fire_exit.centery)**2
            )
            fire_reward += 0.001 * (1000 - fire_dist) / 1000  # Closer = better
        
        # Water distance to exit
        if not self.water_died:
            water_exit = exits['water']
            water_dist = np.sqrt(
                (self.water_agent.rect.centerx - water_exit.centerx)**2 +
                (self.water_agent.rect.centery - water_exit.centery)**2
            )
            water_reward += 0.001 * (1000 - water_dist) / 1000
        
        return fire_reward, water_reward
    
    def _get_info(self, fire_won: bool, water_won: bool, both_won: bool) -> Dict:
        """Get additional info dictionary"""
        return {
            'step_count': self.step_count,
            'bridge_up': self.bridge_up,
            'gate_open': self.gate_open,
            'fire_won': fire_won,
            'water_won': water_won,
            'both_won': both_won,
            'fire_died': self.fire_died,
            'water_died': self.water_died,
            'cooperation_events': int(self.bridge_activated) + int(self.gate_activated)
        }
    
    def get_state_for_rendering(self) -> Dict:
        """
        Export complete state for visualization
        
        This allows the visualization script to render the game
        without needing to understand the internal state structure
        """
        return {
            'fire': {
                'rect': (self.fire_agent.rect.x, self.fire_agent.rect.y, 
                        self.fire_agent.rect.width, self.fire_agent.rect.height),
                'velocity': self.fire_agent.velocity,
                'grounded': self.fire_agent.grounded,
                'died': self.fire_died
            },
            'water': {
                'rect': (self.water_agent.rect.x, self.water_agent.rect.y,
                        self.water_agent.rect.width, self.water_agent.rect.height),
                'velocity': self.water_agent.velocity,
                'grounded': self.water_agent.grounded,
                'died': self.water_died
            },
            'bridge_up': self.bridge_up,
            'gate_open': self.gate_open,
            'step_count': self.step_count,
            'level': self.level
        }
    
    def get_action_space_size(self) -> int:
        """Get number of possible actions"""
        return 6
    
    def get_observation_space_size(self) -> int:
        """Get size of observation vector"""
        return 52


# Example usage
if __name__ == "__main__":
    print("Testing FireWaterEnv...")
    
    env = FireWaterEnv()
    print(f"Action space size: {env.get_action_space_size()}")
    print(f"Observation space size: {env.get_observation_space_size()}")
    
    # Run a random episode
    fire_obs, water_obs = env.reset()
    print(f"\nInitial fire observation shape: {fire_obs.shape}")
    print(f"Initial water observation shape: {water_obs.shape}")
    
    total_fire_reward = 0
    total_water_reward = 0
    
    for step in range(100):
        # Random actions
        fire_action = np.random.randint(0, 6)
        water_action = np.random.randint(0, 6)
        
        (fire_obs, water_obs), (fire_reward, water_reward), (fire_done, water_done), info = env.step(
            fire_action, water_action
        )
        
        total_fire_reward += fire_reward
        total_water_reward += water_reward
        
        if fire_done or water_done:
            print(f"\nEpisode ended at step {step}")
            print(f"Fire total reward: {total_fire_reward:.2f}")
            print(f"Water total reward: {total_water_reward:.2f}")
            print(f"Info: {info}")
            break
    
    print("\nEnvironment test completed successfully!")
