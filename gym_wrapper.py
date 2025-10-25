"""
Gymnasium-Compatible Wrapper
Provides standard Gym interface for compatibility with RL libraries like Stable-Baselines3
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from game_environment import FireWaterEnv
from typing import Tuple, Dict, Any, Optional


class FireWaterGymEnv(gym.Env):
    """
    Single-agent Gym wrapper for Fire & Water environment
    Controls one agent while the other uses a policy/heuristic
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        agent_type: str = "fire",
        partner_policy: Optional[callable] = None,
        render_mode: Optional[str] = None,
        max_steps: int = 3000
    ):
        """
        Initialize Gym environment
        
        Args:
            agent_type: Which agent to control ("fire" or "water")
            partner_policy: Policy for partner agent (None = random)
            render_mode: "human" or "rgb_array"
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.agent_type = agent_type
        self.partner_policy = partner_policy or self._random_policy
        self.render_mode = render_mode
        
        # Create base environment
        self.env = FireWaterEnv(max_steps=max_steps)
        
        # Define action space: 6 discrete actions
        self.action_space = spaces.Discrete(6)
        
        # Define observation space: 52-dimensional continuous
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(52,),
            dtype=np.float32
        )
        
        # Rendering
        self.visualizer = None
        if render_mode == "human":
            self._init_visualizer()
    
    def _init_visualizer(self):
        """Initialize Pygame visualizer"""
        try:
            from visualize import Visualizer
            self.visualizer = Visualizer()
        except ImportError:
            print("Warning: Pygame not available. Rendering disabled.")
            self.visualizer = None
    
    def _random_policy(self, observation):
        """Default random policy for partner"""
        return self.action_space.sample()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment
        
        Returns:
            observation: Initial observation
            info: Additional info dictionary
        """
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        fire_obs, water_obs = self.env.reset()
        
        # Return observation for controlled agent
        obs = fire_obs if self.agent_type == "fire" else water_obs
        
        return obs.astype(np.float32), {}
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step
        
        Args:
            action: Action for controlled agent
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            terminated: Whether episode ended naturally
            truncated: Whether episode was truncated
            info: Additional info
        """
        # Get partner action
        partner_obs = self.env.get_state_for_rendering()
        if self.agent_type == "fire":
            fire_obs_full = self.env.physics.get_state_vector(
                self.env.fire_agent, self.env.water_agent,
                {'bridge_up': self.env.bridge_up, 'gate_open': self.env.gate_open},
                self.env.level.get_solids(self.env.bridge_up, self.env.gate_open),
                self.env.level.get_exits()
            )
            water_obs_full = self.env.physics.get_state_vector(
                self.env.water_agent, self.env.fire_agent,
                {'bridge_up': self.env.bridge_up, 'gate_open': self.env.gate_open},
                self.env.level.get_solids(self.env.bridge_up, self.env.gate_open),
                self.env.level.get_exits()
            )
            partner_action = self.partner_policy(water_obs_full)
            fire_action, water_action = action, partner_action
        else:
            fire_obs_full = self.env.physics.get_state_vector(
                self.env.fire_agent, self.env.water_agent,
                {'bridge_up': self.env.bridge_up, 'gate_open': self.env.gate_open},
                self.env.level.get_solids(self.env.bridge_up, self.env.gate_open),
                self.env.level.get_exits()
            )
            water_obs_full = self.env.physics.get_state_vector(
                self.env.water_agent, self.env.fire_agent,
                {'bridge_up': self.env.bridge_up, 'gate_open': self.env.gate_open},
                self.env.level.get_solids(self.env.bridge_up, self.env.gate_open),
                self.env.level.get_exits()
            )
            partner_action = self.partner_policy(fire_obs_full)
            fire_action, water_action = partner_action, action
        
        # Step environment
        (fire_obs, water_obs), (fire_reward, water_reward), (fire_done, water_done), info = self.env.step(
            fire_action, water_action
        )
        
        # Get observation and reward for controlled agent
        if self.agent_type == "fire":
            obs = fire_obs
            reward = fire_reward
            done = fire_done
        else:
            obs = water_obs
            reward = water_reward
            done = water_done
        
        terminated = done
        truncated = False
        
        # Render if needed
        if self.render_mode == "human" and self.visualizer:
            self.visualizer.render(self.env)
            self.visualizer.tick(60)
        
        return obs.astype(np.float32), float(reward), terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and self.visualizer:
            self.visualizer.render(self.env)
        elif self.render_mode == "rgb_array":
            # TODO: Return RGB array
            pass
    
    def close(self):
        """Clean up resources"""
        if self.visualizer:
            import pygame
            pygame.quit()


class MultiAgentFireWaterGymEnv(gym.Env):
    """
    Multi-agent Gym wrapper
    Controls both agents simultaneously (more realistic for cooperative learning)
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        max_steps: int = 3000
    ):
        """
        Initialize multi-agent Gym environment
        
        Args:
            render_mode: "human" or "rgb_array"
            max_steps: Maximum steps per episode
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # Create base environment
        self.env = FireWaterEnv(max_steps=max_steps)
        
        # Define action space: Two discrete actions (one per agent)
        self.action_space = spaces.MultiDiscrete([6, 6])
        
        # Define observation space: Dictionary with fire and water observations
        self.observation_space = spaces.Dict({
            "fire": spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32),
            "water": spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)
        })
        
        # Rendering
        self.visualizer = None
        if render_mode == "human":
            self._init_visualizer()
    
    def _init_visualizer(self):
        """Initialize Pygame visualizer"""
        try:
            from visualize import Visualizer
            self.visualizer = Visualizer()
        except ImportError:
            print("Warning: Pygame not available. Rendering disabled.")
            self.visualizer = None
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment"""
        super().reset(seed=seed)
        
        if seed is not None:
            np.random.seed(seed)
        
        fire_obs, water_obs = self.env.reset()
        
        obs = {
            "fire": fire_obs.astype(np.float32),
            "water": water_obs.astype(np.float32)
        }
        
        return obs, {}
    
    def step(
        self,
        action: Tuple[int, int]
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step"""
        fire_action, water_action = action
        
        # Step environment
        (fire_obs, water_obs), (fire_reward, water_reward), (fire_done, water_done), info = self.env.step(
            fire_action, water_action
        )
        
        # Combine observations
        obs = {
            "fire": fire_obs.astype(np.float32),
            "water": water_obs.astype(np.float32)
        }
        
        # Combine rewards (you can customize this)
        reward = fire_reward + water_reward
        
        terminated = fire_done or water_done
        truncated = False
        
        # Add per-agent info
        info['fire_reward'] = fire_reward
        info['water_reward'] = water_reward
        info['fire_done'] = fire_done
        info['water_done'] = water_done
        
        # Render if needed
        if self.render_mode == "human" and self.visualizer:
            self.visualizer.render(self.env)
            self.visualizer.tick(60)
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        """Render the environment"""
        if self.render_mode == "human" and self.visualizer:
            self.visualizer.render(self.env)
        elif self.render_mode == "rgb_array":
            pass
    
    def close(self):
        """Clean up resources"""
        if self.visualizer:
            import pygame
            pygame.quit()


# Example usage
if __name__ == "__main__":
    print("Testing Gym wrapper...")
    
    # Test single-agent wrapper
    print("\n1. Single-agent wrapper (Fire agent):")
    env = FireWaterGymEnv(agent_type="fire")
    
    obs, info = env.reset()
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Initial observation shape: {obs.shape}")
    
    # Run a few steps
    total_reward = 0
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"   Total reward after 10 steps: {total_reward:.2f}")
    env.close()
    
    # Test multi-agent wrapper
    print("\n2. Multi-agent wrapper:")
    env = MultiAgentFireWaterGymEnv()
    
    obs, info = env.reset()
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    print(f"   Fire observation shape: {obs['fire'].shape}")
    print(f"   Water observation shape: {obs['water'].shape}")
    
    # Run a few steps
    total_reward = 0
    for _ in range(10):
        action = (env.action_space[0].sample(), env.action_space[1].sample())
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated:
            break
    
    print(f"   Total reward after 10 steps: {total_reward:.2f}")
    env.close()
    
    print("\n✓ Gym wrapper test complete!")
