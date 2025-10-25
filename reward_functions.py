"""
Modular Reward System
Easy-to-customize reward functions for different training objectives
"""

import numpy as np
from typing import Dict, Tuple
from physics_engine import Agent, Rect


class RewardConfig:
    """Configuration for reward shaping - CUSTOMIZE THESE VALUES!"""
    
    def __init__(self):
        # ========== WIN/LOSE REWARDS ==========
        self.both_win_reward = 100.0        # Both agents reach exits
        self.death_penalty = -100.0         # Agent dies in hazard
        
        # ========== COOPERATION REWARDS ==========
        self.bridge_activation = 10.0       # Water activates bridge
        self.gate_activation = 10.0         # Fire opens gate
        self.cooperation_bonus = 5.0        # Partner benefits from switch
        
        # ========== PROGRESS REWARDS ==========
        self.step_penalty = -0.01           # Small penalty per step (encourages speed)
        self.distance_weight = 0.001        # Reward for getting closer to goal
        self.velocity_penalty = 0.0         # Penalty for excessive movement
        
        # ========== EXPLORATION REWARDS ==========
        self.exploration_bonus = 0.0        # Reward for visiting new areas
        self.stuck_penalty = -0.1           # Penalty for staying in same place
        
        # ========== SAFETY REWARDS ==========
        self.near_hazard_penalty = -0.5     # Penalty for being near hazards
        self.hazard_distance_threshold = 50 # Distance to consider "near"
        
        # ========== COORDINATION REWARDS ==========
        self.proximity_bonus = 0.0          # Reward for agents being close
        self.synchronization_bonus = 0.0    # Reward for coordinated movement


class BaseRewardFunction:
    """Base class for reward functions"""
    
    def __init__(self, config: RewardConfig = None):
        self.config = config or RewardConfig()
        
        # Track state for reward shaping
        self.prev_fire_distance = None
        self.prev_water_distance = None
        self.bridge_reward_given = False
        self.gate_reward_given = False
    
    def reset(self):
        """Reset tracking variables for new episode"""
        self.prev_fire_distance = None
        self.prev_water_distance = None
        self.bridge_reward_given = False
        self.gate_reward_given = False
    
    def calculate_rewards(
        self,
        fire_agent: Agent,
        water_agent: Agent,
        fire_action: int,
        water_action: int,
        fire_died: bool,
        water_died: bool,
        fire_won: bool,
        water_won: bool,
        bridge_activated: bool,
        gate_activated: bool,
        exits: Dict[str, Rect],
        hazards: Dict[str, Rect]
    ) -> Tuple[float, float]:
        """
        Calculate rewards for both agents
        
        Override this method to implement custom reward logic
        
        Returns:
            (fire_reward, water_reward)
        """
        raise NotImplementedError("Implement in subclass")


class SparseRewardFunction(BaseRewardFunction):
    """Sparse rewards - only win/lose, no shaping"""
    
    def calculate_rewards(self, fire_agent, water_agent, fire_action, water_action,
                         fire_died, water_died, fire_won, water_won,
                         bridge_activated, gate_activated, exits, hazards):
        fire_reward = 0.0
        water_reward = 0.0
        
        # Death penalty
        if fire_died:
            fire_reward = self.config.death_penalty
        if water_died:
            water_reward = self.config.death_penalty
        
        # Win reward
        if fire_won and water_won:
            fire_reward += self.config.both_win_reward
            water_reward += self.config.both_win_reward
        
        return fire_reward, water_reward


class DenseRewardFunction(BaseRewardFunction):
    """Dense rewards with distance-based shaping"""
    
    def calculate_rewards(self, fire_agent, water_agent, fire_action, water_action,
                         fire_died, water_died, fire_won, water_won,
                         bridge_activated, gate_activated, exits, hazards):
        fire_reward = 0.0
        water_reward = 0.0
        
        # ===== WIN/LOSE =====
        if fire_died:
            fire_reward += self.config.death_penalty
        if water_died:
            water_reward += self.config.death_penalty
        
        if fire_won and water_won:
            fire_reward += self.config.both_win_reward
            water_reward += self.config.both_win_reward
            return fire_reward, water_reward  # Episode done
        
        # ===== STEP PENALTY =====
        fire_reward += self.config.step_penalty
        water_reward += self.config.step_penalty
        
        # ===== COOPERATION REWARDS =====
        if bridge_activated and not self.bridge_reward_given:
            water_reward += self.config.bridge_activation
            fire_reward += self.config.cooperation_bonus
            self.bridge_reward_given = True
        
        if gate_activated and not self.gate_reward_given:
            fire_reward += self.config.gate_activation
            water_reward += self.config.cooperation_bonus
            self.gate_reward_given = True
        
        # ===== DISTANCE-BASED SHAPING =====
        if not fire_died:
            fire_exit = exits['fire']
            fire_distance = np.sqrt(
                (fire_agent.rect.centerx - fire_exit.centerx)**2 +
                (fire_agent.rect.centery - fire_exit.centery)**2
            )
            
            # Reward for getting closer
            if self.prev_fire_distance is not None:
                distance_delta = self.prev_fire_distance - fire_distance
                fire_reward += distance_delta * self.config.distance_weight
            
            self.prev_fire_distance = fire_distance
        
        if not water_died:
            water_exit = exits['water']
            water_distance = np.sqrt(
                (water_agent.rect.centerx - water_exit.centerx)**2 +
                (water_agent.rect.centery - water_exit.centery)**2
            )
            
            # Reward for getting closer
            if self.prev_water_distance is not None:
                distance_delta = self.prev_water_distance - water_distance
                water_reward += distance_delta * self.config.distance_weight
            
            self.prev_water_distance = water_distance
        
        return fire_reward, water_reward


class SafetyRewardFunction(DenseRewardFunction):
    """Dense rewards with additional safety penalties"""
    
    def calculate_rewards(self, fire_agent, water_agent, fire_action, water_action,
                         fire_died, water_died, fire_won, water_won,
                         bridge_activated, gate_activated, exits, hazards):
        # Get base dense rewards
        fire_reward, water_reward = super().calculate_rewards(
            fire_agent, water_agent, fire_action, water_action,
            fire_died, water_died, fire_won, water_won,
            bridge_activated, gate_activated, exits, hazards
        )
        
        # ===== SAFETY PENALTIES =====
        # Penalty for being near hazards
        if 'water_pool' in hazards and hazards['water_pool']:
            water_pool = hazards['water_pool']
            fire_dist_to_water = self._distance_to_rect(fire_agent.rect, water_pool)
            
            if fire_dist_to_water < self.config.hazard_distance_threshold:
                penalty = (1.0 - fire_dist_to_water / self.config.hazard_distance_threshold)
                fire_reward += self.config.near_hazard_penalty * penalty
        
        return fire_reward, water_reward
    
    def _distance_to_rect(self, agent_rect: Rect, hazard_rect: Rect) -> float:
        """Calculate minimum distance from agent to hazard"""
        # Closest point on hazard to agent
        closest_x = np.clip(agent_rect.centerx, hazard_rect.left, hazard_rect.right)
        closest_y = np.clip(agent_rect.centery, hazard_rect.top, hazard_rect.bottom)
        
        dx = agent_rect.centerx - closest_x
        dy = agent_rect.centery - closest_y
        
        return np.sqrt(dx**2 + dy**2)


class CooperationRewardFunction(DenseRewardFunction):
    """Emphasizes cooperation between agents"""
    
    def calculate_rewards(self, fire_agent, water_agent, fire_action, water_action,
                         fire_died, water_died, fire_won, water_won,
                         bridge_activated, gate_activated, exits, hazards):
        # Get base dense rewards
        fire_reward, water_reward = super().calculate_rewards(
            fire_agent, water_agent, fire_action, water_action,
            fire_died, water_died, fire_won, water_won,
            bridge_activated, gate_activated, exits, hazards
        )
        
        # ===== PROXIMITY BONUS =====
        if self.config.proximity_bonus > 0:
            distance_between = np.sqrt(
                (fire_agent.rect.centerx - water_agent.rect.centerx)**2 +
                (fire_agent.rect.centery - water_agent.rect.centery)**2
            )
            
            # Closer = better (up to a point)
            optimal_distance = 200
            if distance_between < optimal_distance:
                proximity_reward = self.config.proximity_bonus * (
                    1.0 - distance_between / optimal_distance
                )
                fire_reward += proximity_reward
                water_reward += proximity_reward
        
        # ===== SYNCHRONIZATION BONUS =====
        if self.config.synchronization_bonus > 0:
            # Reward for moving in same direction
            fire_moving_right = fire_action in [2, 5]
            fire_moving_left = fire_action in [1, 4]
            water_moving_right = water_action in [2, 5]
            water_moving_left = water_action in [1, 4]
            
            if (fire_moving_right and water_moving_right) or \
               (fire_moving_left and water_moving_left):
                fire_reward += self.config.synchronization_bonus
                water_reward += self.config.synchronization_bonus
        
        return fire_reward, water_reward


class CurriculumRewardFunction(BaseRewardFunction):
    """Adaptive rewards that change based on training progress"""
    
    def __init__(self, config: RewardConfig = None):
        super().__init__(config)
        self.episode_count = 0
        self.success_rate = 0.0
        
        # Start with sparse rewards, gradually add shaping
        self.sparse_fn = SparseRewardFunction(config)
        self.dense_fn = DenseRewardFunction(config)
    
    def set_progress(self, episode: int, success_rate: float):
        """Update curriculum based on training progress"""
        self.episode_count = episode
        self.success_rate = success_rate
    
    def calculate_rewards(self, fire_agent, water_agent, fire_action, water_action,
                         fire_died, water_died, fire_won, water_won,
                         bridge_activated, gate_activated, exits, hazards):
        # Early training: sparse rewards
        if self.episode_count < 1000 or self.success_rate < 0.1:
            return self.sparse_fn.calculate_rewards(
                fire_agent, water_agent, fire_action, water_action,
                fire_died, water_died, fire_won, water_won,
                bridge_activated, gate_activated, exits, hazards
            )
        
        # Later training: dense rewards
        else:
            return self.dense_fn.calculate_rewards(
                fire_agent, water_agent, fire_action, water_action,
                fire_died, water_died, fire_won, water_won,
                bridge_activated, gate_activated, exits, hazards
            )


# ========== REWARD FUNCTION REGISTRY ==========
REWARD_FUNCTIONS = {
    'sparse': SparseRewardFunction,
    'dense': DenseRewardFunction,
    'safety': SafetyRewardFunction,
    'cooperation': CooperationRewardFunction,
    'curriculum': CurriculumRewardFunction
}


def get_reward_function(name: str, config: RewardConfig = None) -> BaseRewardFunction:
    """
    Get reward function by name
    
    Args:
        name: One of 'sparse', 'dense', 'safety', 'cooperation', 'curriculum'
        config: Custom reward configuration
    
    Returns:
        Reward function instance
    """
    if name not in REWARD_FUNCTIONS:
        raise ValueError(f"Unknown reward function: {name}. Choose from {list(REWARD_FUNCTIONS.keys())}")
    
    return REWARD_FUNCTIONS[name](config)


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    print("Testing reward functions...\n")
    
    # Create dummy agents
    from physics_engine import Agent, Rect
    
    fire_agent = Agent(Rect(100, 100, 28, 36), [0, 0], False, 'fire')
    water_agent = Agent(Rect(200, 200, 28, 36), [0, 0], False, 'water')
    
    exits = {
        'fire': Rect(880, 388, 36, 36),
        'water': Rect(60, 308, 36, 36)
    }
    
    hazards = {
        'water_pool': Rect(520, 420, 180, 80)
    }
    
    # Test different reward functions
    for name, cls in REWARD_FUNCTIONS.items():
        print(f"Testing {name} reward function:")
        reward_fn = cls()
        
        fire_reward, water_reward = reward_fn.calculate_rewards(
            fire_agent, water_agent, 0, 0,
            False, False, False, False,
            False, False, exits, hazards
        )
        
        print(f"  Fire reward: {fire_reward:.3f}")
        print(f"  Water reward: {water_reward:.3f}\n")
    
    # Test custom config
    print("Testing custom reward config:")
    custom_config = RewardConfig()
    custom_config.both_win_reward = 500.0  # Much higher win reward
    custom_config.step_penalty = -0.1      # Higher step penalty
    custom_config.distance_weight = 0.01   # Stronger distance shaping
    
    reward_fn = DenseRewardFunction(custom_config)
    fire_reward, water_reward = reward_fn.calculate_rewards(
        fire_agent, water_agent, 0, 0,
        False, False, True, True,  # Both won!
        False, False, exits, hazards
    )
    
    print(f"  Fire reward (custom): {fire_reward:.3f}")
    print(f"  Water reward (custom): {water_reward:.3f}")
    
    print("\n✓ Reward function tests complete!")
