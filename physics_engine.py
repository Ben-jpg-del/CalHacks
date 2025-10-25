"""
Physics Engine - Pure Python implementation without Pygame dependency
Handles all game physics, collisions, and state calculations
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


@dataclass
class Rect:
    """Lightweight rectangle class without Pygame dependency"""
    x: float
    y: float
    width: float
    height: float
    
    @property
    def left(self) -> float:
        return self.x
    
    @property
    def right(self) -> float:
        return self.x + self.width
    
    @property
    def top(self) -> float:
        return self.y
    
    @property
    def bottom(self) -> float:
        return self.y + self.height
    
    @property
    def centerx(self) -> float:
        return self.x + self.width / 2
    
    @property
    def centery(self) -> float:
        return self.y + self.height / 2
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.centerx, self.centery)
    
    def colliderect(self, other: 'Rect') -> bool:
        """Check if this rectangle collides with another"""
        return not (self.right <= other.left or 
                   self.left >= other.right or 
                   self.bottom <= other.top or 
                   self.top >= other.bottom)
    
    def collidepoint(self, point: Tuple[float, float]) -> bool:
        """Check if a point is inside this rectangle"""
        x, y = point
        return (self.left <= x <= self.right and 
                self.top <= y <= self.bottom)
    
    def copy(self) -> 'Rect':
        """Create a copy of this rectangle"""
        return Rect(self.x, self.y, self.width, self.height)


@dataclass
class Agent:
    """Agent state without Pygame dependency"""
    rect: Rect
    velocity: List[float]  # [vx, vy]
    grounded: bool
    agent_type: str  # 'fire' or 'water'
    
    def copy(self) -> 'Agent':
        """Create a deep copy of this agent"""
        return Agent(
            rect=self.rect.copy(),
            velocity=self.velocity.copy(),
            grounded=self.grounded,
            agent_type=self.agent_type
        )


class PhysicsEngine:
    """Pure Python physics engine for game simulation"""
    
    def __init__(self, width: int = 960, height: int = 540):
        self.W = width
        self.H = height
        self.VMAX = 3.4  # Maximum horizontal velocity
        self.GRAV = 0.5  # Gravity acceleration
        self.JUMP_VEL = -15.73  # Jump velocity
        self.TERMINAL_VEL = 12  # Maximum fall velocity
    
    def apply_movement(self, agent: Agent, action: int, solids: List[Rect], 
                       hazards: Dict[str, Rect] = None) -> Optional[Agent]:
        """
        Apply physics update based on action
        
        Args:
            agent: The agent to move
            action: Action ID (0=idle, 1=left, 2=right, 3=jump, 4=left+jump, 5=right+jump)
            solids: List of solid obstacles
            hazards: Dictionary of hazard rectangles
            
        Returns:
            Updated agent, or None if agent died
        """
        # Create a copy to avoid modifying original
        agent = agent.copy()
        
        # Horizontal movement
        if action in [1, 4]:  # left or left+jump
            agent.velocity[0] = -self.VMAX
        elif action in [2, 5]:  # right or right+jump
            agent.velocity[0] = self.VMAX
        else:  # idle or jump
            agent.velocity[0] = 0
        
        # Jumping (only if grounded)
        if action in [3, 4, 5] and agent.grounded:
            agent.velocity[1] = self.JUMP_VEL
        
        # Apply gravity
        agent.velocity[1] += self.GRAV
        agent.velocity[1] = min(agent.velocity[1], self.TERMINAL_VEL)
        
        # Horizontal collision detection and resolution
        agent.rect.x += agent.velocity[0]
        for solid in solids:
            if agent.rect.colliderect(solid):
                if agent.velocity[0] > 0:  # Moving right
                    agent.rect.x = solid.left - agent.rect.width
                elif agent.velocity[0] < 0:  # Moving left
                    agent.rect.x = solid.right
        
        # Vertical collision detection and resolution
        agent.rect.y += agent.velocity[1]
        agent.grounded = False
        for solid in solids:
            if agent.rect.colliderect(solid):
                if agent.velocity[1] > 0:  # Falling
                    agent.rect.y = solid.top - agent.rect.height
                    agent.grounded = True
                else:  # Moving up
                    agent.rect.y = solid.bottom
                agent.velocity[1] = 0
        
        # Hazard checking
        if hazards:
            for hazard_name, hazard_rect in hazards.items():
                if hazard_rect and agent.rect.colliderect(hazard_rect):
                    if agent.agent_type == 'fire' and hazard_name == 'water_pool':
                        return None  # Fire agent died in water
                    elif agent.agent_type == 'water' and hazard_name == 'lava_pool':
                        return None  # Water agent died in lava
        
        return agent
    
    def get_state_vector(self, agent: Agent, partner: Agent, 
                         switches: Dict[str, bool], solids: List[Rect],
                         exits: Dict[str, Rect]) -> np.ndarray:
        """
        Extract state vector for RL training
        
        State vector composition (52 dimensions):
        - 0-4: Agent state (pos_x, pos_y, vel_x, vel_y, grounded)
        - 5-9: Partner state (pos_x, pos_y, vel_x, vel_y, grounded)
        - 10-11: Switch states (bridge_up, gate_open)
        - 12-29: Radial clearance (18 rays)
        - 30-31: Distance to exit (normalized)
        - 32-51: Reserved for future features
        """
        state = np.zeros(52)
        
        # Agent state (normalized)
        state[0] = agent.rect.centerx / self.W
        state[1] = agent.rect.centery / self.H
        state[2] = agent.velocity[0] / 10.0
        state[3] = agent.velocity[1] / 10.0
        state[4] = 1.0 if agent.grounded else 0.0
        
        # Partner state (normalized)
        state[5] = partner.rect.centerx / self.W
        state[6] = partner.rect.centery / self.H
        state[7] = partner.velocity[0] / 10.0
        state[8] = partner.velocity[1] / 10.0
        state[9] = 1.0 if partner.grounded else 0.0
        
        # Switch states
        state[10] = 1.0 if switches.get('bridge_up', False) else 0.0
        state[11] = 1.0 if switches.get('gate_open', False) else 0.0
        
        # Radial clearance (18 rays)
        clearances = self.get_radial_clearance(agent.rect, solids, num_rays=18)
        for i, (_, distance, _) in enumerate(clearances[:18]):
            state[12 + i] = min(distance / 500.0, 1.0)  # Normalize and cap at 1.0
        
        # Distance to exit
        if agent.agent_type == 'fire' and 'fire' in exits:
            exit_rect = exits['fire']
            dx = agent.rect.centerx - exit_rect.centerx
            dy = agent.rect.centery - exit_rect.centery
            distance = math.sqrt(dx**2 + dy**2)
            state[30] = min(distance / 1000.0, 1.0)  # Normalized distance
            state[31] = math.atan2(dy, dx) / math.pi  # Angle to exit
        elif agent.agent_type == 'water' and 'water' in exits:
            exit_rect = exits['water']
            dx = agent.rect.centerx - exit_rect.centerx
            dy = agent.rect.centery - exit_rect.centery
            distance = math.sqrt(dx**2 + dy**2)
            state[30] = min(distance / 1000.0, 1.0)
            state[31] = math.atan2(dy, dx) / math.pi
        
        return state
    
    def get_radial_clearance(self, agent_rect: Rect, solids: List[Rect], 
                            hazards: List[Rect] = None, num_rays: int = 18, 
                            max_distance: float = 500) -> List[Tuple[float, float, Tuple[float, float]]]:
        """
        Calculate clearance in radial directions around agent
        
        Returns:
            List of (angle_degrees, distance, endpoint) tuples
        """
        center = agent_rect.center
        clearances = []
        obstacles = list(solids)
        if hazards:
            obstacles.extend([h for h in hazards if h is not None])
        
        angle_step = 360.0 / num_rays
        for i in range(num_rays):
            angle = math.radians(i * angle_step)
            dx, dy = math.cos(angle), math.sin(angle)
            
            min_dist = max_distance
            for obstacle in obstacles:
                t = self._ray_rect_intersection(center, (dx, dy), obstacle, max_distance)
                if t is not None and t < min_dist:
                    min_dist = t
            
            endpoint = (center[0] + dx * min_dist, center[1] + dy * min_dist)
            clearances.append((i * angle_step, min_dist, endpoint))
        
        return clearances
    
    def _ray_rect_intersection(self, start: Tuple[float, float], 
                               direction: Tuple[float, float], 
                               rect: Rect, max_dist: float) -> Optional[float]:
        """
        Calculate ray-rectangle intersection distance
        
        Returns:
            Distance to intersection, or None if no intersection
        """
        dx, dy = direction
        if dx == 0 and dy == 0:
            return None
        
        t_values = []
        
        # Check intersection with each edge
        if abs(dx) > 1e-6:  # Avoid division by zero
            # Left edge
            t = (rect.left - start[0]) / dx
            if 0 <= t <= max_dist:
                y = start[1] + t * dy
                if rect.top <= y <= rect.bottom:
                    t_values.append(t)
            # Right edge
            t = (rect.right - start[0]) / dx
            if 0 <= t <= max_dist:
                y = start[1] + t * dy
                if rect.top <= y <= rect.bottom:
                    t_values.append(t)
        
        if abs(dy) > 1e-6:  # Avoid division by zero
            # Top edge
            t = (rect.top - start[1]) / dy
            if 0 <= t <= max_dist:
                x = start[0] + t * dx
                if rect.left <= x <= rect.right:
                    t_values.append(t)
            # Bottom edge
            t = (rect.bottom - start[1]) / dy
            if 0 <= t <= max_dist:
                x = start[0] + t * dx
                if rect.left <= x <= rect.right:
                    t_values.append(t)
        
        return min(t_values) if t_values else None
    
    def predict_trajectory(self, agent: Agent, action: int, solids: List[Rect],
                          max_steps: int = 100) -> List[Tuple[float, float]]:
        """
        Predict agent's trajectory for a given action
        
        Returns:
            List of (x, y) positions along trajectory
        """
        trajectory = []
        sim_agent = agent.copy()
        
        for step in range(max_steps):
            sim_agent = self.apply_movement(sim_agent, action, solids)
            if sim_agent is None:  # Agent died
                break
            trajectory.append((sim_agent.rect.centerx, sim_agent.rect.centery))
            
            # Stop if agent went off screen
            if sim_agent.rect.centery > self.H:
                break
        
        return trajectory
