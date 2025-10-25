"""
Staged Milestone Reward Function
Mathematical reward structure with explicit stage progression

Based on the staged milestone reward specification:
- Stage 0: Navigate to pressure plates
- Stage 1: Navigate to exits
- Stage 2: Mission complete

Reward Components:
  r_prog(t) = max(D_t - D_{t+1}, 0) / diam  # Progress toward current goal
  r_plates(t) = β * IND[stage 0→1]          # Plate activation bonus
  r_finish(t) = Γ * IND[stage 1→2]          # Completion bonus
"""

import numpy as np
from typing import Dict, Tuple
from physics_engine import Agent, Rect


class StagedMilestoneRewardConfig:
    """Configuration for staged milestone rewards"""

    def __init__(self):
        # ========== MILESTONE BONUSES ==========
        self.beta = 50.0            # β: Bonus for activating plates (stage 0→1)
        self.gamma = 200.0          # Γ: Bonus for reaching exits (stage 1→2)

        # ========== DISTANCE NORMALIZATION ==========
        self.diam_0 = 1000.0        # Distance normalization for stage 0 (to plates)
        self.diam_1 = 1000.0        # Distance normalization for stage 1 (to exits)

        # ========== DETECTION RADII ==========
        self.rho_p = 20.0           # ρ_p: Plate detection radius (pixels)
        self.rho_e = 20.0           # ρ_e: Exit detection radius (pixels)

        # ========== PENALTIES ==========
        self.step_penalty = -0.1    # Small step penalty (encourages efficiency)
        self.death_penalty = -100.0 # Death penalty


class StagedMilestoneRewardFunction:
    """
    Staged Milestone Reward Function

    Implements a mathematical reward structure with three stages:
    - Stage 0: Navigate to pressure plates (bridge/gate)
    - Stage 1: Navigate to exits
    - Stage 2: Mission complete

    Rewards are structured to guide agents through sub-goals:
    1. Progress rewards (only positive when getting closer)
    2. Milestone bonuses (for stage transitions)
    3. Small step penalty (encourages efficiency)
    """

    def __init__(self, config: StagedMilestoneRewardConfig = None):
        self.config = config or StagedMilestoneRewardConfig()

        # Stage tracking
        self.stage = 0  # 0 = navigate to plates, 1 = navigate to exits, 2 = done

        # Previous distances for progress calculation
        self.prev_fire_distance = None
        self.prev_water_distance = None

        # Track if milestones were just achieved
        self.just_activated_plates = False
        self.just_reached_exits = False

    def reset(self):
        """Reset for new episode"""
        self.stage = 0
        self.prev_fire_distance = None
        self.prev_water_distance = None
        self.just_activated_plates = False
        self.just_reached_exits = False

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
        Calculate staged milestone rewards

        Returns:
            (fire_reward, water_reward)
        """
        fire_reward = 0.0
        water_reward = 0.0

        # ========== DEATH PENALTY ==========
        if fire_died:
            return self.config.death_penalty, 0.0
        if water_died:
            return 0.0, self.config.death_penalty

        # ========== SUCCESS BONUS (STAGE 1→2) ==========
        if fire_won and water_won:
            fire_reward += self.config.gamma
            water_reward += self.config.gamma
            self.stage = 2
            return fire_reward, water_reward

        # ========== STEP PENALTY ==========
        fire_reward += self.config.step_penalty
        water_reward += self.config.step_penalty

        # ========== STAGE PROGRESSION ==========

        # Stage 0→1: Check if plates activated
        if self.stage == 0 and bridge_activated and gate_activated:
            fire_reward += self.config.beta
            water_reward += self.config.beta
            self.stage = 1
            self.prev_fire_distance = None  # Reset distance tracking
            self.prev_water_distance = None
            self.just_activated_plates = True

        # ========== PROGRESS REWARDS ==========

        if self.stage == 0:
            # Stage 0: Navigate to plates
            fire_reward += self._calculate_progress_to_plates(
                fire_agent, bridge_activated, 'fire'
            )
            water_reward += self._calculate_progress_to_plates(
                water_agent, gate_activated, 'water'
            )

        elif self.stage == 1:
            # Stage 1: Navigate to exits
            fire_reward += self._calculate_progress_to_exit(
                fire_agent, exits['fire'], 'fire'
            )
            water_reward += self._calculate_progress_to_exit(
                water_agent, exits['water'], 'water'
            )

        return fire_reward, water_reward

    def _calculate_progress_to_plates(
        self,
        agent: Agent,
        plate_activated: bool,
        agent_type: str
    ) -> float:
        """
        Calculate progress reward toward pressure plate

        Returns only positive rewards when getting closer
        """
        if plate_activated:
            return 0.0  # Already activated, no more progress needed

        # Get plate position (approximated - adjust based on your level)
        # Fire agent needs to reach bridge plate
        # Water agent needs to reach gate plate
        if agent_type == 'fire':
            # Bridge plate location (approximate - adjust for your level)
            plate_pos = (700, 400)  # Example position
        else:
            # Gate plate location (approximate - adjust for your level)
            plate_pos = (300, 400)  # Example position

        # Calculate current distance
        current_distance = np.sqrt(
            (agent.rect.centerx - plate_pos[0])**2 +
            (agent.rect.centery - plate_pos[1])**2
        )

        # Progress reward: only positive when getting closer
        progress_reward = 0.0
        if self.prev_fire_distance is not None if agent_type == 'fire' else self.prev_water_distance is not None:
            prev_dist = self.prev_fire_distance if agent_type == 'fire' else self.prev_water_distance
            distance_delta = prev_dist - current_distance

            if distance_delta > 0:  # Only reward progress (getting closer)
                progress_reward = distance_delta / self.config.diam_0

        # Update previous distance
        if agent_type == 'fire':
            self.prev_fire_distance = current_distance
        else:
            self.prev_water_distance = current_distance

        return progress_reward

    def _calculate_progress_to_exit(
        self,
        agent: Agent,
        exit_rect: Rect,
        agent_type: str
    ) -> float:
        """
        Calculate progress reward toward exit

        Returns only positive rewards when getting closer
        """
        # Calculate current distance to exit
        current_distance = np.sqrt(
            (agent.rect.centerx - exit_rect.centerx)**2 +
            (agent.rect.centery - exit_rect.centery)**2
        )

        # Progress reward: only positive when getting closer
        progress_reward = 0.0
        if agent_type == 'fire':
            if self.prev_fire_distance is not None:
                distance_delta = self.prev_fire_distance - current_distance
                if distance_delta > 0:  # Only reward progress
                    progress_reward = distance_delta / self.config.diam_1
            self.prev_fire_distance = current_distance
        else:
            if self.prev_water_distance is not None:
                distance_delta = self.prev_water_distance - current_distance
                if distance_delta > 0:  # Only reward progress
                    progress_reward = distance_delta / self.config.diam_1
            self.prev_water_distance = current_distance

        return progress_reward

    def get_stage_info(self) -> Dict:
        """Get current stage information"""
        return {
            'stage': self.stage,
            'stage_name': ['Navigate to Plates', 'Navigate to Exits', 'Complete'][self.stage],
            'just_activated_plates': self.just_activated_plates,
            'just_reached_exits': self.just_reached_exits
        }


# ========== EXAMPLE USAGE ==========
if __name__ == "__main__":
    print("Testing Staged Milestone Reward Function\n")
    print("=" * 60)

    from physics_engine import Agent, Rect

    # Create dummy agents
    fire_agent = Agent(Rect(100, 100, 28, 36), [0, 0], False, 'fire')
    water_agent = Agent(Rect(200, 200, 28, 36), [0, 0], False, 'water')

    exits = {
        'fire': Rect(880, 388, 36, 36),
        'water': Rect(60, 308, 36, 36)
    }

    hazards = {}

    # Create reward function
    config = StagedMilestoneRewardConfig()
    config.beta = 50.0
    config.gamma = 200.0

    reward_fn = StagedMilestoneRewardFunction(config)

    print("Configuration:")
    print(f"  β (plate bonus): {config.beta}")
    print(f"  Γ (finish bonus): {config.gamma}")
    print(f"  diam_0: {config.diam_0}")
    print(f"  diam_1: {config.diam_1}")
    print(f"  Step penalty: {config.step_penalty}\n")

    # Test Stage 0 (navigate to plates)
    print("Stage 0: Navigate to plates")
    print("-" * 60)

    for i in range(3):
        fire_reward, water_reward = reward_fn.calculate_rewards(
            fire_agent, water_agent, 0, 0,
            False, False, False, False,
            False, False, exits, hazards
        )
        print(f"  Step {i+1}: Fire={fire_reward:.3f}, Water={water_reward:.3f}")

    # Activate plates (transition to Stage 1)
    print("\nPlates activated! Transitioning to Stage 1...")
    fire_reward, water_reward = reward_fn.calculate_rewards(
        fire_agent, water_agent, 0, 0,
        False, False, False, False,
        True, True, exits, hazards  # Bridge and gate activated
    )
    print(f"  Milestone reward: Fire={fire_reward:.3f}, Water={water_reward:.3f}")
    print(f"  Current stage: {reward_fn.stage}\n")

    # Test Stage 1 (navigate to exits)
    print("Stage 1: Navigate to exits")
    print("-" * 60)

    for i in range(3):
        fire_reward, water_reward = reward_fn.calculate_rewards(
            fire_agent, water_agent, 0, 0,
            False, False, False, False,
            True, True, exits, hazards
        )
        print(f"  Step {i+1}: Fire={fire_reward:.3f}, Water={water_reward:.3f}")

    # Reach exits (transition to Stage 2)
    print("\nExits reached! Mission complete...")
    fire_reward, water_reward = reward_fn.calculate_rewards(
        fire_agent, water_agent, 0, 0,
        False, False, True, True,  # Both won
        True, True, exits, hazards
    )
    print(f"  Completion reward: Fire={fire_reward:.3f}, Water={water_reward:.3f}")
    print(f"  Final stage: {reward_fn.stage}\n")

    print("=" * 60)
    print("✓ Staged Milestone Reward Function test complete!")
