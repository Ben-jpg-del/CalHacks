"""
GPU-Accelerated Fire & Water Environment
Runs thousands of game instances in parallel on GPU
Compatible with existing checkpoint/visualization systems
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_physics import TorchPhysicsEngine


class TorchFireWaterEnv:
    """
    Vectorized Fire & Water environment running on GPU
    All game instances execute in parallel
    """

    def __init__(
        self,
        num_envs: int,
        level_configs: List,  # List of level configs from map_config.py
        device: str = 'cuda',
        max_solids: int = 50,  # Maximum platforms per level
        max_hazards: int = 10,  # Maximum hazards per level
        max_switches: int = 5   # Maximum pressure plates per level
    ):
        self.num_envs = num_envs
        self.device = device
        self.max_solids = max_solids
        self.max_hazards = max_hazards
        self.max_switches = max_switches

        # Physics engine
        self.physics = TorchPhysicsEngine(num_envs, device=device)

        # Store level configs
        self.level_configs = level_configs
        assert len(level_configs) == num_envs, "Must provide one level config per environment"

        # Initialize level geometry on GPU
        self._initialize_levels()

        # Agent states [num_envs, 2] (x, y positions)
        self.fire_pos = torch.zeros((num_envs, 2), device=device)
        self.water_pos = torch.zeros((num_envs, 2), device=device)
        self.fire_vel = torch.zeros((num_envs, 2), device=device)
        self.water_vel = torch.zeros((num_envs, 2), device=device)

        # Game state
        self.fire_on_ground = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.water_on_ground = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fire_reached_exit = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.water_reached_exit = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.fire_dead = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.water_dead = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Switch states [num_envs, max_switches]
        self.plate_a_active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.plate_b_active = torch.zeros(num_envs, dtype=torch.bool, device=device)

        # Observation space: 52 features per agent
        self.obs_dim = 52
        self.action_dim = 6

    def _initialize_levels(self):
        """Convert level configs to GPU tensors"""
        # Base solids [num_envs, max_solids, 4]
        self.base_solids = torch.zeros(
            (self.num_envs, self.max_solids, 4),
            device=self.device
        )
        self.base_solids_mask = torch.zeros(
            (self.num_envs, self.max_solids),
            dtype=torch.bool,
            device=self.device
        )

        # Dynamic platforms (bridges/gates)
        self.bridge_rects = torch.zeros((self.num_envs, 4), device=self.device)
        self.gate_rects = torch.zeros((self.num_envs, 4), device=self.device)
        self.has_bridge = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_gate = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Hazards [num_envs, max_hazards, 4]
        self.hazards_water = torch.zeros(
            (self.num_envs, self.max_hazards, 4),
            device=self.device
        )
        self.hazards_lava = torch.zeros(
            (self.num_envs, self.max_hazards, 4),
            device=self.device
        )
        self.hazards_water_mask = torch.zeros(
            (self.num_envs, self.max_hazards),
            dtype=torch.bool,
            device=self.device
        )
        self.hazards_lava_mask = torch.zeros(
            (self.num_envs, self.max_hazards),
            dtype=torch.bool,
            device=self.device
        )

        # Pressure plates
        self.plate_a_rects = torch.zeros((self.num_envs, 4), device=self.device)
        self.plate_b_rects = torch.zeros((self.num_envs, 4), device=self.device)
        self.has_plate_a = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_plate_b = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Exit zones
        self.exit_fire_rects = torch.zeros((self.num_envs, 4), device=self.device)
        self.exit_water_rects = torch.zeros((self.num_envs, 4), device=self.device)

        # Spawn positions
        self.fire_spawn = torch.zeros((self.num_envs, 2), device=self.device)
        self.water_spawn = torch.zeros((self.num_envs, 2), device=self.device)

        # Load each level config
        for i, config in enumerate(self.level_configs):
            # Base solids
            base_solids = config.base_solids
            n_solids = min(len(base_solids), self.max_solids)
            for j, solid in enumerate(base_solids[:n_solids]):
                self.base_solids[i, j] = torch.tensor(
                    [solid.x, solid.y, solid.width, solid.height],
                    device=self.device
                )
                self.base_solids_mask[i, j] = True

            # Dynamic platforms
            if hasattr(config, 'bridge') and config.bridge is not None:
                b = config.bridge
                self.bridge_rects[i] = torch.tensor(
                    [b.x, b.y, b.width, b.height],
                    device=self.device
                )
                self.has_bridge[i] = True

            if hasattr(config, 'gate') and config.gate is not None:
                g = config.gate
                self.gate_rects[i] = torch.tensor(
                    [g.x, g.y, g.width, g.height],
                    device=self.device
                )
                self.has_gate[i] = True

            # Hazards
            hazards = config.get_hazards()
            if 'water_pool' in hazards:
                h = hazards['water_pool']
                self.hazards_water[i, 0] = torch.tensor(
                    [h.x, h.y, h.width, h.height],
                    device=self.device
                )
                self.hazards_water_mask[i, 0] = True

            if 'lava_pool' in hazards:
                h = hazards['lava_pool']
                self.hazards_lava[i, 0] = torch.tensor(
                    [h.x, h.y, h.width, h.height],
                    device=self.device
                )
                self.hazards_lava_mask[i, 0] = True

            # Pressure plates
            switches = config.get_switches()
            if 'plate_a' in switches:
                p = switches['plate_a']
                self.plate_a_rects[i] = torch.tensor(
                    [p.x, p.y, p.width, p.height],
                    device=self.device
                )
                self.has_plate_a[i] = True

            if 'plate_b' in switches:
                p = switches['plate_b']
                self.plate_b_rects[i] = torch.tensor(
                    [p.x, p.y, p.width, p.height],
                    device=self.device
                )
                self.has_plate_b[i] = True

            # Exit zones
            exits = config.get_exits()
            if 'fire' in exits:
                e = exits['fire']
                self.exit_fire_rects[i] = torch.tensor(
                    [e.x, e.y, e.width, e.height],
                    device=self.device
                )
            if 'water' in exits:
                e = exits['water']
                self.exit_water_rects[i] = torch.tensor(
                    [e.x, e.y, e.width, e.height],
                    device=self.device
                )

            # Spawn positions
            self.fire_spawn[i] = torch.tensor(
                [config.fire_start[0], config.fire_start[1]],
                device=self.device
            )
            self.water_spawn[i] = torch.tensor(
                [config.water_start[0], config.water_start[1]],
                device=self.device
            )

    def _get_current_solids(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get active solid platforms based on current switch states

        Returns:
            solids: [num_envs, max_solids+2, 4]
            solids_mask: [num_envs, max_solids+2]
        """
        # Start with base solids
        solids = self.base_solids.clone()
        solids_mask = self.base_solids_mask.clone()

        # Add dynamic platforms if activated
        for i in range(self.num_envs):
            idx = self.base_solids_mask[i].sum()  # Next available slot

            # Add bridge if active
            if self.has_bridge[i] and self.plate_a_active[i] and idx < self.max_solids:
                solids[i, idx] = self.bridge_rects[i]
                solids_mask[i, idx] = True
                idx += 1

            # Add gate if active
            if self.has_gate[i] and self.plate_b_active[i] and idx < self.max_solids:
                solids[i, idx] = self.gate_rects[i]
                solids_mask[i, idx] = True

        return solids, solids_mask

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset all environments

        Returns:
            fire_obs: [num_envs, obs_dim]
            water_obs: [num_envs, obs_dim]
        """
        # Reset positions
        self.fire_pos = self.fire_spawn.clone()
        self.water_pos = self.water_spawn.clone()

        # Reset velocities
        self.fire_vel.zero_()
        self.water_vel.zero_()

        # Reset states
        self.fire_on_ground.zero_()
        self.water_on_ground.zero_()
        self.fire_reached_exit.zero_()
        self.water_reached_exit.zero_()
        self.fire_dead.zero_()
        self.water_dead.zero_()
        self.plate_a_active.zero_()
        self.plate_b_active.zero_()

        # Get observations
        fire_obs = self._get_observation('fire')
        water_obs = self._get_observation('water')

        return fire_obs, water_obs

    def step(
        self,
        fire_actions: torch.Tensor,  # [num_envs]
        water_actions: torch.Tensor  # [num_envs]
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],  # observations
        Tuple[torch.Tensor, torch.Tensor],  # rewards
        Tuple[torch.Tensor, torch.Tensor],  # dones
        List[Dict]  # infos
    ]:
        """Step all environments in parallel on GPU"""

        # Get current solid platforms
        solids, solids_mask = self._get_current_solids()

        # Update fire agent physics
        self.fire_pos, self.fire_vel = self.physics.update_physics_batch(
            self.fire_pos,
            self.fire_vel,
            fire_actions,
            self.fire_on_ground,
            solids,
            solids_mask
        )

        # Update water agent physics
        self.water_pos, self.water_vel = self.physics.update_physics_batch(
            self.water_pos,
            self.water_vel,
            water_actions,
            self.water_on_ground,
            solids,
            solids_mask
        )

        # Update on_ground status
        _, _, self.fire_on_ground = self.physics.check_collision_batch(
            self.fire_pos,
            torch.zeros_like(self.fire_vel),
            solids,
            solids_mask
        )
        _, _, self.water_on_ground = self.physics.check_collision_batch(
            self.water_pos,
            torch.zeros_like(self.water_vel),
            solids,
            solids_mask
        )

        # Check pressure plates
        self._update_switches()

        # Check hazards
        self._check_hazards()

        # Check exit zones
        self._check_exits()

        # Compute rewards and dones
        fire_reward, water_reward = self._compute_rewards()
        fire_done, water_done = self._compute_dones()

        # Get observations
        fire_obs = self._get_observation('fire')
        water_obs = self._get_observation('water')

        # Build info dicts (convert to CPU for compatibility)
        infos = self._build_infos()

        return (fire_obs, water_obs), \
               (fire_reward, water_reward), \
               (fire_done, water_done), \
               infos

    def _update_switches(self):
        """Check if agents are on pressure plates"""
        # Water activates plate_a
        if self.has_plate_a.any():
            plate_a_expanded = self.plate_a_rects.unsqueeze(1)  # [num_envs, 1, 4]
            water_on_plate_a = self.physics.check_point_in_rect_batch(
                self.water_pos,
                plate_a_expanded,
                self.has_plate_a.unsqueeze(1)
            ).squeeze(1)
            self.plate_a_active = water_on_plate_a

        # Fire activates plate_b
        if self.has_plate_b.any():
            plate_b_expanded = self.plate_b_rects.unsqueeze(1)  # [num_envs, 1, 4]
            fire_on_plate_b = self.physics.check_point_in_rect_batch(
                self.fire_pos,
                plate_b_expanded,
                self.has_plate_b.unsqueeze(1)
            ).squeeze(1)
            self.plate_b_active = fire_on_plate_b

    def _check_hazards(self):
        """Check if agents touched hazards"""
        # Fire dies in water pools
        fire_in_water = self.physics.check_point_in_rect_batch(
            self.fire_pos,
            self.hazards_water,
            self.hazards_water_mask
        ).any(dim=1)
        self.fire_dead |= fire_in_water

        # Water dies in lava pools
        water_in_lava = self.physics.check_point_in_rect_batch(
            self.water_pos,
            self.hazards_lava,
            self.hazards_lava_mask
        ).any(dim=1)
        self.water_dead |= water_in_lava

    def _check_exits(self):
        """Check if agents reached exit zones"""
        # Fire exit
        fire_exit_expanded = self.exit_fire_rects.unsqueeze(1)
        fire_at_exit = self.physics.check_point_in_rect_batch(
            self.fire_pos,
            fire_exit_expanded,
            torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        ).squeeze(1)
        self.fire_reached_exit |= fire_at_exit

        # Water exit
        water_exit_expanded = self.exit_water_rects.unsqueeze(1)
        water_at_exit = self.physics.check_point_in_rect_batch(
            self.water_pos,
            water_exit_expanded,
            torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        ).squeeze(1)
        self.water_reached_exit |= water_at_exit

    def _compute_rewards(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute rewards for all environments"""
        fire_reward = torch.zeros(self.num_envs, device=self.device)
        water_reward = torch.zeros(self.num_envs, device=self.device)

        # Penalty for death
        fire_reward[self.fire_dead] = -10.0
        water_reward[self.water_dead] = -10.0

        # Reward for reaching exit
        fire_reward[self.fire_reached_exit] = 100.0
        water_reward[self.water_reached_exit] = 100.0

        # Bonus if both reached exit
        both_won = self.fire_reached_exit & self.water_reached_exit
        fire_reward[both_won] += 100.0
        water_reward[both_won] += 100.0

        # Small living penalty to encourage speed
        fire_reward -= 0.1
        water_reward -= 0.1

        return fire_reward, water_reward

    def _compute_dones(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Check if episodes are done"""
        fire_done = self.fire_dead | self.fire_reached_exit
        water_done = self.water_dead | self.water_reached_exit

        return fire_done, water_done

    def _get_observation(self, agent: str) -> torch.Tensor:
        """
        Build observation vector for agent

        Returns:
            obs: [num_envs, obs_dim]
        """
        if agent == 'fire':
            own_pos = self.fire_pos
            own_vel = self.fire_vel
            own_on_ground = self.fire_on_ground
            other_pos = self.water_pos
        else:
            own_pos = self.water_pos
            own_vel = self.water_vel
            own_on_ground = self.water_on_ground
            other_pos = self.fire_pos

        obs_list = []

        # Own state (6)
        obs_list.append(own_pos / torch.tensor([960, 540], device=self.device))  # Normalized position
        obs_list.append(own_vel / 20.0)  # Normalized velocity
        obs_list.append(own_on_ground.float().unsqueeze(1))  # On ground flag
        obs_list.append(self.fire_reached_exit.float().unsqueeze(1) if agent == 'fire' else self.water_reached_exit.float().unsqueeze(1))

        # Other agent position (2)
        obs_list.append(other_pos / torch.tensor([960, 540], device=self.device))

        # Switch states (2)
        obs_list.append(self.plate_a_active.float().unsqueeze(1))
        obs_list.append(self.plate_b_active.float().unsqueeze(1))

        # Distance to exits (2)
        if agent == 'fire':
            dist_to_exit = torch.norm(own_pos - self.exit_fire_rects[:, :2], dim=1, keepdim=True) / 1000.0
        else:
            dist_to_exit = torch.norm(own_pos - self.exit_water_rects[:, :2], dim=1, keepdim=True) / 1000.0
        obs_list.append(dist_to_exit)

        # Nearest hazard distance (1)
        if agent == 'fire':
            hazard_dists = torch.norm(
                own_pos.unsqueeze(1) - self.hazards_water[:, :, :2],
                dim=2
            )
            hazard_dists = torch.where(
                self.hazards_water_mask,
                hazard_dists,
                torch.full_like(hazard_dists, 10000.0)
            )
            min_hazard_dist = hazard_dists.min(dim=1)[0].unsqueeze(1) / 1000.0
        else:
            hazard_dists = torch.norm(
                own_pos.unsqueeze(1) - self.hazards_lava[:, :, :2],
                dim=2
            )
            hazard_dists = torch.where(
                self.hazards_lava_mask,
                hazard_dists,
                torch.full_like(hazard_dists, 10000.0)
            )
            min_hazard_dist = hazard_dists.min(dim=1)[0].unsqueeze(1) / 1000.0
        obs_list.append(min_hazard_dist)

        # Nearest platform info (up to 5 platforms * 8 = 40 features)
        solids, solids_mask = self._get_current_solids()
        platform_dists = torch.norm(
            own_pos.unsqueeze(1) - solids[:, :, :2],
            dim=2
        )
        platform_dists = torch.where(
            solids_mask,
            platform_dists,
            torch.full_like(platform_dists, 10000.0)
        )

        # Get 5 nearest platforms
        topk = min(5, solids.size(1))
        nearest_idx = torch.topk(platform_dists, topk, dim=1, largest=False)[1]

        for k in range(5):
            if k < topk:
                idx = nearest_idx[:, k]
                platform_info = torch.stack([
                    solids[torch.arange(self.num_envs), idx, 0] / 960.0,  # x
                    solids[torch.arange(self.num_envs), idx, 1] / 540.0,  # y
                    solids[torch.arange(self.num_envs), idx, 2] / 960.0,  # w
                    solids[torch.arange(self.num_envs), idx, 3] / 540.0,  # h
                    platform_dists[torch.arange(self.num_envs), idx] / 1000.0,  # dist
                    (own_pos[:, 0] - solids[torch.arange(self.num_envs), idx, 0]) / 500.0,  # dx
                    (own_pos[:, 1] - solids[torch.arange(self.num_envs), idx, 1]) / 500.0,  # dy
                    solids_mask[torch.arange(self.num_envs), idx].float()  # valid
                ], dim=1)
            else:
                platform_info = torch.zeros((self.num_envs, 8), device=self.device)
            obs_list.append(platform_info)

        # Concatenate all features
        obs = torch.cat(obs_list, dim=1)

        # Pad or trim to obs_dim
        if obs.size(1) < self.obs_dim:
            padding = torch.zeros(
                (self.num_envs, self.obs_dim - obs.size(1)),
                device=self.device
            )
            obs = torch.cat([obs, padding], dim=1)
        elif obs.size(1) > self.obs_dim:
            obs = obs[:, :self.obs_dim]

        return obs

    def _build_infos(self) -> List[Dict]:
        """Build info dictionaries (move to CPU for compatibility)"""
        infos = []
        for i in range(self.num_envs):
            info = {
                'fire_reached_exit': self.fire_reached_exit[i].item(),
                'water_reached_exit': self.water_reached_exit[i].item(),
                'both_won': (self.fire_reached_exit[i] & self.water_reached_exit[i]).item(),
                'fire_dead': self.fire_dead[i].item(),
                'water_dead': self.water_dead[i].item(),
            }
            infos.append(info)
        return infos

    def get_state_for_visualization(self, env_idx: int = 0) -> Dict:
        """
        Export state of a specific environment for visualization

        Args:
            env_idx: Which environment to export

        Returns:
            state: Dictionary compatible with visualize.py
        """
        i = env_idx

        # Get current solids
        solids, solids_mask = self._get_current_solids()
        active_solids = []
        for j in range(self.max_solids):
            if solids_mask[i, j]:
                s = solids[i, j]
                active_solids.append({
                    'x': s[0].item(),
                    'y': s[1].item(),
                    'width': s[2].item(),
                    'height': s[3].item()
                })

        state = {
            'fire_pos': {
                'x': self.fire_pos[i, 0].item(),
                'y': self.fire_pos[i, 1].item()
            },
            'water_pos': {
                'x': self.water_pos[i, 0].item(),
                'y': self.water_pos[i, 1].item()
            },
            'solids': active_solids,
            'fire_reached_exit': self.fire_reached_exit[i].item(),
            'water_reached_exit': self.water_reached_exit[i].item(),
            'fire_dead': self.fire_dead[i].item(),
            'water_dead': self.water_dead[i].item(),
            'plate_a_active': self.plate_a_active[i].item(),
            'plate_b_active': self.plate_b_active[i].item(),
        }

        return state
