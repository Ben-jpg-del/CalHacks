"""
GPU-Accelerated Physics Engine for Fire & Water Game
All physics computed in parallel on GPU using PyTorch
Supports thousands of environments simultaneously
"""

import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional


class TorchPhysicsEngine:
    """
    Vectorized physics engine running entirely on GPU
    Handles collision detection, gravity, and movement for N parallel environments
    """

    def __init__(
        self,
        num_envs: int,
        width: int = 960,
        height: int = 540,
        device: str = 'cuda'
    ):
        self.num_envs = num_envs
        self.width = width
        self.height = height
        self.device = device

        # Physics constants
        self.gravity = 0.5
        self.jump_velocity = -12.0
        self.move_speed = 5.0
        self.max_fall_speed = 15.0
        self.friction = 0.8

        # Agent dimensions
        self.agent_width = 28
        self.agent_height = 36

    def check_collision_batch(
        self,
        pos: torch.Tensor,  # [num_envs, 2] (x, y)
        vel: torch.Tensor,  # [num_envs, 2] (vx, vy)
        solids: torch.Tensor,  # [num_envs, max_solids, 4] (x, y, w, h)
        solids_mask: torch.Tensor  # [num_envs, max_solids] (valid platforms)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorized AABB collision detection for all environments in parallel

        Returns:
            new_pos: [num_envs, 2] - Updated positions after collision
            new_vel: [num_envs, 2] - Updated velocities after collision
            on_ground: [num_envs] - Boolean mask of agents on ground
        """
        # Agent bounding boxes [num_envs, 4] (x, y, w, h)
        agent_boxes = torch.stack([
            pos[:, 0],
            pos[:, 1],
            torch.full((self.num_envs,), self.agent_width, device=self.device),
            torch.full((self.num_envs,), self.agent_height, device=self.device)
        ], dim=1)

        # Proposed new position
        new_pos = pos + vel
        new_agent_boxes = agent_boxes.clone()
        new_agent_boxes[:, :2] = new_pos

        # Expand dimensions for broadcasting
        # agent: [num_envs, 1, 4], solids: [num_envs, max_solids, 4]
        agent_exp = new_agent_boxes.unsqueeze(1)

        # AABB collision test
        # Check if agent overlaps with each solid
        overlap_x = (
            (agent_exp[:, :, 0] < solids[:, :, 0] + solids[:, :, 2]) &
            (agent_exp[:, :, 0] + agent_exp[:, :, 2] > solids[:, :, 0])
        )
        overlap_y = (
            (agent_exp[:, :, 1] < solids[:, :, 1] + solids[:, :, 3]) &
            (agent_exp[:, :, 1] + agent_exp[:, :, 3] > solids[:, :, 1])
        )

        # Collision occurs when both x and y overlap and solid is valid
        collisions = overlap_x & overlap_y & solids_mask  # [num_envs, max_solids]

        # Check if any collision occurred per environment
        has_collision = collisions.any(dim=1)  # [num_envs]

        # Initialize output
        final_pos = new_pos.clone()
        final_vel = vel.clone()
        on_ground = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # For environments with collisions, resolve them
        if has_collision.any():
            # Get the first colliding platform for each env (simplified)
            collision_indices = torch.where(collisions)
            env_idx = collision_indices[0]
            solid_idx = collision_indices[1]

            # For each environment, process first collision only
            processed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

            for i in range(len(env_idx)):
                e = env_idx[i]
                if processed[e]:
                    continue
                processed[e] = True

                s = solid_idx[i]
                solid = solids[e, s]

                # Calculate penetration depths
                agent_left = new_pos[e, 0]
                agent_right = new_pos[e, 0] + self.agent_width
                agent_top = new_pos[e, 1]
                agent_bottom = new_pos[e, 1] + self.agent_height

                solid_left = solid[0]
                solid_right = solid[0] + solid[2]
                solid_top = solid[1]
                solid_bottom = solid[1] + solid[3]

                # Penetration on each side
                pen_left = agent_right - solid_left
                pen_right = solid_right - agent_left
                pen_top = agent_bottom - solid_top
                pen_bottom = solid_bottom - agent_top

                # Resolve smallest penetration
                min_pen = min(pen_left, pen_right, pen_top, pen_bottom)

                if min_pen == pen_top and vel[e, 1] > 0:
                    # Collision from top - agent lands on platform
                    final_pos[e, 1] = solid_top - self.agent_height
                    final_vel[e, 1] = 0
                    on_ground[e] = True
                elif min_pen == pen_bottom and vel[e, 1] < 0:
                    # Collision from bottom - agent hits ceiling
                    final_pos[e, 1] = solid_bottom
                    final_vel[e, 1] = 0
                elif min_pen == pen_left and vel[e, 0] > 0:
                    # Collision from left
                    final_pos[e, 0] = solid_left - self.agent_width
                    final_vel[e, 0] = 0
                elif min_pen == pen_right and vel[e, 0] < 0:
                    # Collision from right
                    final_pos[e, 0] = solid_right
                    final_vel[e, 0] = 0

        return final_pos, final_vel, on_ground

    def update_physics_batch(
        self,
        positions: torch.Tensor,  # [num_envs, 2]
        velocities: torch.Tensor,  # [num_envs, 2]
        actions: torch.Tensor,  # [num_envs] (0-5: NOOP, LEFT, RIGHT, UP, DOWN, JUMP)
        on_ground: torch.Tensor,  # [num_envs]
        solids: torch.Tensor,  # [num_envs, max_solids, 4]
        solids_mask: torch.Tensor  # [num_envs, max_solids]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update physics for all agents in parallel

        Returns:
            new_positions: [num_envs, 2]
            new_velocities: [num_envs, 2]
        """
        new_vel = velocities.clone()

        # Apply gravity
        new_vel[:, 1] += self.gravity
        new_vel[:, 1] = torch.clamp(new_vel[:, 1], -self.max_fall_speed, self.max_fall_speed)

        # Apply actions
        # Action 0: NOOP
        # Action 1: LEFT
        left_mask = (actions == 1)
        new_vel[left_mask, 0] = -self.move_speed

        # Action 2: RIGHT
        right_mask = (actions == 2)
        new_vel[right_mask, 0] = self.move_speed

        # Action 5: JUMP (only if on ground)
        jump_mask = (actions == 5) & on_ground
        new_vel[jump_mask, 1] = self.jump_velocity

        # Apply friction to horizontal velocity
        no_move_mask = (actions != 1) & (actions != 2)
        new_vel[no_move_mask, 0] *= self.friction

        # Check collisions and resolve
        new_pos, new_vel, new_on_ground = self.check_collision_batch(
            positions, new_vel, solids, solids_mask
        )

        # Clamp to screen bounds
        new_pos[:, 0] = torch.clamp(new_pos[:, 0], 0, self.width - self.agent_width)
        new_pos[:, 1] = torch.clamp(new_pos[:, 1], 0, self.height - self.agent_height)

        return new_pos, new_vel

    def check_point_in_rect_batch(
        self,
        points: torch.Tensor,  # [num_envs, 2] (x, y)
        rects: torch.Tensor,  # [num_envs, max_rects, 4] (x, y, w, h)
        rects_mask: torch.Tensor  # [num_envs, max_rects]
    ) -> torch.Tensor:
        """
        Check if agent center points are inside any rectangles

        Returns:
            inside: [num_envs, max_rects] - Boolean mask of containment
        """
        # Agent center points
        centers = points + torch.tensor(
            [self.agent_width / 2, self.agent_height / 2],
            device=self.device
        )

        # Expand for broadcasting [num_envs, 1, 2]
        centers_exp = centers.unsqueeze(1)

        # Check if center is inside each rect
        inside_x = (centers_exp[:, :, 0] >= rects[:, :, 0]) & \
                   (centers_exp[:, :, 0] <= rects[:, :, 0] + rects[:, :, 2])
        inside_y = (centers_exp[:, :, 1] >= rects[:, :, 1]) & \
                   (centers_exp[:, :, 1] <= rects[:, :, 1] + rects[:, :, 3])

        inside = inside_x & inside_y & rects_mask

        return inside
