"""
Parallel Multi-Map Environment for Generalized Policy Training
Uses PyTorch for GPU-accelerated parallel environment execution
Supports multiple maps simultaneously for robust policy learning
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from game_environment import FireWaterEnv
from map_config import LevelLibrary
from map_1 import LevelLibrary as Map1Library
from map_2 import LevelLibrary as Map2Library  # Add your new map


class MapRegistry:
    """Registry for all available maps - easy to extend with new maps"""

    @staticmethod
    def get_all_maps():
        """Get all available maps"""
        return {
            'tutorial': LevelLibrary.get_tutorial_level(),
            'tower': Map1Library.get_tower_level(),
            'map2': Map2Library.get_map2_level(),  # Your new map
            # Add more maps here as you create them:
            # 'map_name': MapLibrary.get_map(),
        }

    @staticmethod
    def get_map(name: str):
        """Get a specific map by name"""
        maps = MapRegistry.get_all_maps()
        if name not in maps:
            raise ValueError(f"Unknown map: {name}. Available: {list(maps.keys())}")
        return maps[name]

    @staticmethod
    def get_map_names():
        """Get list of all map names"""
        return list(MapRegistry.get_all_maps().keys())


class ParallelMultiMapEnv:
    """
    Parallel environment that runs multiple instances across different maps
    Uses CPU-based parallelization with batch processing for GPU training
    """

    def __init__(
        self,
        num_envs_per_map: int = 4,
        map_distribution: Optional[Dict[str, float]] = None,
        reward_function=None,
        device: str = 'cuda'
    ):
        """
        Args:
            num_envs_per_map: Number of parallel environments per map
            map_distribution: Dict mapping map names to sampling weights
                             If None, uses uniform distribution
                             Example: {'tutorial': 0.7, 'tower': 0.3}
            reward_function: Custom reward function (shared across all maps)
            device: PyTorch device for batch processing
        """
        self.device = device

        # Get all available maps
        all_maps = MapRegistry.get_all_maps()

        # Setup map distribution
        if map_distribution is None:
            # Uniform distribution across all maps
            map_names = list(all_maps.keys())
            num_maps = len(map_names)
            self.map_distribution = {name: 1.0 / num_maps for name in map_names}
        else:
            # Validate distribution
            if not np.isclose(sum(map_distribution.values()), 1.0):
                raise ValueError("Map distribution weights must sum to 1.0")
            self.map_distribution = map_distribution

        # Create environments for each map
        self.envs = []
        self.env_map_names = []  # Track which map each env uses

        for map_name, weight in self.map_distribution.items():
            num_envs = int(num_envs_per_map * weight)
            if num_envs == 0:
                num_envs = 1  # At least one env per map

            level = all_maps[map_name]
            for _ in range(num_envs):
                env = FireWaterEnv(level=level)
                # Note: reward_function parameter not yet supported by FireWaterEnv
                # Will need to be added to game_environment.py if custom rewards needed
                self.envs.append(env)
                self.env_map_names.append(map_name)

        self.num_envs = len(self.envs)

        # Get observation and action dimensions from first env
        sample_env = self.envs[0]
        fire_obs, water_obs = sample_env.reset()
        self.obs_dim = fire_obs.shape[0]
        self.action_dim = 6  # Fixed action space

        print(f"[OK] Parallel Multi-Map Environment initialized:")
        print(f"   Total environments: {self.num_envs}")
        print(f"   Maps: {dict(zip(*np.unique(self.env_map_names, return_counts=True)))}")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Device: {self.device}")

    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset all environments

        Returns:
            fire_obs: Tensor of shape [num_envs, obs_dim]
            water_obs: Tensor of shape [num_envs, obs_dim]
        """
        fire_obs_list = []
        water_obs_list = []

        for env in self.envs:
            fire_obs, water_obs = env.reset()
            fire_obs_list.append(fire_obs)
            water_obs_list.append(water_obs)

        fire_obs_batch = torch.FloatTensor(np.array(fire_obs_list)).to(self.device)
        water_obs_batch = torch.FloatTensor(np.array(water_obs_list)).to(self.device)

        return fire_obs_batch, water_obs_batch

    def step(
        self,
        fire_actions: torch.Tensor,
        water_actions: torch.Tensor
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor],  # next_obs (fire, water)
        Tuple[torch.Tensor, torch.Tensor],  # rewards (fire, water)
        Tuple[torch.Tensor, torch.Tensor],  # dones (fire, water)
        List[Dict]                           # infos
    ]:
        """
        Step all environments in parallel

        Args:
            fire_actions: Tensor of shape [num_envs] with fire agent actions
            water_actions: Tensor of shape [num_envs] with water agent actions

        Returns:
            next_observations: (fire_obs, water_obs) tensors [num_envs, obs_dim]
            rewards: (fire_rewards, water_rewards) tensors [num_envs]
            dones: (fire_dones, water_dones) tensors [num_envs]
            infos: List of info dicts
        """
        # Convert to CPU numpy for environment stepping
        fire_actions_np = fire_actions.cpu().numpy()
        water_actions_np = water_actions.cpu().numpy()

        fire_obs_list = []
        water_obs_list = []
        fire_rewards_list = []
        water_rewards_list = []
        fire_dones_list = []
        water_dones_list = []
        infos = []

        # Step each environment
        for i, env in enumerate(self.envs):
            (fire_obs, water_obs), (fire_reward, water_reward), \
            (fire_done, water_done), info = env.step(
                int(fire_actions_np[i]),
                int(water_actions_np[i])
            )

            fire_obs_list.append(fire_obs)
            water_obs_list.append(water_obs)
            fire_rewards_list.append(fire_reward)
            water_rewards_list.append(water_reward)
            fire_dones_list.append(fire_done)
            water_dones_list.append(water_done)
            info['map_name'] = self.env_map_names[i]  # Add map name to info
            infos.append(info)

        # Convert to tensors
        fire_obs_batch = torch.FloatTensor(np.array(fire_obs_list)).to(self.device)
        water_obs_batch = torch.FloatTensor(np.array(water_obs_list)).to(self.device)
        fire_rewards = torch.FloatTensor(np.array(fire_rewards_list)).to(self.device)
        water_rewards = torch.FloatTensor(np.array(water_rewards_list)).to(self.device)
        fire_dones = torch.FloatTensor(np.array(fire_dones_list)).to(self.device)
        water_dones = torch.FloatTensor(np.array(water_dones_list)).to(self.device)

        return (fire_obs_batch, water_obs_batch), \
               (fire_rewards, water_rewards), \
               (fire_dones, water_dones), \
               infos

    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the environment distribution"""
        from collections import Counter
        map_counts = Counter(self.env_map_names)

        return {
            'total_envs': self.num_envs,
            'map_distribution': dict(map_counts),
            'map_percentages': {
                name: count / self.num_envs * 100
                for name, count in map_counts.items()
            }
        }

    def close(self):
        """Close all environments"""
        for env in self.envs:
            if hasattr(env, 'close'):
                env.close()
            # FireWaterEnv doesn't have close() method, just clear reference
            del env
        self.envs = []


class CurriculumMultiMapEnv(ParallelMultiMapEnv):
    """
    Adaptive environment that adjusts map difficulty based on performance
    Starts with easier maps and gradually introduces harder ones
    """

    def __init__(
        self,
        num_envs_per_map: int = 4,
        initial_map: str = 'tutorial',
        curriculum_schedule: Optional[Dict[int, Dict[str, float]]] = None,
        reward_function=None,
        device: str = 'cuda'
    ):
        """
        Args:
            num_envs_per_map: Number of parallel environments per map
            initial_map: Map to start training with
            curriculum_schedule: Episode -> map_distribution mapping
                Example: {
                    0: {'tutorial': 1.0},
                    1000: {'tutorial': 0.7, 'tower': 0.3},
                    2000: {'tutorial': 0.5, 'tower': 0.5},
                }
            reward_function: Custom reward function
            device: PyTorch device
        """
        if curriculum_schedule is None:
            curriculum_schedule = {
                0: {'tutorial': 1.0},
                1000: {'tutorial': 0.7, 'tower': 0.3},
                2000: {'tutorial': 0.5, 'tower': 0.5},
            }

        self.curriculum_schedule = curriculum_schedule
        self.current_episode = 0

        # Start with initial distribution
        initial_distribution = curriculum_schedule[0]

        super().__init__(
            num_envs_per_map=num_envs_per_map,
            map_distribution=initial_distribution,
            reward_function=None,  # Not yet supported
            device=device
        )

    def update_curriculum(self, episode: int):
        """
        Update map distribution based on curriculum schedule
        Call this at the end of each episode
        """
        self.current_episode = episode

        # Find the latest curriculum milestone we've passed
        milestones = sorted([e for e in self.curriculum_schedule.keys() if e <= episode])
        if milestones:
            latest_milestone = milestones[-1]
            new_distribution = self.curriculum_schedule[latest_milestone]

            # Check if distribution changed
            if new_distribution != self.map_distribution:
                print(f"\n[CURRICULUM] Update at Episode {episode}:")
                print(f"   Old distribution: {self.map_distribution}")
                print(f"   New distribution: {new_distribution}")

                # Rebuild environments with new distribution
                # Close old environments
                for env in self.envs:
                    if hasattr(env, 'close'):
                        env.close()
                    del env

                # Reinitialize with new distribution
                ParallelMultiMapEnv.__init__(
                    self,
                    num_envs_per_map=len(self.envs) // len(self.map_distribution),
                    map_distribution=new_distribution,
                    reward_function=None,
                    device=self.device
                )

                # Restore curriculum tracking
                self.curriculum_schedule = self.curriculum_schedule
                self.current_episode = episode


# ============================================
# EXAMPLE USAGE
# ============================================

if __name__ == "__main__":
    import torch
    print("Testing Parallel Multi-Map Environment\n")

    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    # Test 1: Uniform distribution
    print("=" * 60)
    print("Test 1: Uniform Distribution")
    print("=" * 60)
    env = ParallelMultiMapEnv(num_envs_per_map=4, device=device)
    print(env.get_statistics())

    # Reset and step
    fire_obs, water_obs = env.reset()
    print(f"\nObservation shapes: fire={fire_obs.shape}, water={water_obs.shape}")

    # Random actions
    fire_actions = torch.randint(0, 6, (env.num_envs,))
    water_actions = torch.randint(0, 6, (env.num_envs,))

    (fire_obs_next, water_obs_next), (fire_rewards, water_rewards), \
    (fire_dones, water_dones), infos = env.step(fire_actions, water_actions)

    print(f"Rewards: fire={fire_rewards.mean():.2f}, water={water_rewards.mean():.2f}")
    print(f"Maps used: {[info['map_name'] for info in infos]}")

    env.close()

    # Test 2: Custom distribution
    print("\n" + "=" * 60)
    print("Test 2: Custom Distribution (70% tutorial, 30% tower)")
    print("=" * 60)
    env = ParallelMultiMapEnv(
        num_envs_per_map=10,
        map_distribution={'tutorial': 0.7, 'tower': 0.3},
        device=device
    )
    print(env.get_statistics())
    env.close()

    # Test 3: Curriculum learning
    print("\n" + "=" * 60)
    print("Test 3: Curriculum Learning")
    print("=" * 60)
    env = CurriculumMultiMapEnv(num_envs_per_map=4, device=device)
    print(f"Initial: {env.get_statistics()}")

    # Simulate progression
    env.update_curriculum(1000)
    print(f"After 1000 episodes: {env.get_statistics()}")

    env.update_curriculum(2000)
    print(f"After 2000 episodes: {env.get_statistics()}")

    env.close()

    print("\n[OK] All tests passed!")
