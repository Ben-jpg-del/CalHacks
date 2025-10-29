"""
GPU-Accelerated Training Module for Fire & Water

This module provides a complete GPU implementation of the game environment
for massive training speedup (50-100x faster than CPU).

Main components:
- torch_physics: Vectorized physics engine
- torch_env: Parallel GPU environment
- train_gpu: Training script with GPU-optimized replay buffer

Usage:
    from gpu_training.torch_env import TorchFireWaterEnv
    from gpu_training.train_gpu import train_gpu

    # Train with 1024 parallel environments
    train_gpu(num_envs=1024)
"""

from .torch_physics import TorchPhysicsEngine
from .torch_env import TorchFireWaterEnv
from .train_gpu import GPUDQNAgent, GPUReplayBuffer, train_gpu

__all__ = [
    'TorchPhysicsEngine',
    'TorchFireWaterEnv',
    'GPUDQNAgent',
    'GPUReplayBuffer',
    'train_gpu',
]

__version__ = '1.0.0'
