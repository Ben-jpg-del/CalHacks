"""
Quick test script to verify GPU training works correctly
Run this before full training to catch any issues
"""

import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch_env import TorchFireWaterEnv
from torch_physics import TorchPhysicsEngine
from train_gpu import GPUDQNAgent, GPUReplayBuffer
from map_config import LevelLibrary
from map_1 import LevelLibrary as Map1Library
from map_2 import LevelLibrary as Map2Library


def test_physics_engine():
    """Test vectorized physics"""
    print("Testing Physics Engine...")

    num_envs = 8
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    physics = TorchPhysicsEngine(num_envs, device=device)

    # Test positions
    positions = torch.randn(num_envs, 2, device=device) * 100 + 200
    velocities = torch.randn(num_envs, 2, device=device) * 5

    # Test collision detection
    solids = torch.tensor([[100, 200, 50, 20]], device=device).repeat(num_envs, 5, 1)
    solids_mask = torch.ones(num_envs, 5, dtype=torch.bool, device=device)

    new_pos, new_vel, on_ground = physics.check_collision_batch(
        positions, velocities, solids, solids_mask
    )

    assert new_pos.shape == (num_envs, 2)
    assert new_vel.shape == (num_envs, 2)
    assert on_ground.shape == (num_envs,)

    print("  ✅ Physics engine working")


def test_environment():
    """Test GPU environment"""
    print("\nTesting GPU Environment...")

    num_envs = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create mixed level configs
    level_configs = []
    level_configs.extend([LevelLibrary.get_tutorial_level() for _ in range(5)])
    level_configs.extend([Map1Library.get_tower_level() for _ in range(6)])
    level_configs.extend([Map2Library.get_map2_level() for _ in range(5)])

    env = TorchFireWaterEnv(
        num_envs=num_envs,
        level_configs=level_configs,
        device=device
    )

    print("  ✅ Environment created")

    # Test reset
    fire_obs, water_obs = env.reset()
    assert fire_obs.shape == (num_envs, 52)
    assert water_obs.shape == (num_envs, 52)
    assert fire_obs.device.type == device.split(':')[0]

    print("  ✅ Reset working")

    # Test multiple steps
    for _ in range(10):
        fire_actions = torch.randint(0, 6, (num_envs,), device=device)
        water_actions = torch.randint(0, 6, (num_envs,), device=device)

        (fire_obs, water_obs), (fire_rewards, water_rewards), \
        (fire_dones, water_dones), infos = env.step(fire_actions, water_actions)

        assert fire_obs.shape == (num_envs, 52)
        assert fire_rewards.shape == (num_envs,)
        assert len(infos) == num_envs

    print("  ✅ Stepping working")

    # Test visualization export
    state = env.get_state_for_visualization(0)
    assert 'fire_pos' in state
    assert 'water_pos' in state
    assert 'solids' in state

    print("  ✅ Visualization export working")


def test_agent():
    """Test GPU DQN agent"""
    print("\nTesting GPU Agent...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = GPUDQNAgent(device=device)

    # Test forward pass
    states = torch.randn(16, 52, device=device)
    q_values = agent(states)
    assert q_values.shape == (16, 6)

    print("  ✅ Forward pass working")

    # Test action selection
    actions = agent.select_actions(states, epsilon=0.1)
    assert actions.shape == (16,)
    assert actions.device.type == device.split(':')[0]

    print("  ✅ Action selection working")


def test_replay_buffer():
    """Test GPU replay buffer"""
    print("\nTesting Replay Buffer...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    buffer = GPUReplayBuffer(
        capacity=1000,
        state_dim=52,
        device=device
    )

    # Add some experiences
    batch_size = 32
    for _ in range(5):
        states = torch.randn(batch_size, 52, device=device)
        actions = torch.randint(0, 6, (batch_size,), device=device)
        rewards = torch.randn(batch_size, device=device)
        next_states = torch.randn(batch_size, 52, device=device)
        dones = torch.zeros(batch_size, device=device)

        buffer.push(states, actions, rewards, next_states, dones)

    assert len(buffer) == batch_size * 5

    print("  ✅ Buffer push working")

    # Sample
    batch = buffer.sample(64)
    assert batch[0].shape == (64, 52)  # states
    assert batch[1].shape == (64,)      # actions

    print("  ✅ Buffer sampling working")


def test_training_loop():
    """Test mini training loop"""
    print("\nTesting Training Loop...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Small setup
    num_envs = 8
    level_configs = [LevelLibrary.get_tutorial_level() for _ in range(num_envs)]

    env = TorchFireWaterEnv(
        num_envs=num_envs,
        level_configs=level_configs,
        device=device
    )

    agent = GPUDQNAgent(device=device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=3e-4)
    buffer = GPUReplayBuffer(capacity=10000, state_dim=52, device=device)

    # Run mini episode
    fire_obs, water_obs = env.reset()

    for step in range(50):
        # Select actions
        fire_actions = agent.select_actions(fire_obs, epsilon=0.5)
        water_actions = agent.select_actions(water_obs, epsilon=0.5)

        # Step
        (fire_obs_next, water_obs_next), (fire_rewards, water_rewards), \
        (fire_dones, water_dones), infos = env.step(fire_actions, water_actions)

        # Store
        buffer.push(fire_obs, fire_actions, fire_rewards, fire_obs_next, fire_dones)

        # Train if enough samples
        if len(buffer) >= 32:
            states, actions, rewards, next_states, dones = buffer.sample(32)

            # Forward pass
            q_values = agent(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Target
            with torch.no_grad():
                next_q = agent(next_states).max(dim=1)[0]
                targets = rewards + 0.99 * (1 - dones) * next_q

            # Loss
            loss = torch.nn.functional.mse_loss(q_values, targets)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update
        fire_obs = fire_obs_next
        water_obs = water_obs_next

        if fire_dones.all():
            break

    print("  ✅ Training loop working")
    print(f"  Steps: {step + 1}, Buffer: {len(buffer)}")


def main():
    """Run all tests"""
    print("=" * 60)
    print("GPU Training System Tests")
    print("=" * 60)

    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU available, using CPU (will be slow)")

    print()

    try:
        test_physics_engine()
        test_environment()
        test_agent()
        test_replay_buffer()
        test_training_loop()

        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        print("\nGPU training system is ready to use!")
        print("\nNext steps:")
        print("  1. Open GPU_Training.ipynb in Jupyter")
        print("  2. Or run: python train_gpu.py")

    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ TEST FAILED!")
        print("=" * 60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
