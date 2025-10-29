"""
Debug script to diagnose GPU training issues
Run this to see exactly where the problem occurs
"""

import torch
import sys
import os

print("="*60)
print("GPU TRAINING DEBUG SCRIPT")
print("="*60)

# Step 1: Check CUDA
print("\n1. Checking CUDA...")
print(f"   PyTorch version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
else:
    print("   ❌ NO CUDA! Training will fail!")
    sys.exit(1)

# Step 2: Check paths
print("\n2. Checking paths...")
sys.path.insert(0, '/content/CalHacks/gpu_training')
sys.path.insert(0, '/content/CalHacks')
print(f"   Python path: {sys.path[:3]}")

# Step 3: Import modules
print("\n3. Importing modules...")
try:
    from map_config import LevelLibrary
    print("   ✅ map_config imported")
except Exception as e:
    print(f"   ❌ map_config import failed: {e}")
    sys.exit(1)

try:
    from torch_env import TorchFireWaterEnv
    print("   ✅ torch_env imported")
except Exception as e:
    print(f"   ❌ torch_env import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from train_gpu import GPUDQNAgent, GPUReplayBuffer
    print("   ✅ train_gpu imported")
except Exception as e:
    print(f"   ❌ train_gpu import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Create small environment
print("\n4. Creating small test environment (64 envs)...")
try:
    configs = [LevelLibrary.get_tutorial_level() for _ in range(64)]
    env = TorchFireWaterEnv(
        num_envs=64,
        level_configs=configs,
        device='cuda'
    )
    print(f"   ✅ Environment created")
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    print(f"   ❌ Environment creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test reset
print("\n5. Testing reset...")
try:
    fire_obs, water_obs = env.reset()
    print(f"   ✅ Reset successful")
    print(f"   Fire obs shape: {fire_obs.shape}, device: {fire_obs.device}")
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    print(f"   ❌ Reset failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Test step
print("\n6. Testing environment step...")
try:
    fire_actions = torch.randint(0, 6, (64,), device='cuda')
    water_actions = torch.randint(0, 6, (64,), device='cuda')

    (fire_obs_next, water_obs_next), (fire_rewards, water_rewards), \
    (fire_dones, water_dones), infos = env.step(fire_actions, water_actions)

    print(f"   ✅ Step successful")
    print(f"   Rewards: fire={fire_rewards[0]:.2f}, water={water_rewards[0]:.2f}")
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    print(f"   ❌ Step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 7: Create agents
print("\n7. Creating agents...")
try:
    fire_agent = GPUDQNAgent(device='cuda')
    water_agent = GPUDQNAgent(device='cuda')
    print(f"   ✅ Agents created")
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    print(f"   ❌ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 8: Create replay buffer
print("\n8. Creating replay buffer...")
try:
    buffer = GPUReplayBuffer(
        capacity=10000,
        state_dim=52,
        device='cuda'
    )
    print(f"   ✅ Buffer created")
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    print(f"   ❌ Buffer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 9: Test full training step
print("\n9. Testing full training step...")
try:
    # Get actions
    fire_actions = fire_agent.select_actions(fire_obs, epsilon=0.5)
    water_actions = water_agent.select_actions(water_obs, epsilon=0.5)

    # Step
    (fire_obs_next, water_obs_next), (fire_rewards, water_rewards), \
    (fire_dones, water_dones), infos = env.step(fire_actions, water_actions)

    # Store in buffer
    buffer.push(fire_obs, fire_actions, fire_rewards, fire_obs_next, fire_dones)

    print(f"   ✅ Training step successful")
    print(f"   Buffer size: {len(buffer)}")
    print(f"   GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    print(f"   ❌ Training step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 10: Estimate memory for full training
print("\n10. Memory estimate for full training...")
num_envs = 1024
buffer_capacity = 1000000

env_mem = torch.cuda.memory_allocated() / 1e9 * (num_envs / 64)
buffer_mem = buffer_capacity * 52 * 4 / 1e9

print(f"   Current memory (64 envs): {torch.cuda.memory_allocated()/1e9:.2f} GB")
print(f"   Estimated for {num_envs} envs: {env_mem:.2f} GB")
print(f"   Estimated buffer: {buffer_mem:.2f} GB")
print(f"   Total estimate: {env_mem + buffer_mem:.2f} GB")

# Get GPU total memory
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"   GPU total memory: {total_mem:.1f} GB")

if env_mem + buffer_mem > total_mem * 0.9:
    print(f"   ⚠️  WARNING: Estimated memory exceeds GPU capacity!")
    print(f"   Recommendation: Reduce NUM_ENVS to {int(num_envs * 0.5)}")
else:
    print(f"   ✅ Should fit in GPU memory")

print("\n" + "="*60)
print("ALL TESTS PASSED! ✅")
print("GPU training should work!")
print("="*60)
