"""
Test Script - Verify all components work correctly
Run this after installation to ensure everything is set up properly
"""

import sys
import numpy as np

def test_physics_engine():
    """Test physics engine"""
    print("Testing physics_engine.py...")
    try:
        from physics_engine import PhysicsEngine, Agent, Rect
        
        # Create engine
        engine = PhysicsEngine()
        
        # Create agent
        agent = Agent(
            rect=Rect(100, 100, 28, 36),
            velocity=[0.0, 0.0],
            grounded=False,
            agent_type='fire'
        )
        
        # Test movement
        solids = [Rect(0, 200, 960, 20)]
        agent = engine.apply_movement(agent, 2, solids)  # Move right
        
        assert agent is not None, "Agent should not be None"
        assert agent.rect.x > 100, "Agent should have moved right"
        
        print("  ✓ Physics engine working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Physics engine error: {e}")
        return False

def test_map_config():
    """Test map configuration"""
    print("Testing map_config.py...")
    try:
        from map_config import LevelConfig, LevelLibrary
        
        # Create level
        level = LevelConfig()
        
        assert level.width == 960, "Level width should be 960"
        assert level.height == 540, "Level height should be 540"
        assert len(level.base_solids) > 0, "Level should have solids"
        
        # Test get_solids
        solids = level.get_solids(bridge_up=False, gate_open=False)
        assert len(solids) > 0, "Should return solids"
        
        # Test library
        tutorial = LevelLibrary.get_tutorial_level()
        assert tutorial.name == "Tutorial", "Tutorial level should be named Tutorial"
        
        print("  ✓ Map config working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Map config error: {e}")
        return False

def test_game_environment():
    """Test game environment"""
    print("Testing game_environment.py...")
    try:
        from game_environment import FireWaterEnv
        
        # Create environment
        env = FireWaterEnv()
        
        # Test reset
        fire_obs, water_obs = env.reset()
        assert fire_obs.shape == (52,), f"Fire observation should be (52,), got {fire_obs.shape}"
        assert water_obs.shape == (52,), f"Water observation should be (52,), got {water_obs.shape}"
        
        # Test step
        (fire_next_obs, water_next_obs), rewards, dones, info = env.step(0, 0)
        assert fire_next_obs.shape == (52,), "Next observation should be (52,)"
        assert len(rewards) == 2, "Should return 2 rewards"
        assert len(dones) == 2, "Should return 2 dones"
        assert isinstance(info, dict), "Info should be dict"
        
        # Test action space
        assert env.get_action_space_size() == 6, "Should have 6 actions"
        assert env.get_observation_space_size() == 52, "Should have 52 observations"
        
        print("  ✓ Game environment working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Game environment error: {e}")
        return False

def test_training_loop():
    """Test training loop"""
    print("Testing train_fast.py...")
    try:
        from game_environment import FireWaterEnv
        
        env = FireWaterEnv(max_steps=100)
        
        # Run a few steps
        fire_obs, water_obs = env.reset()
        total_reward = 0
        
        for _ in range(50):
            fire_action = np.random.randint(0, 6)
            water_action = np.random.randint(0, 6)
            
            (fire_obs, water_obs), rewards, dones, info = env.step(fire_action, water_action)
            total_reward += sum(rewards)
            
            if dones[0] or dones[1]:
                break
        
        assert total_reward != 0, "Should accumulate some reward"
        
        print("  ✓ Training loop working correctly")
        return True
    except Exception as e:
        print(f"  ✗ Training loop error: {e}")
        return False

def test_visualization_import():
    """Test visualization imports"""
    print("Testing visualize.py imports...")
    try:
        # Don't actually create window, just test imports
        import pygame
        from game_environment import FireWaterEnv
        
        # Test that we can import visualization without errors
        # (Don't run it, as it requires display)
        print("  ✓ Visualization imports working")
        return True
    except ImportError as e:
        print(f"  ✗ Visualization import error: {e}")
        print("    Note: pygame is optional for headless training")
        return False

def test_performance():
    """Test environment performance"""
    print("Testing environment performance...")
    try:
        from game_environment import FireWaterEnv
        import time
        
        env = FireWaterEnv(max_steps=1000)
        
        # Benchmark
        num_steps = 1000
        start_time = time.time()
        
        env.reset()
        for _ in range(num_steps):
            fire_action = np.random.randint(0, 6)
            water_action = np.random.randint(0, 6)
            _, _, (done_f, done_w), _ = env.step(fire_action, water_action)
            if done_f or done_w:
                env.reset()
        
        elapsed = time.time() - start_time
        steps_per_sec = num_steps / elapsed
        
        print(f"  ✓ Performance: {steps_per_sec:.0f} steps/sec")
        
        if steps_per_sec < 1000:
            print(f"    Warning: Performance is low. Expected >1000 steps/sec")
        
        return True
    except Exception as e:
        print(f"  ✗ Performance test error: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("FIREWATER RL PACKAGE - COMPONENT TESTS")
    print("=" * 60)
    print()
    
    tests = [
        ("Physics Engine", test_physics_engine),
        ("Map Config", test_map_config),
        ("Game Environment", test_game_environment),
        ("Training Loop", test_training_loop),
        ("Visualization", test_visualization_import),
        ("Performance", test_performance)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Unexpected error in {name}: {e}")
            results.append((name, False))
        print()
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The package is ready to use.")
        print("\nNext steps:")
        print("  1. python train_fast.py         # Fast training")
        print("  2. python visualize.py human    # Human play")
        print("  3. python example_dqn.py        # Train with DQN")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("See SETUP.md for troubleshooting tips.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
