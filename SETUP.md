# Fire & Water RL Package - Setup Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

Or install minimal requirements:
```bash
pip install numpy pygame
```

### 2. Test Installation
```bash
# Test environment
python game_environment.py

# Benchmark speed
python train_fast.py benchmark

# Human play mode
python visualize.py human
```

## Usage Examples

### Fast Training (No Visualization)
```bash
python train_fast.py
```

### Train with DQN Example
```bash
python example_dqn.py
```

### Visualize Random Agents
```bash
python visualize.py random 10
```

### Human Play
```bash
python visualize.py human
```

## File Overview

| File | Purpose | Dependencies |
|------|---------|--------------|
| `physics_engine.py` | Pure Python physics | numpy |
| `map_config.py` | Level definitions | physics_engine |
| `game_environment.py` | Gym-like environment | physics_engine, map_config |
| `train_fast.py` | Headless training | game_environment |
| `visualize.py` | Pygame rendering | game_environment, pygame |
| `example_dqn.py` | DQN implementation | game_environment, torch |

## Performance Tips

1. **Disable visualization for training**
   - Use `train_fast.py` instead of rendering
   - Expected: 10,000+ steps/second

2. **Use GPU if available**
   - PyTorch will automatically use CUDA
   - Check with: `torch.cuda.is_available()`

3. **Batch updates**
   - Process multiple environments in parallel
   - Use vectorized operations

4. **Profile your code**
   ```python
   import cProfile
   cProfile.run('train_headless(num_episodes=100)')
   ```

## Troubleshooting

### "No module named 'pygame'"
```bash
pip install pygame
```

### "No module named 'torch'"
PyTorch is optional. Install with:
```bash
pip install torch
```

### Slow training
- Make sure you're using `train_fast.py` (not visualize.py)
- Disable W&B logging if not needed
- Reduce max_steps_per_episode

### "CUDA out of memory"
- Reduce batch size
- Reduce network size
- Use CPU instead: `device='cpu'`

## Next Steps

1. Read the full README.md
2. Implement your RL algorithm (see example_dqn.py)
3. Customize rewards in game_environment.py
4. Create custom levels in map_config.py
5. Train and visualize!

## Support

For issues or questions:
- Check README.md for detailed documentation
- Review example_dqn.py for implementation reference
- Test components individually to isolate issues
