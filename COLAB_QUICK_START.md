# Google Colab Quick Start

**Train your Fire & Water RL agents with free GPU acceleration!**

## 🚀 Quick Steps

1. **Open notebook**: Upload `FireWater_Training_Colab.ipynb` to [Google Colab](https://colab.research.google.com/)

2. **Enable GPU**: Runtime → Change runtime type → GPU → Save

3. **Run all cells**: Runtime → Run all

4. **Wait**: Training takes 4-6 hours for 2000 episodes

5. **Download**: Get `checkpoints.zip` from Files panel

6. **Visualize locally**:
   ```bash
   python visualize.py trained checkpoints/fire_final.pth checkpoints/water_final.pth --map tower
   ```

## ⚙️ Configuration

Edit cell 4 in the notebook:

```python
MAP_NAME = "tower"          # or "tutorial"
NUM_EPISODES = 2000         # More = better trained
MAX_STEPS = 3000            # Max steps per episode
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
USE_WANDB = False           # Set True for tracking
RESUME_EPISODE = None       # or episode number
```

## 📊 Expected Results

| Episodes | Tutorial Success | Tower Success | Time (GPU) |
|----------|-----------------|---------------|------------|
| 500 | 20-30% | 10-20% | 1-2 hrs |
| 1000 | 40-50% | 20-30% | 2-3 hrs |
| 2000 | 50-70% | 30-50% | 4-6 hrs |
| 5000 | 60-80% | 40-60% | 10-12 hrs |

## 🔧 Common Issues

### No GPU?
Runtime → Change runtime type → GPU → Save → Restart runtime

### Out of memory?
Set `BATCH_SIZE = 32` in cell 4

### Disconnected?
Find last checkpoint: `!ls checkpoints/ | tail -5`
Set `RESUME_EPISODE = 1500` (or your last episode)

### Can't download?
Left sidebar → Files → Right-click `checkpoints.zip` → Download

## 📁 Files Created

After training, you'll have:
```
checkpoints/
├── fire_staged_dqn_ep100.pth
├── water_staged_dqn_ep100.pth
├── fire_staged_dqn_ep200.pth
├── water_staged_dqn_ep200.pth
├── ...
├── fire_final.pth
└── water_final.pth
```

## 🎮 Local Visualization

```bash
# Extract checkpoints
unzip checkpoints.zip

# Visualize
python visualize.py trained checkpoints/fire_final.pth checkpoints/water_final.pth --map tower

# Or specific episode
python visualize.py trained checkpoints/fire_staged_dqn_ep1500.pth checkpoints/water_staged_dqn_ep1500.pth --map tower
```

## 💡 Tips

- **Always enable GPU** (10x faster!)
- **Save frequently** (default: every 100 episodes)
- **Use WandB** to monitor training progress
- **Train longer** for better results (5000+ episodes)
- **Download checkpoints** after each session

## 📚 Full Documentation

See `COLAB_TRAINING_GUIDE.md` for:
- Detailed instructions
- Troubleshooting
- Advanced configuration
- Performance tuning
- WandB setup

---

**Need help?** Check the full guide or GitHub issues.
