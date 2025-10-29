# Google Colab Training Guide

This guide explains how to use the Jupyter notebook to train your Fire & Water RL agents on Google Colab with free GPU acceleration.

## Quick Start

1. **Open the notebook in Colab:**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click File → Upload notebook
   - Upload `FireWater_Training_Colab.ipynb`

   **OR**

   - Click this link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ben-jpg-del/CalHacks/blob/custom-map-selection/FireWater_Training_Colab.ipynb)

2. **Enable GPU:**
   - Runtime → Change runtime type
   - Hardware accelerator → **GPU**
   - Click Save

3. **Run all cells** in order (Runtime → Run all)

4. **Wait for training** to complete (can take several hours)

5. **Download checkpoints** when done

## What the Notebook Does

### Setup (Cells 1-3)
- Clones your GitHub repository
- Installs PyTorch, NumPy, Pygame, WandB
- Verifies GPU is available

### Configuration (Cell 4)
- Set training parameters:
  - `MAP_NAME`: Choose "tutorial" or "tower"
  - `NUM_EPISODES`: How many episodes to train (2000 recommended)
  - `MAX_STEPS`: Max steps per episode (3000)
  - `LEARNING_RATE`: 3e-4 works well
  - `BATCH_SIZE`: 64 (reduce to 32 if out of memory)
  - `USE_WANDB`: Enable experiment tracking
  - `RESUME_EPISODE`: Continue from checkpoint

### Training (Cell 6)
- Trains both fire and water agents
- Uses DQN with staged milestone rewards
- Saves checkpoints every 100 episodes
- Shows progress updates every 10 episodes

### Evaluation (Cell 10)
- Tests trained agents
- Runs 100 evaluation episodes
- Reports success rate and average reward

### Download (Cell 9)
- Zips all checkpoints
- Downloads to your computer
- Use locally for visualization

## Training Parameters

### Recommended Settings

**Quick Test (30 minutes):**
```python
NUM_EPISODES = 200
MAX_STEPS = 2000
```

**Short Training (2-3 hours):**
```python
NUM_EPISODES = 1000
MAX_STEPS = 3000
```

**Full Training (6-8 hours):**
```python
NUM_EPISODES = 5000
MAX_STEPS = 3000
```

**Long Training (12+ hours):**
```python
NUM_EPISODES = 10000
MAX_STEPS = 3000
```

### GPU vs CPU Performance

| Device | Speed | Recommended |
|--------|-------|-------------|
| GPU (T4) | ~2-3 episodes/sec | ✅ Yes |
| CPU | ~0.2-0.3 episodes/sec | ❌ Too slow |

**Always enable GPU for practical training times!**

## Using Weights & Biases

WandB provides real-time training visualization:

1. **Set `USE_WANDB = True`** in configuration cell

2. **Run WandB login cell:**
   ```python
   import wandb
   wandb.login()
   ```

3. **Get API key:**
   - Go to https://wandb.ai/
   - Create free account
   - Copy API key
   - Paste into Colab prompt

4. **View training in real-time:**
   - Click the WandB URL in the output
   - Watch loss, rewards, success rate graphs

## Resuming Training

If Colab disconnects or you want to continue training:

1. **Find last checkpoint:**
   ```python
   !ls checkpoints/*ep*.pth | tail -2
   ```

2. **Set resume episode:**
   ```python
   RESUME_EPISODE = 1500  # Use your last saved episode
   ```

3. **Run training cell again** - it will load and continue

## Downloading Trained Models

### Method 1: Zip Download (Recommended)
```python
# Cell 9 - creates checkpoints.zip
!zip -r checkpoints.zip checkpoints/
```
Download from Files panel (left sidebar)

### Method 2: Direct Download
```python
from google.colab import files
files.download('checkpoints/fire_final.pth')
files.download('checkpoints/water_final.pth')
```

### Method 3: Google Drive Sync
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r checkpoints /content/drive/MyDrive/FireWater_Checkpoints
```

## Visualizing Results Locally

After downloading checkpoints:

1. **Extract to your repository:**
   ```bash
   unzip checkpoints.zip -d /path/to/CalHacks/
   ```

2. **Run visualization:**
   ```bash
   cd /path/to/CalHacks
   python visualize.py trained checkpoints/fire_final.pth checkpoints/water_final.pth --map tower
   ```

3. **Watch your trained agents play!**

## Troubleshooting

### ❌ "No GPU detected"

**Solution:**
- Runtime → Change runtime type → GPU
- Click Save
- Runtime → Restart runtime
- Re-run all cells

### ❌ "Out of memory"

**Solution 1** - Reduce batch size:
```python
BATCH_SIZE = 32  # or even 16
```

**Solution 2** - Reduce max steps:
```python
MAX_STEPS = 2000
```

### ❌ "Colab disconnected"

**Solution:**
- This happens after ~12 hours on free tier
- Check last saved checkpoint: `!ls checkpoints/ | tail -5`
- Resume training with `RESUME_EPISODE`
- Consider Colab Pro for longer sessions

### ❌ "Can't download files"

**Solution 1** - Use Files panel:
- Click folder icon on left sidebar
- Navigate to `checkpoints.zip`
- Right-click → Download

**Solution 2** - Use Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
!cp checkpoints.zip /content/drive/MyDrive/
```

### ❌ "Training is very slow"

**Check:**
1. GPU is enabled (cell 2 should show GPU info)
2. Not using CPU by mistake
3. `MAX_STEPS` isn't too high
4. Batch size isn't too small

**Expected speeds:**
- GPU: 2-3 episodes/second
- CPU: 0.2-0.3 episodes/second (10x slower!)

## Performance Expectations

### Tutorial Map

| Episodes | Expected Success Rate | Training Time (GPU) |
|----------|----------------------|---------------------|
| 500 | 20-30% | 1-1.5 hours |
| 1000 | 40-50% | 2-3 hours |
| 2000 | 50-70% | 4-6 hours |
| 5000 | 60-80% | 10-12 hours |

### Tower Map (Custom)

| Episodes | Expected Success Rate | Training Time (GPU) |
|----------|----------------------|---------------------|
| 500 | 10-20% | 1-1.5 hours |
| 1000 | 20-30% | 2-3 hours |
| 2000 | 30-50% | 4-6 hours |
| 5000 | 40-60% | 10-12 hours |

Tower map is harder due to more hazards and vertical navigation.

## Tips for Better Results

### 1. Train Longer
More episodes = better performance. 5000+ episodes recommended for good results.

### 2. Use Curriculum Learning
Start with tutorial map, then transfer to tower:
```python
# First training
MAP_NAME = "tutorial"
NUM_EPISODES = 2000

# Then resume on harder map
MAP_NAME = "tower"
RESUME_EPISODE = 2000
```

### 3. Tune Hyperparameters
Edit reward values in your local `reward_staged_milestone.py`:
- Increase `beta` for faster switch activation
- Increase `gamma` for stronger goal focus
- Adjust `step_penalty` to control speed

### 4. Monitor Training
Enable WandB to watch:
- Success rate over time
- Average reward trends
- Episode length changes
- Loss convergence

### 5. Save Often
Keep `SAVE_FREQUENCY = 100` to have recovery points.

## Colab Limits

### Free Tier
- **GPU Time**: ~12 hours per session
- **RAM**: 12-13 GB
- **Disk**: 100+ GB (plenty for checkpoints)
- **Sessions**: Can disconnect after inactivity

### Colab Pro ($10/month)
- **GPU Time**: ~24 hours per session
- **RAM**: Up to 25 GB
- **Better GPUs**: Faster training
- **Priority**: Less likely to disconnect

### Strategies for Free Tier

**For 12-hour limit:**
1. Train with `NUM_EPISODES = 5000`
2. Set `SAVE_FREQUENCY = 100`
3. If disconnected, resume from last checkpoint
4. Repeat until satisfied with performance

**For multiple sessions:**
```python
# Session 1
NUM_EPISODES = 2000
RESUME_EPISODE = None

# Session 2 (after downloading and re-uploading checkpoints)
NUM_EPISODES = 4000
RESUME_EPISODE = 2000

# Session 3
NUM_EPISODES = 6000
RESUME_EPISODE = 4000
```

## Advanced: Custom Configuration

### Modify Reward Function

Edit the notebook cell 6 to customize rewards before training:

```python
# Add this before training starts
from reward_staged_milestone import StagedMilestoneRewardConfig

reward_config = StagedMilestoneRewardConfig()
reward_config.beta = 100.0      # Higher reward for switches
reward_config.gamma = 300.0     # Higher reward for exits
reward_config.step_penalty = -0.05  # Lower penalty

# Then pass to training...
```

### Change Network Architecture

Edit `example_dqn.py` locally, commit, and re-clone in Colab:

```python
# In example_dqn.py, modify DQNetwork:
def __init__(self, state_dim=52, action_dim=6, hidden_dim=512):  # Bigger network
    # ... rest of code
```

## FAQs

**Q: How long does training take?**
A: 2000 episodes takes ~4-6 hours on GPU, ~40-60 hours on CPU.

**Q: Can I train overnight?**
A: Yes, but Colab may disconnect. Use checkpoints to resume.

**Q: Is the free tier enough?**
A: Yes! You can train excellent agents on free tier with patience.

**Q: Can I visualize on Colab?**
A: No, Pygame doesn't work on Colab. Download checkpoints and visualize locally.

**Q: What if I lose my checkpoints?**
A: Always download after each session. Consider syncing to Google Drive.

**Q: How do I know if training is working?**
A: Watch for increasing success rate and decreasing loss. Enable WandB for graphs.

## Support

For issues:
1. Check Troubleshooting section above
2. Review notebook cell outputs for errors
3. Check GitHub Issues: https://github.com/Ben-jpg-del/CalHacks/issues
4. Ensure GPU is enabled (most common issue!)

---

**Happy Training! 🔥💧**
