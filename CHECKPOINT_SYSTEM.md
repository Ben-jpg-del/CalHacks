# Checkpoint System Explained

This document explains how episode weights are saved, stored, and loaded in both local training and Google Colab.

## How Checkpoints Are Saved

### 1. Automatic Saving During Training

The training script automatically saves checkpoints at regular intervals:

**Location in code:** [train_stage_milestone_dqn.py:389-395](train_stage_milestone_dqn.py#L389-L395)

```python
# Save checkpoints
if (episode + 1) % save_frequency == 0:
    checkpoint_start = time.time()
    fire_agent.save(f"checkpoints/fire_staged_dqn_ep{episode+1}.pth")
    water_agent.save(f"checkpoints/water_staged_dqn_ep{episode+1}.pth")
    checkpoint_time = time.time() - checkpoint_start
    print(f"\n*** CHECKPOINT SAVED @ Episode {episode + 1} ({checkpoint_time:.1f}s) ***\n")
```

**Default save frequency:** Every **100 episodes**

**What gets saved:**
- Episode 100, 200, 300, 400, ..., up to final episode
- Both fire and water agents saved separately

### 2. What's Inside a Checkpoint File

**Location in code:** [example_dqn.py:193-203](example_dqn.py#L193-L203)

Each `.pth` file contains:

```python
{
    'policy_net': self.policy_net.state_dict(),      # Neural network weights (main)
    'target_net': self.target_net.state_dict(),      # Target network weights
    'optimizer': self.optimizer.state_dict(),        # Optimizer state (momentum, etc)
    'epsilon': self.epsilon,                         # Exploration rate
    'steps': self.steps,                             # Total steps taken
    'updates': self.updates                          # Number of gradient updates
}
```

**Explanation:**
- **policy_net**: The actual learned Q-function weights - this is what the agent uses to make decisions
- **target_net**: Stabilizes training by providing consistent Q-value targets
- **optimizer**: Remembers Adam optimizer momentum for smooth learning
- **epsilon**: Current exploration rate (decreases over time)
- **steps/updates**: Training progress counters

### 3. File Structure

After training, you'll have:

```
checkpoints/
├── fire_staged_dqn_ep100.pth       # Fire agent at episode 100
├── water_staged_dqn_ep100.pth      # Water agent at episode 100
├── fire_staged_dqn_ep200.pth       # Fire agent at episode 200
├── water_staged_dqn_ep200.pth      # Water agent at episode 200
├── fire_staged_dqn_ep300.pth
├── water_staged_dqn_ep300.pth
├── ...
├── fire_staged_dqn_ep2000.pth      # Last checkpoint before final
├── water_staged_dqn_ep2000.pth
├── fire_final.pth                  # Final trained agent
└── water_final.pth                 # Final trained agent
```

**File size:** Each checkpoint is typically **2-5 MB** (depends on network size)

## How Checkpoints Are Loaded

### Loading for Resume Training

**Location in code:** [train_stage_milestone_dqn.py:175-196](train_stage_milestone_dqn.py#L175-L196)

```python
if resume_episode:
    checkpoint_path_fire = f"checkpoints/fire_staged_dqn_ep{resume_episode}.pth"
    checkpoint_path_water = f"checkpoints/water_staged_dqn_ep{resume_episode}.pth"

    if os.path.exists(checkpoint_path_fire) and os.path.exists(checkpoint_path_water):
        print(f"Loading checkpoints from episode {resume_episode}...")
        fire_agent.load(checkpoint_path_fire)
        water_agent.load(checkpoint_path_water)
        print(f"Successfully resumed from episode {resume_episode}")
        start_episode = resume_episode
    else:
        print(f"Warning: Checkpoint not found, starting from scratch")
        start_episode = 0
```

### Loading for Visualization

**Location in code:** [visualize.py:417-420](visualize.py#L417-L420)

```python
fire_agent.load(fire_model_path)
water_agent.load(water_model_path)

# Set to evaluation mode
fire_agent.epsilon = 0.0
water_agent.epsilon = 0.0
```

## Google Colab Workflow

### 1. Training on Colab (Session 1)

```python
# In the notebook
MAP_NAME = "tower"
NUM_EPISODES = 2000
RESUME_EPISODE = None  # Fresh start

# Run training cell
train_dqn_with_staged_rewards(...)

# Checkpoints saved to: /content/CalHacks/checkpoints/
# ├── fire_staged_dqn_ep100.pth
# ├── water_staged_dqn_ep100.pth
# ├── fire_staged_dqn_ep200.pth
# ├── ...
# ├── fire_staged_dqn_ep2000.pth
# └── water_staged_dqn_ep2000.pth
```

### 2. Download Checkpoints

```python
# Zip all checkpoints
!zip -r checkpoints.zip checkpoints/

# Download via Files panel or:
from google.colab import files
files.download('checkpoints.zip')
```

**What gets downloaded:**
- Single `checkpoints.zip` file (~50-500 MB depending on number of checkpoints)
- Contains all episode checkpoints from the session

### 3. Resume Training (Session 2)

**Upload checkpoints back to Colab:**

```python
# Cell 3 in notebook
from google.colab import files
uploaded = files.upload()  # Upload checkpoints.zip

# Extract
!unzip -q checkpoints.zip
# Now checkpoints/ directory is restored
```

**Configure resume:**

```python
# Cell 5 in notebook
RESUME_EPISODE = 2000  # Continue from where you left off
NUM_EPISODES = 4000    # Train up to episode 4000

# Training will:
# - Load fire_staged_dqn_ep2000.pth
# - Load water_staged_dqn_ep2000.pth
# - Continue training from episode 2001 → 4000
# - Save new checkpoints: ep2100, ep2200, ..., ep4000
```

## Checkpoint Management

### Disk Space

**Per checkpoint pair (fire + water):** ~4-10 MB

**Example calculations:**
- 100 episodes of training = 1 checkpoint pair = ~5 MB
- 2000 episodes = 20 checkpoint pairs = ~100 MB
- 5000 episodes = 50 checkpoint pairs = ~250 MB

### Checkpoint Frequency

You can adjust save frequency in the configuration:

```python
SAVE_FREQUENCY = 100  # Save every 100 episodes (default)
SAVE_FREQUENCY = 50   # Save every 50 episodes (more checkpoints)
SAVE_FREQUENCY = 200  # Save every 200 episodes (fewer checkpoints)
```

**Recommendations:**
- **More frequent** (50-100): Better for long training runs, allows finer resume points
- **Less frequent** (200-500): Saves disk space, good for stable training
- **For Colab**: Keep at 100 to have recovery points if disconnected

### Cleaning Up Old Checkpoints

If you want to keep only certain checkpoints:

```bash
# Keep only every 500 episodes
cd checkpoints
rm *ep100.pth *ep200.pth *ep300.pth  # etc.
# Keep: ep500, ep1000, ep1500, final

# Or keep only the last 5 checkpoints
ls -t *.pth | tail -n +11 | xargs rm
```

## Advanced: What's Saved in Memory

### Network Architecture

The checkpoint saves **weights**, not the architecture. The network structure is defined in [example_dqn.py:18-52](example_dqn.py#L18-L52):

```python
class DQNetwork(nn.Module):
    def __init__(self, state_dim=52, action_dim=6, hidden_dim=256):
        # Network layers defined here
        # When loading, must match this structure!
```

### Optimizer State

The Adam optimizer remembers momentum for each parameter:

```python
# Inside checkpoint
'optimizer': {
    'state': {
        0: {
            'step': tensor([...]),
            'exp_avg': tensor([...]),      # First moment estimate
            'exp_avg_sq': tensor([...])    # Second moment estimate
        },
        # ... for each parameter
    },
    'param_groups': [...]
}
```

This allows training to continue smoothly without resetting momentum.

## Troubleshooting

### "Checkpoint not found" Error

**Problem:** Resume episode doesn't exist

**Solution:**
```python
# Check what's available
!ls checkpoints/*ep*.pth

# Find latest episode
!ls checkpoints/ | grep "ep" | tail -2

# Set RESUME_EPISODE to an existing episode
```

### "Size mismatch" Error

**Problem:** Network architecture changed

**Solution:**
- Use checkpoints only with same network architecture
- If you changed `hidden_dim`, you can't load old checkpoints
- Start fresh training or retrain

### Corrupted Checkpoint

**Problem:** Download interrupted or file corrupted

**Solution:**
```python
# Test if checkpoint loads
import torch
try:
    checkpoint = torch.load('checkpoints/fire_staged_dqn_ep1000.pth')
    print("✅ Checkpoint OK")
except Exception as e:
    print(f"❌ Corrupted: {e}")
    # Re-download or use earlier checkpoint
```

### Missing Checkpoints After Colab Disconnect

**Problem:** Colab disconnected before downloading

**Solution:**
- Checkpoints are lost when runtime disconnects
- Always download `checkpoints.zip` regularly during training
- Consider using Google Drive sync:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy checkpoints periodically
!cp -r checkpoints /content/drive/MyDrive/FireWater_Backups/
```

## Best Practices

### For Local Training

1. **Keep save frequency at 100** for good balance
2. **Back up checkpoints** to external drive or cloud
3. **Test loading** before long training runs
4. **Keep final checkpoint** always - it's your best model

### For Colab Training

1. **Download checkpoints** after each session
2. **Use Google Drive** for automatic backup
3. **Save checkpoint.zip** with descriptive names:
   - `checkpoints_ep2000_tower_map.zip`
   - `checkpoints_ep4000_tutorial_map.zip`

4. **Monitor disk usage:**
```python
# Check space
!df -h /content
```

5. **Zip before disconnect:**
```python
# Set up automatic backup every 500 episodes
if (episode + 1) % 500 == 0:
    !zip -r checkpoints_backup_ep{episode+1}.zip checkpoints/
    # Then download or save to Drive
```

## Summary

**What gets saved:**
- Neural network weights (policy and target networks)
- Optimizer state (for smooth training continuation)
- Training progress (epsilon, steps, updates)

**When it's saved:**
- Every N episodes (default: 100)
- At the end of training (final.pth)

**Where it's saved:**
- Locally: `./checkpoints/`
- Colab: `/content/CalHacks/checkpoints/`

**How to use:**
- Resume training: Set `RESUME_EPISODE`
- Visualize: Pass checkpoint paths to visualize.py
- Transfer: Download checkpoint.zip and upload to new session

The checkpoint system ensures you never lose training progress and can always continue or evaluate your agents! 💾
