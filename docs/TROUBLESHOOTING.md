# Troubleshooting Guide

Solutions to common issues when using Dynamic Microgrid Resilience v7.0

---

## Table of Contents

- [Installation Issues](#installation-issues)
- [Training Problems](#training-problems)
- [Performance Issues](#performance-issues)
- [Runtime Errors](#runtime-errors)
- [Visualization Problems](#visualization-problems)
- [System-Specific Issues](#system-specific-issues)
- [FAQ](#faq)

---

## Installation Issues

### ImportError: No module named 'torch'

**Problem**: PyTorch not installed

**Solution**:
```bash
pip install torch torchvision
```

For specific versions:
```bash
# CPU only
pip install torch==2.0.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

# CUDA 11.7
pip install torch==2.0.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

**Verify installation**:
```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
```

---

### ModuleNotFoundError: No module named 'matplotlib'

**Problem**: Missing visualization dependencies

**Solution**:
```bash
pip install matplotlib seaborn tqdm pandas numpy
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

---

### Python version incompatibility

**Problem**: `SyntaxError` or `TypeError` with older Python

**Symptoms**:
```
SyntaxError: invalid syntax
  config: MicrogridConfig = field(default_factory=MicrogridConfig)
```

**Solution**: Upgrade to Python 3.8+
```bash
python --version  # Check current version
```

Use conda or pyenv to install Python 3.8+:
```bash
# Using conda
conda create -n microgrid python=3.8
conda activate microgrid

# Or download from python.org
```

---

## Training Problems

### Training Not Converging

**Symptoms**:
- Load met rate stuck at 85-90%
- No improvement after 500 episodes
- Flat reward curve

**Diagnosis**:
```python
# Check training metrics
import pandas as pd
df = pd.read_csv('results/latest_training_progress.csv')
print(df[['episode', 'load_met_rate', 'unmet_energy']].tail(20))
```

**Solutions**:

#### 1. Train Longer
```bash
python cli.py train --episodes 2000
```

#### 2. Adjust Learning Rate
```python
# In Dynamic_Microgrid_Resilience_v7.py
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0003)  # from 0.0002
```

#### 3. Check Reward Function
```python
# Verify rewards are being computed correctly
print(f"Base reward: {base_reward}")
print(f"Pre-evening reward: {pre_evening_reward}")
print(f"Evening reward: {evening_reward}")
```

#### 4. Increase Evening Penalty
```python
# In reward calculation
if hour in {18, 19, 20, 21, 22}:
    if not load_met or soc_pct < 70:
        reward -= 500  # from 400
```

---

### Loss Exploding or NaN

**Symptoms**:
```
Episode 150: Loss = 45.2
Episode 151: Loss = 127.8
Episode 152: Loss = nan
```

**Solutions**:

#### 1. Reduce Learning Rate
```python
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)  # from 0.0002
```

#### 2. Gradient Clipping
```python
# In train_step() method
loss.backward()
torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
self.optimizer.step()
```

#### 3. Check for Invalid States
```python
# Add validation in get_enhanced_state()
assert not torch.isnan(state).any(), f"NaN in state: {state}"
assert torch.isfinite(state).all(), f"Infinite in state: {state}"
```

#### 4. Restart from Checkpoint
```python
# Load last good checkpoint
checkpoint = torch.load('models/checkpoint_ep200.pt')
agent.policy_net.load_state_dict(checkpoint['policy_net'])
```

---

### Memory Issues During Training

**Symptoms**:
```
RuntimeError: CUDA out of memory
```
or
```
MemoryError: Unable to allocate array
```

**Solutions**:

#### 1. Force CPU Training
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

#### 2. Reduce Batch Size
```python
# In DQNAgent
self.batch_size = 32  # from 64
```

#### 3. Reduce Replay Buffer
```python
# In DQNAgent
self.memory = ReplayBuffer(capacity=5000)  # from 10000
```

#### 4. Clear Memory
```python
# After each episode
torch.cuda.empty_cache()  # If using GPU
import gc
gc.collect()
```

---

## Performance Issues

### Poor Evening Coverage (<95%)

**Symptoms**:
- Daytime: 100% coverage
- Evening (18-22h): 80-90% coverage
- Pre-evening SOC < 70%

**Diagnosis**:
```python
# Check hourly performance
df = pd.read_csv('results/latest_resilience_detailed.csv')
evening = df[df['hour'].isin([18, 19, 20, 21, 22])]
print(f"Evening coverage: {evening['load_met'].mean()*100:.1f}%")
print(f"Pre-evening SOC: {df[df['hour']==17]['soc_percent'].values[0]:.1f}%")
```

**Solutions**:

#### 1. Increase Evening Penalty
```python
EVENING_PENALTY = -500  # from -400
```

#### 2. Increase Pre-Evening Bonus
```python
PRE_EVENING_BONUS = 300  # from 200
PRE_EVENING_TARGET_SOC = 0.80  # from 0.75
```

#### 3. Verify Pre-Evening Flag
```python
# Check state feature 7
state = agent.get_enhanced_state(soc, 16, renewable, load)
print(f"Pre-evening flag (hour 16): {state[6].item()}")  # Should be 1.0
```

#### 4. Longer Training
```bash
# Pre-evening strategy takes time to learn
python cli.py train --episodes 1500
```

---

### Battery Not Charging During Day

**Symptoms**:
- Battery SOC stays low (30-50%) during sunny hours
- Agent prefers `hold` action
- Poor evening performance as a result

**Solutions**:

#### 1. Increase Daytime Charging Bonus
```python
DAYTIME_CHARGE_BONUS = 100  # from 50
```

#### 2. Penalize Low SOC During Daytime
```python
if hour in range(10, 15) and soc_pct < 60:
    reward -= 100
```

#### 3. Check Solar Generation
```python
# Verify PV is producing power
profile = generator.generate_episode_profile(0)
print(f"Daytime PV: {profile['pv_power'][12]} kW")  # Should be >3 kW
```

---

### Overcharging (Battery at 100% Too Often)

**Symptoms**:
- Battery at 100% by 10 AM
- Wasted excess generation
- Poor nighttime performance

**Solutions**:

#### 1. Reduce Daytime Charging Bonus
```python
DAYTIME_CHARGE_BONUS = 25  # from 50
```

#### 2. Add Overcharge Penalty
```python
if hour < 15 and soc_pct > 85:
    reward -= 50
```

#### 3. Target-Based Reward
```python
# Only reward if SOC below target
if hour in range(6, 15):
    target_soc = 0.75  # Target 75%
    if is_charging and soc_pct < target_soc * 100:
        reward += 50
```

---

## Runtime Errors

### FileNotFoundError: No such file or directory: 'models/'

**Problem**: Missing directories

**Solution**:
```bash
mkdir -p models plots results
```

Or run:
```python
import os
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)
```

---

### RuntimeError: Expected all tensors to be on the same device

**Problem**: CPU/GPU mismatch

**Solution 1**: Force CPU
```python
device = torch.device('cpu')
agent = DQNAgent(config, device=device)
```

**Solution 2**: Ensure consistency
```python
# Move all tensors to same device
state = state.to(agent.device)
```

---

### AssertionError: invalid SOC

**Problem**: Battery SOC out of bounds

**Diagnosis**:
```python
# Add debugging
print(f"Current energy: {battery.current_energy_kwh}")
print(f"Capacity: {battery.capacity_kwh}")
print(f"SOC: {battery.get_SOC_percent()}%")
```

**Solutions**:

#### 1. Check Battery Constraints
```python
# In BatterySystem
def charge(self, power_kw, duration_hours=1.0):
    energy = power_kw * duration_hours * self.efficiency
    self.current_energy_kwh = min(
        self.current_energy_kwh + energy,
        self.max_soc * self.capacity_kwh
    )
```

#### 2. Validate Actions
```python
# Before executing action
if action.startswith('charge'):
    available_capacity = battery.capacity_kwh - battery.current_energy_kwh
    power = min(power, available_capacity / duration_hours)
```

---

### ValueError: operands could not be broadcast together

**Problem**: Array shape mismatch

**Diagnosis**:
```python
print(f"State shape: {state.shape}")  # Should be [7]
print(f"Expected: torch.Size([7])")
```

**Solution**:
```python
# Ensure state has correct shape
state = agent.get_enhanced_state(soc, hour, renewable, load)
assert state.shape == (7,), f"Invalid state shape: {state.shape}"
```

---

## Visualization Problems

### Plots Not Generating

**Problem**: `--plot` flag not working

**Diagnosis**:
```bash
python cli.py train --episodes 100 --plot --verbose
```

**Solutions**:

#### 1. Check matplotlib backend
```python
import matplotlib
print(matplotlib.get_backend())

# Try different backend
matplotlib.use('Agg')  # Non-interactive
```

#### 2. Verify results exist
```bash
ls -la results/
```

#### 3. Manually generate plots
```python
from cli import TrainingVisualizer

visualizer = TrainingVisualizer()
visualizer.plot_training_progress('results/latest_training_progress.csv')
```

---

### Empty or Corrupt Plots

**Problem**: Plots generated but show no data

**Solutions**:

#### 1. Check CSV data
```python
import pandas as pd
df = pd.read_csv('results/latest_training_progress.csv')
print(df.head())
print(df.describe())
```

#### 2. Verify data range
```python
# Load met rate should be 0-100
assert df['load_met_rate'].between(0, 100).all()
```

#### 3. Regenerate with fresh data
```bash
rm results/*.csv plots/*.png
python cli.py train --episodes 500 --plot
```

---

### Real-time Monitor Not Updating

**Problem**: `--monitor` flag shows static output

**Solutions**:

#### 1. Check terminal support
```python
import sys
print(f"stdout.isatty(): {sys.stdout.isatty()}")
```

#### 2. Use alternative monitoring
```bash
# Watch CSV file instead
watch -n 1 tail -n 1 results/latest_training_progress.csv
```

#### 3. Disable monitor, use verbose
```bash
python cli.py train --episodes 500 --verbose
```

---

## System-Specific Issues

### Windows: ANSI Escape Codes Not Working

**Problem**: Monitor shows garbled text

**Solution**:
```python
# Disable color codes
import os
os.environ['NO_COLOR'] = '1'
```

Or use:
```bash
python cli.py train --episodes 500  # Don't use --monitor
```

---

### macOS: Slow Parallel Processing

**Problem**: `--parallel` flag not speeding up

**Cause**: macOS uses `spawn` instead of `fork`

**Solution**:
```python
# Use fewer workers
python cli.py compare --capacities 15 18 20 --parallel --workers 2
```

Or:
```bash
# Don't use parallel
python cli.py compare --capacities 15 18 20
```

---

### Linux: Permission Denied Writing to models/

**Problem**: Cannot save models

**Solution**:
```bash
chmod 755 models/ plots/ results/
```

Or:
```bash
sudo chown $USER:$USER models/ plots/ results/
```

---

## FAQ

### Q: How long should training take?

**A**: 
- CPU (laptop): 20-25 minutes for 1000 episodes
- CPU (desktop): 15-20 minutes
- GPU: 10-15 minutes

If much slower, check:
- Background processes
- Disk I/O (use SSD if possible)
- Memory swapping

---

### Q: Can I train on GPU?

**A**: Yes, if you have CUDA-capable GPU:

```python
# Auto-detect and use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
agent = DQNAgent(config, device=device)
```

**Speedup**: ~2x faster than CPU

---

### Q: How much disk space needed?

**A**:
- Model files: ~2-3 MB each
- CSV results: ~100 KB each
- Plots: ~2 MB each
- **Total**: ~50-100 MB for typical session

---

### Q: Can I pause and resume training?

**A**: Yes!

```python
# Save checkpoint
torch.save({
    'policy_net': agent.policy_net.state_dict(),
    'target_net': agent.target_net.state_dict(),
    'optimizer': agent.optimizer.state_dict(),
    'episode': episode,
    'epsilon': agent.epsilon
}, 'checkpoint.pt')

# Resume
checkpoint = torch.load('checkpoint.pt')
agent.policy_net.load_state_dict(checkpoint['policy_net'])
agent.target_net.load_state_dict(checkpoint['target_net'])
agent.optimizer.load_state_dict(checkpoint['optimizer'])
start_episode = checkpoint['episode']
agent.epsilon = checkpoint['epsilon']
```

---

### Q: Model not improving after 500 episodes?

**A**: Try:
1. Check convergence criteria (may already be good)
2. Train longer (up to 2000 episodes)
3. Adjust hyperparameters
4. Verify reward function
5. Check for bugs in action execution

---

### Q: How to compare multiple models?

**A**:
```bash
# Train multiple configurations
python cli.py train --battery 15 --episodes 1000
python cli.py train --battery 18 --episodes 1000
python cli.py train --battery 20 --episodes 1000

# Compare
python cli.py compare --capacities 15 18 20 --parallel --plot
```

---

### Q: Best practices for reproducibility?

**A**:
```python
# Set all random seeds
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

---

### Q: Model file too large?

**A**: 
Typical model: ~2.4 MB

If larger:
```python
# Save only essentials
torch.save({
    'policy_net': agent.policy_net.state_dict()
}, 'model_small.pt', _use_new_zipfile_serialization=True)
```

---

### Q: How to export results for analysis?

**A**:
```python
# All results saved as CSV
import pandas as pd

# Training progress
train_df = pd.read_csv('results/latest_training_progress.csv')

# Hourly performance
hourly_df = pd.read_csv('results/latest_resilience_detailed.csv')

# Export to Excel
with pd.ExcelWriter('results_analysis.xlsx') as writer:
    train_df.to_excel(writer, sheet_name='Training')
    hourly_df.to_excel(writer, sheet_name='Hourly')
```

---

## Still Having Issues?

### Debugging Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Sufficient disk space
- [ ] Latest code version
- [ ] Tried restarting Python kernel
- [ ] Checked error message carefully
- [ ] Searched existing GitHub issues

### Getting Help

1. **GitHub Issues**: [Create new issue](https://github.com/YOUR_USERNAME/dynamic-microgrid-resilience/issues/new)
   - Include error message
   - Provide minimal reproducible example
   - Specify OS, Python version, PyTorch version

2. **GitHub Discussions**: [Ask question](https://github.com/YOUR_USERNAME/dynamic-microgrid-resilience/discussions)
   - General questions
   - Usage advice
   - Feature requests

3. **Email**: davila.alfonso@gmail.com
   - Complex issues
   - Research collaboration
   - Custom deployments

---

## Error Message Index

Quick lookup for common errors:

| Error | Section |
|-------|---------|
| `ImportError: No module named 'torch'` | [Installation](#installation-issues) |
| `RuntimeError: CUDA out of memory` | [Memory Issues](#memory-issues-during-training) |
| `FileNotFoundError: 'models/'` | [Runtime Errors](#runtime-errors) |
| `Loss = nan` | [Loss Exploding](#loss-exploding-or-nan) |
| `Not converging` | [Training Not Converging](#training-not-converging) |
| `Poor evening coverage` | [Performance Issues](#poor-evening-coverage-95) |
| `Plots not showing` | [Visualization Problems](#plots-not-generating) |

---

For more information:
- [API Reference](API.md)
- [Training Guide](TRAINING.md)
- [Methodology](METHODOLOGY.md)
- [Main README](../README.md)
