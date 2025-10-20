# Training Guide

Complete guide to training and optimizing the Anticipatory DQN model.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Training Options](#training-options)
- [Hyperparameter Tuning](#hyperparameter-tuning)
- [Monitoring Training](#monitoring-training)
- [Convergence Analysis](#convergence-analysis)
- [Common Issues](#common-issues)
- [Advanced Techniques](#advanced-techniques)
- [Best Practices](#best-practices)

---

## Quick Start

### Basic Training (5 minutes)

```bash
# Quick demo with 100 episodes
python cli.py demo
```

**Output:**
- Training progress printed to console
- Final performance metrics
- Model saved to `models/`

### Full Training (20 minutes)

```bash
# Complete training with visualizations
python cli.py train --full --plot --economics
```

**Output:**
- `models/microgrid_v7_*.pt` - Trained model
- `plots/*_progress.png` - Training curves
- `results/*_progress.csv` - Training data
- `results/*_config_report.txt` - System report

### Custom Training

```bash
# Train with specific configuration
python cli.py train \
    --episodes 1000 \
    --battery 20 \
    --plot \
    --monitor
```

---

## Training Options

### CLI Arguments

```bash
python cli.py train [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--episodes` | 500 | Number of training episodes |
| `--battery` | 18 | Battery capacity (kWh) |
| `--full` | False | Use 2000 episodes |
| `--plot` | False | Generate plots |
| `--economics` | False | Economic analysis |
| `--monitor` | False | Real-time monitoring |

### Example Configurations

#### Quick Training (Testing)
```bash
python cli.py train --episodes 100
```
- Time: ~2 minutes
- Use: Quick iteration, debugging
- Performance: 70-85% (not converged)

#### Standard Training
```bash
python cli.py train --episodes 500 --plot
```
- Time: ~10 minutes
- Use: Most use cases
- Performance: 95-97%

#### Full Training (Production)
```bash
python cli.py train --full --plot --economics
```
- Time: ~20 minutes
- Use: Best performance, publication
- Performance: 98-99%

#### Custom Battery Size
```bash
python cli.py train --battery 20 --episodes 1000 --plot
```
- Test different battery capacities
- Useful for sensitivity analysis
- Compare multiple configurations

---

## Hyperparameter Tuning

### Default Configuration

```python
# Optimal parameters (found through grid search)
learning_rate = 0.0002
discount_factor = 0.97
batch_size = 64
replay_buffer_size = 10000
epsilon_start = 1.0
epsilon_decay = 0.9992
epsilon_min = 0.01
target_update_freq = 10
```

### Tuning Guidelines

#### Learning Rate

```python
# Too high (>0.001): Unstable training
learning_rate = 0.0005  # ❌ Oscillating loss

# Too low (<0.00005): Slow convergence
learning_rate = 0.00005  # ❌ 2000+ episodes needed

# Optimal
learning_rate = 0.0002  # ✅ Stable + fast
```

**Symptoms:**
- Too high: Loss spikes, unstable rewards
- Too low: Flat learning curve, slow improvement

**Recommendation**: Start with 0.0002, divide by 2 if unstable

#### Discount Factor (γ)

```python
# Too low (<0.95): Myopic policy
discount_factor = 0.90  # ❌ Poor evening preparation

# Too high (>0.99): Slow learning
discount_factor = 0.99  # ❌ Overweights distant rewards

# Optimal
discount_factor = 0.97  # ✅ Good horizon
```

**Impact:**
- γ = 0.90: 92% evening coverage
- γ = 0.95: 96% evening coverage
- γ = 0.97: 99% evening coverage
- γ = 0.99: 98% evening coverage (slower)

**Recommendation**: Use 0.97 for 24-hour horizon

#### Epsilon Decay

```python
# Too fast (>0.999): Premature exploitation
epsilon_decay = 0.995  # ❌ Stuck in local optimum

# Too slow (<0.998): Excessive exploration
epsilon_decay = 0.998  # ❌ Never converges

# Optimal
epsilon_decay = 0.9992  # ✅ Balanced
```

**Schedule:**
- Episode 0: ε = 1.00 (100% exploration)
- Episode 100: ε = 0.82
- Episode 500: ε = 0.37
- Episode 1000: ε = 0.13
- Episode 2000: ε = 0.02

**Recommendation**: 0.9992 gives good exploration→exploitation transition

#### Batch Size

```python
# Too small (<32): Noisy gradients
batch_size = 16  # ❌ Unstable updates

# Too large (>256): Slow adaptation
batch_size = 256  # ❌ Less frequent updates

# Optimal
batch_size = 64  # ✅ Stable + efficient
```

**Trade-off:**
- Smaller: More updates, noisier
- Larger: Fewer updates, smoother

**Recommendation**: 64 for most cases, 128 for GPU

### Advanced Tuning

#### Reward Weights

```python
# In Dynamic_Microgrid_Resilience_v7.py

# Modify reward magnitudes
BASE_REWARD_MET = 100      # Default
BASE_PENALTY_UNMET = -300  # Default

PRE_EVENING_BONUS = 200    # For SOC ≥ 75%
PRE_EVENING_PENALTY = -250 # For SOC < 75%

EVENING_BONUS = 250        # For success + SOC ≥ 70%
EVENING_PENALTY = -400     # For failure

DAYTIME_CHARGE_BONUS = 50  # For charging when SOC < 80%
```

**Tuning Tips:**
1. Increase evening penalty if evening coverage < 95%
2. Increase pre-evening bonus if pre-evening SOC < 75%
3. Adjust daytime bonus to encourage more/less daytime charging

#### Network Architecture

```python
# In Dynamic_Microgrid_Resilience_v7.py, class DQNNetwork

# Default architecture
self.fc1 = nn.Linear(state_dim, 256)
self.fc2 = nn.Linear(256, 128)
self.fc3 = nn.Linear(128, 64)
self.fc4 = nn.Linear(64, action_dim)

# Smaller (faster, less capacity)
self.fc1 = nn.Linear(state_dim, 128)
self.fc2 = nn.Linear(128, 64)
self.fc3 = nn.Linear(64, action_dim)

# Larger (slower, more capacity)
self.fc1 = nn.Linear(state_dim, 512)
self.fc2 = nn.Linear(512, 256)
self.fc3 = nn.Linear(256, 128)
self.fc4 = nn.Linear(128, action_dim)
```

**When to change:**
- Smaller: Simple problems, faster training
- Larger: Complex patterns, more data

**Default is optimal for this problem**

---

## Monitoring Training

### Real-Time Monitor

```bash
python cli.py train --episodes 1000 --monitor
```

**Displays:**
```
Episode 100/1000 | Reward: +1847.3 | Load Met: 96.2% | ε: 0.82 | Loss: 2.14
```

### Training Metrics

Monitor these key metrics:

#### 1. Episode Reward
- **Good**: Increasing trend
- **Bad**: Flat or decreasing
- **Target**: +2000 to +2500

#### 2. Load Met Rate
- **Good**: >95% by episode 500
- **Bad**: <90% after 500 episodes
- **Target**: 98-100%

#### 3. Unmet Energy
- **Good**: <0.5 kWh by episode 500
- **Bad**: >2 kWh after 500 episodes
- **Target**: 0.0-0.2 kWh

#### 4. Training Loss
- **Good**: Decreasing, stabilizes around 1-3
- **Bad**: Increasing or >10
- **Target**: 1-5

#### 5. Epsilon
- **Good**: Smooth decay from 1.0 to ~0.1
- **Bad**: Stuck at 1.0 or jumps
- **Target**: ~0.15 at episode 1000

### Visualization

After training with `--plot`:

#### Training Progress Plot
```
4 panels:
- Load Met Rate: Should increase to 98%+
- Unmet Energy: Should decrease to ~0
- Epsilon: Should decay smoothly
- Training Loss: Should stabilize
```

#### Hourly Performance Plot
```
4 panels:
- Power flows: Check renewable vs load
- Battery SOC: Should reach 87% by hour 15
- Load coverage: Should be 100% all hours
- Resilience: Should be high during evening
```

---

## Convergence Analysis

### What is Convergence?

Convergence occurs when:
1. Load met rate ≥ 98%
2. Unmet energy < 0.2 kWh
3. Episode reward > +2000
4. Performance stable for 100+ episodes

### Typical Convergence Timeline

```
Episode 0-50:    Random exploration (50-70% coverage)
Episode 50-100:  Learning basics (70-85% coverage)
Episode 100-200: Discovering pre-evening strategy (85-95%)
Episode 200-500: Refinement (95-98%)
Episode 500+:    Fine-tuning (98-100%)
```

### Best Policy Tracking

The CLI automatically tracks best policy:

```python
# Validation every 100 episodes
if episode % 100 == 0:
    validate and save if best
```

**Output:**
```
Episode 100: Validation score = 2138.5 (NEW BEST)
Episode 200: Validation score = 2247.1 (NEW BEST)
Episode 300: Validation score = 2189.3
...
```

### Early Stopping

Training automatically stops if:
- Perfect performance (100% coverage) for 5 consecutive validations
- Typically occurs around episode 100-300

**Example:**
```
Episode 100: 100.0% load met - NEW BEST
Episode 200: 100.0% load met
Episode 300: 100.0% load met
Episode 400: 100.0% load met
Episode 500: 100.0% load met

Early stopping: Perfect performance achieved!
Best policy from episode 100
```

---

## Common Issues

### Issue 1: Not Converging

**Symptoms:**
- Load met rate stuck at 85-90%
- Unmet energy not decreasing
- Flat learning curve

**Solutions:**

1. **Train longer**
   ```bash
   python cli.py train --episodes 2000
   ```

2. **Increase learning rate**
   ```python
   learning_rate = 0.0003  # from 0.0002
   ```

3. **Reduce epsilon decay**
   ```python
   epsilon_decay = 0.999  # from 0.9992
   ```

4. **Check reward function**
   - Ensure evening penalties are active
   - Verify pre-evening rewards working

### Issue 2: Unstable Training

**Symptoms:**
- Loss spiking
- Rewards oscillating
- Performance degrading

**Solutions:**

1. **Reduce learning rate**
   ```python
   learning_rate = 0.0001  # from 0.0002
   ```

2. **Increase batch size**
   ```python
   batch_size = 128  # from 64
   ```

3. **More frequent target updates**
   ```python
   target_update_freq = 5  # from 10
   ```

### Issue 3: Poor Evening Coverage

**Symptoms:**
- Daytime: 100% coverage
- Evening: 80-90% coverage
- Pre-evening SOC < 70%

**Solutions:**

1. **Increase evening penalty**
   ```python
   EVENING_PENALTY = -500  # from -400
   ```

2. **Increase pre-evening bonus**
   ```python
   PRE_EVENING_BONUS = 300  # from 200
   ```

3. **Check pre-evening flag**
   ```python
   # Verify state feature 7 is active
   print(state[6])  # Should be 1.0 for hours 15-17
   ```

### Issue 4: Overcharging

**Symptoms:**
- Battery at 100% SOC most of the day
- Wasted excess generation
- Poor nighttime performance

**Solutions:**

1. **Reduce daytime charging bonus**
   ```python
   DAYTIME_CHARGE_BONUS = 25  # from 50
   ```

2. **Add overcharge penalty**
   ```python
   if soc > 0.85 and hour < 15:
       reward -= 50
   ```

---

## Advanced Techniques

### Multi-Seed Training

Train with multiple random seeds for robust policy:

```bash
# Train 5 times with different seeds
for seed in {0..4}; do
    python cli.py train --episodes 1000 --plot
    mv models/microgrid_v7_*.pt models/seed_${seed}_model.pt
done

# Evaluate all models
python examples/batch_evaluation.py --compare models/seed_*.pt
```

### Transfer Learning

Use pre-trained model for new configuration:

```python
import torch
from Dynamic_Microgrid_Resilience_v7 import DQNAgent, MicrogridConfig

# New configuration (20 kWh battery)
new_config = MicrogridConfig(battery_capacity_kwh=20.0)
agent = DQNAgent(new_config)

# Load pre-trained weights (18 kWh battery)
checkpoint = torch.load('models/microgrid_v7_18kwh_model.pt')
agent.policy_net.load_state_dict(checkpoint['policy_net'])

# Fine-tune on new configuration
# (Will converge faster than training from scratch)
```

### Curriculum Learning

Start with easier problems, gradually increase difficulty:

```python
# Phase 1: Constant generation/load (100 episodes)
for episode in range(100):
    # Fixed profiles, no stochasticity
    
# Phase 2: Low variability (200 episodes)
for episode in range(100, 300):
    # ±5% variability
    
# Phase 3: Full stochasticity (700 episodes)
for episode in range(300, 1000):
    # ±10% variability, full weather variation
```

### Ensemble Methods

Combine multiple models for robust predictions:

```python
models = [load_model(f'models/seed_{i}_model.pt') for i in range(5)]

def ensemble_action(state):
    q_values = [model(state) for model in models]
    avg_q = torch.mean(torch.stack(q_values), dim=0)
    return torch.argmax(avg_q).item()
```

---

## Best Practices

### 1. Start Simple

```bash
# First run: Demo mode
python cli.py demo

# Verify it works before full training
```

### 2. Save Regularly

```python
# Models are auto-saved, but you can save manually
torch.save({
    'policy_net': agent.policy_net.state_dict(),
    'episode': episode,
    'config': config.__dict__
}, f'models/checkpoint_ep{episode}.pt')
```

### 3. Monitor Live

```bash
# Use monitor flag for long training runs
python cli.py train --episodes 2000 --monitor
```

### 4. Generate Plots

```bash
# Always generate plots for analysis
python cli.py train --full --plot --economics
```

### 5. Test Thoroughly

```bash
# After training, evaluate on 100 diverse scenarios
python cli.py evaluate --auto-latest --scenarios 100 --plot
```

### 6. Document Configuration

```python
# Save configuration with model
config_dict = {
    'pv_capacity': config.pv_capacity_kw,
    'battery_capacity': config.battery_capacity_kwh,
    'learning_rate': 0.0002,
    'discount_factor': 0.97,
    'training_episodes': 1000
}
torch.save(config_dict, 'models/config.json')
```

### 7. Version Control

```bash
# Use git to track different experiments
git commit -m "Training run: 1000 eps, lr=0.0002, γ=0.97"
git tag v7.0-experiment-1
```

### 8. Reproducibility

```python
# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## Performance Targets

### Episode Milestones

| Episode | Load Coverage | Unmet Energy | Evening SOC |
|---------|---------------|--------------|-------------|
| 100 | 85-90% | 1.5-2.5 kWh | 65-70% |
| 200 | 90-95% | 0.8-1.5 kWh | 70-75% |
| 500 | 95-98% | 0.2-0.8 kWh | 75-82% |
| 1000 | 98-100% | 0.0-0.2 kWh | 82-87% |

### Final Performance (Episode 1000+)

```
✓ Load Met Rate: 98-100%
✓ Unmet Energy: 0.0-0.2 kWh
✓ Pre-Evening SOC: 85-90%
✓ Evening SOC: 80-85%
✓ Perfect Days: 90-100%
```

---

## Troubleshooting Checklist

Before asking for help, check:

- [ ] Using latest code version
- [ ] Correct Python version (3.8+)
- [ ] All dependencies installed
- [ ] Sufficient disk space for models/results
- [ ] GPU drivers (if using CUDA)
- [ ] Training for enough episodes (≥500)
- [ ] Monitoring key metrics
- [ ] Plots generated for analysis

---

For more information:
- [API Reference](API.md)
- [Methodology](METHODOLOGY.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Main README](../README.md)

---

**Need help? Check [Troubleshooting](TROUBLESHOOTING.md) or open an issue on GitHub.**
