# API Reference

Complete API documentation for Dynamic Microgrid Resilience v7.0

---

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Agent Interface](#agent-interface)
- [Environment](#environment)
- [Training Functions](#training-functions)
- [Evaluation Functions](#evaluation-functions)
- [Utility Functions](#utility-functions)

---

## Core Classes

### `MicrogridConfig`

System configuration dataclass.

```python
from Dynamic_Microgrid_Resilience_v7 import MicrogridConfig

config = MicrogridConfig(
    pv_capacity_kw=5.0,
    wind_capacity_kw=3.0,
    battery_capacity_kwh=18.0,
    battery_power_kw=6.0,
    battery_efficiency=0.95
)
```

**Parameters:**
- `pv_capacity_kw` (float): Solar PV capacity in kW. Default: 5.0
- `wind_capacity_kw` (float): Wind turbine capacity in kW. Default: 3.0
- `battery_capacity_kwh` (float): Battery energy capacity in kWh. Default: 18.0
- `battery_power_kw` (float): Battery power rating in kW. Default: 6.0
- `battery_efficiency` (float): Round-trip efficiency [0-1]. Default: 0.95
- `min_soc` (float): Minimum state of charge [0-1]. Default: 0.2
- `max_soc` (float): Maximum state of charge [0-1]. Default: 1.0

**NEC Compliance:**
- Follows NEC 2023 Articles 690 (Solar), 694 (Wind), 706 (Energy Storage)
- DC voltage: ≤600V (NEC 690.7)
- Overcurrent protection: 125% rating (NEC 690.8)
- Battery disconnects per NEC 706.15

---

### `DQNAgent`

Deep Q-Network agent for energy management.

```python
from Dynamic_Microgrid_Resilience_v7 import DQNAgent

agent = DQNAgent(config)
```

**Attributes:**
- `state_dim` (int): State space dimension (7 features)
- `action_dim` (int): Action space dimension (5 actions)
- `actions` (list): Action names
- `policy_net` (nn.Module): Policy network (training)
- `target_net` (nn.Module): Target network (stable Q-values)
- `optimizer` (torch.optim.Adam): Adam optimizer
- `memory` (ReplayBuffer): Experience replay buffer
- `epsilon` (float): Current exploration rate

**Methods:**

#### `select_action(state, explore=True)`
Select action using epsilon-greedy policy.

**Parameters:**
- `state` (torch.Tensor): Current state (7 features)
- `explore` (bool): Whether to explore. Default: True

**Returns:**
- `action_idx` (int): Selected action index [0-4]

**Example:**
```python
state = agent.get_enhanced_state(soc, hour, renewable, load)
action_idx = agent.select_action(state, explore=True)
action = agent.actions[action_idx]
```

#### `get_enhanced_state(soc, hour, renewable, load)`
Create 7-feature state representation.

**Parameters:**
- `soc` (float): Battery energy in kWh
- `hour` (int): Hour of day [0-23]
- `renewable` (float): Total renewable generation in kW
- `load` (float): Load demand in kW

**Returns:**
- `state` (torch.Tensor): 7-feature state vector

**State Features:**
1. `soc_normalized`: SOC / capacity [0-1]
2. `hour_sin`: sin(2π * hour / 24)
3. `hour_cos`: cos(2π * hour / 24)
4. `renewable_ratio`: renewable / (load + 1e-6)
5. `net_balance`: (renewable - load) / (load + 1e-6)
6. `is_evening`: 1 if hour in [18-22], else 0
7. `is_pre_evening`: 1 if hour in [15-17], else 0

#### `train_step()`
Perform one training step using experience replay.

**Returns:**
- `loss` (float): Training loss value, or None if insufficient data

**Example:**
```python
if len(agent.memory) >= agent.batch_size:
    loss = agent.train_step()
```

#### `update_target_network()`
Copy weights from policy network to target network.

**Example:**
```python
if episode % 10 == 0:
    agent.update_target_network()
```

---

### `BatterySystem`

Battery energy storage system.

```python
from Dynamic_Microgrid_Resilience_v7 import BatterySystem

battery = BatterySystem(config)
```

**Attributes:**
- `capacity_kwh` (float): Maximum energy capacity
- `power_kw` (float): Maximum power rating
- `efficiency` (float): Round-trip efficiency
- `current_energy_kwh` (float): Current stored energy

**Methods:**

#### `charge(power_kw, duration_hours=1.0)`
Charge the battery.

**Parameters:**
- `power_kw` (float): Charging power in kW
- `duration_hours` (float): Duration in hours. Default: 1.0

**Returns:**
- `energy_charged` (float): Actual energy added in kWh

**Example:**
```python
excess_power = renewable - load
if excess_power > 0:
    charged = battery.charge(min(excess_power, battery.power_kw))
```

#### `discharge(power_kw, duration_hours=1.0)`
Discharge the battery.

**Parameters:**
- `power_kw` (float): Discharge power in kW
- `duration_hours` (float): Duration in hours. Default: 1.0

**Returns:**
- `energy_discharged` (float): Actual energy removed in kWh

**Example:**
```python
shortfall = load - renewable
if shortfall > 0:
    discharged = battery.discharge(min(shortfall, battery.power_kw))
```

#### `get_SOC_percent()`
Get state of charge as percentage.

**Returns:**
- `soc` (float): State of charge in percent [0-100]

#### `reset(initial_soc=0.5)`
Reset battery to initial state.

**Parameters:**
- `initial_soc` (float): Initial SOC fraction [0-1]. Default: 0.5

---

### `StochasticTimeSeriesGenerator`

Generate stochastic renewable and load profiles.

```python
from Dynamic_Microgrid_Resilience_v7 import StochasticTimeSeriesGenerator

generator = StochasticTimeSeriesGenerator(config)
```

**Methods:**

#### `generate_episode_profile(episode, load_type='residential')`
Generate one episode profile.

**Parameters:**
- `episode` (int): Episode number (used for seeding)
- `load_type` (str): Load profile type. Options: 'residential', 'commercial', 'mixed'. Default: 'residential'

**Returns:**
- `profile` (dict): Dictionary with keys:
  - `'pv_power'`: List[float] (24 hours)
  - `'wind_power'`: List[float] (24 hours)
  - `'load_demand'`: List[float] (24 hours)

**Example:**
```python
profile = generator.generate_episode_profile(
    episode=42,
    load_type='residential'
)

for hour in range(24):
    pv = profile['pv_power'][hour]
    wind = profile['wind_power'][hour]
    load = profile['load_demand'][hour]
```

---

## Training Functions

### `train_agent(agent, episodes=1000, verbose=True)`

Train the DQN agent.

**Parameters:**
- `agent` (DQNAgent): Agent to train
- `episodes` (int): Number of training episodes. Default: 1000
- `verbose` (bool): Print progress. Default: True

**Returns:**
- `results` (dict): Training results with keys:
  - `'episode_rewards'`: List[float]
  - `'load_met_rates'`: List[float]
  - `'unmet_energies'`: List[float]
  - `'epsilons'`: List[float]
  - `'losses'`: List[float]

**Example:**
```python
from Dynamic_Microgrid_Resilience_v7 import train_agent

results = train_agent(agent, episodes=1000, verbose=True)

# Access results
final_load_met = results['load_met_rates'][-1]
print(f"Final performance: {final_load_met:.1f}%")
```

---

## Evaluation Functions

### `evaluate_policy(agent, num_scenarios=100, config=None)`

Evaluate trained policy across multiple scenarios.

**Parameters:**
- `agent` (DQNAgent): Trained agent
- `num_scenarios` (int): Number of test scenarios. Default: 100
- `config` (MicrogridConfig): System config. If None, uses agent.config

**Returns:**
- `metrics` (dict): Performance metrics:
  - `'mean_load_met'`: float
  - `'std_load_met'`: float
  - `'mean_unmet_energy'`: float
  - `'scenarios_perfect'`: int (count with 100% coverage)
  - `'scenarios_good'`: int (count with ≥95% coverage)

**Example:**
```python
from evaluation import evaluate_policy

metrics = evaluate_policy(agent, num_scenarios=100)

print(f"Mean load coverage: {metrics['mean_load_met']:.1f}%")
print(f"Perfect scenarios: {metrics['scenarios_perfect']}/100")
```

---

## Action Space

The agent has 5 discrete actions:

| Index | Action | Description |
|-------|--------|-------------|
| 0 | `charge_full` | Charge at 100% power |
| 1 | `charge_half` | Charge at 50% power |
| 2 | `hold` | No battery action |
| 3 | `discharge_half` | Discharge at 50% power |
| 4 | `discharge_full` | Discharge at 100% power |

**Action Execution:**
```python
action = agent.actions[action_idx]

if action == 'charge_full':
    power = config.battery_power_kw * 1.0
    battery.charge(power)
elif action == 'charge_half':
    power = config.battery_power_kw * 0.5
    battery.charge(power)
elif action == 'hold':
    pass  # No action
elif action == 'discharge_half':
    power = config.battery_power_kw * 0.5
    battery.discharge(power)
elif action == 'discharge_full':
    power = config.battery_power_kw * 1.0
    battery.discharge(power)
```

---

## Reward Function

The reward function has hierarchical structure:

```python
# Base reward (all hours)
R_base = +100 if load_met else -300

# Pre-evening preparation (hours 15-17)
if hour in [15, 16, 17]:
    R_pre_evening = +200 if soc_pct >= 75 else -250
else:
    R_pre_evening = 0

# Evening peak (hours 18-22)
if hour in [18, 19, 20, 21, 22]:
    if load_met and soc_pct >= 70:
        R_evening = +250
    else:
        R_evening = -400
else:
    R_evening = 0

# Daytime charging (hours 6-14)
if hour in range(6, 15):
    if is_charging and soc_pct < 80:
        R_daytime = +50
    else:
        R_daytime = 0
else:
    R_daytime = 0

# Total reward
R_total = R_base + R_pre_evening + R_evening + R_daytime
```

---

## Network Architecture

```
Input: 7 features
    ↓
Linear(7, 256) + ReLU
    ↓
Linear(256, 128) + ReLU
    ↓
Linear(128, 64) + ReLU
    ↓
Linear(64, 5)
    ↓
Output: 5 Q-values (one per action)
```

**Total Parameters:** ~67,000

**Optimizer:** Adam with learning rate 0.0002

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0002 | Adam optimizer |
| Discount Factor (γ) | 0.97 | Future reward discount |
| Batch Size | 64 | Experience replay batch |
| Replay Buffer | 10,000 | Transition storage |
| ε-start | 1.0 | Initial exploration |
| ε-decay | 0.9992 | Per episode decay |
| ε-min | 0.01 | Minimum exploration |
| Target Update | 10 episodes | Target network sync |

---

## Complete Example

```python
import torch
from Dynamic_Microgrid_Resilience_v7 import (
    MicrogridConfig,
    DQNAgent,
    StochasticTimeSeriesGenerator,
    BatterySystem
)

# 1. Configuration
config = MicrogridConfig(
    pv_capacity_kw=5.0,
    wind_capacity_kw=3.0,
    battery_capacity_kwh=18.0,
    battery_power_kw=6.0
)

# 2. Create agent
agent = DQNAgent(config)
generator = StochasticTimeSeriesGenerator(config)

# 3. Training loop
for episode in range(100):
    battery = BatterySystem(config)
    profile = generator.generate_episode_profile(episode)
    
    episode_reward = 0
    
    for hour in range(24):
        # Get state
        soc = battery.get_SOC_energy()
        renewable = profile['pv_power'][hour] + profile['wind_power'][hour]
        load = profile['load_demand'][hour]
        state = agent.get_enhanced_state(soc, hour, renewable, load)
        
        # Select action
        action_idx = agent.select_action(state, explore=True)
        action = agent.actions[action_idx]
        
        # Execute action
        if action.startswith('charge'):
            power = config.battery_power_kw * (1.0 if 'full' in action else 0.5)
            battery.charge(min(max(0, renewable - load), power))
        elif action.startswith('discharge'):
            power = config.battery_power_kw * (1.0 if 'full' in action else 0.5)
            battery.discharge(min(max(0, load - renewable), power))
        
        # Compute reward
        net_power = renewable + battery.get_SOC_energy() * 0.1
        load_met = net_power >= load - 0.001
        reward = 100 if load_met else -300
        episode_reward += reward
        
        # Store transition
        next_soc = battery.get_SOC_energy()
        next_hour = (hour + 1) % 24
        next_state = agent.get_enhanced_state(next_soc, next_hour, renewable, load)
        agent.memory.push(state, action_idx, reward, next_state, False)
        
        # Train
        if len(agent.memory) >= agent.batch_size:
            agent.train_step()
    
    # Update epsilon
    agent.epsilon = max(agent.epsilon * 0.9992, 0.01)
    
    # Update target network
    if episode % 10 == 0:
        agent.update_target_network()
    
    print(f"Episode {episode}: Reward = {episode_reward:.1f}")

# 4. Save model
torch.save({
    'policy_net': agent.policy_net.state_dict(),
    'target_net': agent.target_net.state_dict(),
    'optimizer': agent.optimizer.state_dict()
}, 'trained_model.pt')

# 5. Evaluate
agent.epsilon = 0.0  # No exploration
battery = BatterySystem(config)
profile = generator.generate_episode_profile(999)

hours_met = 0
for hour in range(24):
    soc = battery.get_SOC_energy()
    renewable = profile['pv_power'][hour] + profile['wind_power'][hour]
    load = profile['load_demand'][hour]
    state = agent.get_enhanced_state(soc, hour, renewable, load)
    
    with torch.no_grad():
        action_idx = agent.select_action(state, explore=False)
    
    # Execute best action...
    
print(f"Evaluation: {hours_met}/24 hours met ({hours_met/24*100:.1f}%)")
```

---

## Error Handling

Common errors and solutions:

### `RuntimeError: CUDA out of memory`
**Solution:** Force CPU usage
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### `ValueError: invalid SOC`
**Solution:** Check battery constraints
```python
assert 0 <= battery.get_SOC_energy() <= config.battery_capacity_kwh
```

### `IndexError: action out of range`
**Solution:** Ensure action_idx in [0, 4]
```python
action_idx = max(0, min(4, action_idx))
```

---

## Performance Tips

1. **Use GPU for large batches:**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   agent = DQNAgent(config, device=device)
   ```

2. **Increase batch size for faster convergence:**
   ```python
   agent.batch_size = 128  # default is 64
   ```

3. **Adjust epsilon decay for faster exploitation:**
   ```python
   agent.epsilon = max(agent.epsilon * 0.995, 0.01)  # faster decay
   ```

4. **Use larger replay buffer for more diverse experience:**
   ```python
   agent.memory = ReplayBuffer(capacity=50000)  # default is 10000
   ```

---

## Version Information

**Version:** 7.0.0  
**Released:** October 2025  
**PyTorch:** 2.0+  
**Python:** 3.8+

---

For more details, see:
- [Methodology](METHODOLOGY.md)
- [Training Guide](TRAINING.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Main README](../README.md)
