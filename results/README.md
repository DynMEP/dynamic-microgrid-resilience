# Results Directory

This directory contains training results, validation metrics, and configuration reports in CSV and text format.

## File Types

### Training Progress (`*_training_progress.csv`)
Episode-by-episode metrics during training:
- `episode`: Episode number
- `total_reward`: Cumulative reward
- `load_met_rate`: Percentage of hours with load met
- `unmet_energy`: Total unmet energy (kWh)
- `epsilon`: Exploration rate
- `avg_loss`: Average training loss

### Validation Results (`*_validation_results.csv`)
Periodic validation metrics (every 100 episodes):
- `episode`: Episode number
- `load_met_rate`: Validation load coverage
- `unmet_energy`: Validation unmet energy
- `performance_score`: Combined metric
- `is_best`: Boolean flag for best policy

### Resilience Details (`*_resilience_detailed.csv`)
Hour-by-hour system performance:
- `hour`: Hour of day (0-23)
- `soc_percent`: Battery state of charge (%)
- `pv_power`, `wind_power`, `load`: Power flows (kW)
- `charged`, `discharged`: Battery actions (kW)
- `load_met`: Boolean flag
- `action`: Agent's action
- `resilience_score`: Hourly resilience metric

### Hourly Summary (`*_resilience_hourly_summary.csv`)
Statistical summary by hour across episodes:
- `hour`: Hour of day
- `avg_soc`, `min_soc`, `max_soc`: SOC statistics
- `avg_load_met_rate`: Load coverage by hour
- `avg_resilience`: Average resilience score

### Configuration Report (`*_config_report.txt`)
Complete system configuration and final performance:
- System parameters (PV, wind, battery specs)
- Training hyperparameters
- Best policy performance metrics
- NEC compliance details
- Convergence information

## Example Usage

### Load Training Progress
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/microgrid_v7_20251019_094723_training_progress.csv')

plt.figure(figsize=(10, 6))
plt.plot(df['episode'], df['load_met_rate'])
plt.xlabel('Episode')
plt.ylabel('Load Met Rate (%)')
plt.title('Training Convergence')
plt.show()
```

### Analyze Best Episode
```python
val_df = pd.read_csv('results/microgrid_v7_20251019_094723_validation_results.csv')
best_episode = val_df[val_df['is_best']]['episode'].values[0]
print(f"Best policy found at episode: {best_episode}")
```

### Examine Hourly Performance
```python
hourly_df = pd.read_csv('results/microgrid_v7_20251019_094723_resilience_detailed.csv')

# Plot SOC throughout day
plt.figure(figsize=(12, 6))
plt.plot(hourly_df['hour'], hourly_df['soc_percent'])
plt.axhline(y=75, color='g', linestyle='--', label='Target Pre-Evening SOC')
plt.axhline(y=70, color='orange', linestyle='--', label='Target Evening SOC')
plt.xlabel('Hour of Day')
plt.ylabel('Battery SOC (%)')
plt.legend()
plt.show()
```

## File Naming Convention

```
microgrid_v7_YYYYMMDD_HHMMSS_<type>.<ext>
```

Examples:
- `microgrid_v7_20251019_094723_training_progress.csv`
- `microgrid_v7_20251019_094723_validation_results.csv`
- `microgrid_v7_20251019_094723_config_report.txt`

## Data Retention

Results from all training runs are saved for:
- Reproducibility
- Performance comparison
- Hyperparameter analysis
- Publication figures

## Note

CSV and TXT files are excluded from git tracking (see `.gitignore`).
This prevents repository bloat while maintaining local analysis capability.
