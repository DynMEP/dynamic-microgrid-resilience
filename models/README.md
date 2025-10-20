# Models Directory

This directory contains saved model checkpoints from training.

## File Naming Convention

Models are saved with timestamps:
```
microgrid_v7_YYYYMMDD_HHMMSS_model.pt
```

Example: `microgrid_v7_20251019_094723_model.pt`

## Model Contents

Each `.pt` file contains:
- `policy_net`: Policy network state dict
- `target_net`: Target network state dict  
- `optimizer`: Optimizer state dict
- `episode`: Episode number when saved
- `config`: Model configuration
- `performance`: Training metrics

## Loading Models

```python
import torch
from Dynamic_Microgrid_Resilience_v7 import DQNAgent, MicrogridConfig

# Load configuration
config = MicrogridConfig()
agent = DQNAgent(config)

# Load trained model
checkpoint = torch.load('models/microgrid_v7_20251019_094723_model.pt')
agent.policy_net.load_state_dict(checkpoint['policy_net'])
agent.policy_net.eval()

# Use for inference
state = agent.get_initial_state()
action_idx = agent.select_action(state, explore=False)
```

## Pre-trained Models

Download pre-trained models from releases:
- [Latest Release](https://github.com/DynMEP/dynamic-microgrid-resilience/releases/latest)

## Model Size

Typical model size: ~2.4 MB

## Note

`.pt` files are excluded from git tracking (see `.gitignore`).
Download pre-trained models from GitHub Releases.
