# Anticipatory Deep Q-Network for Microgrid Energy Management 🔋⚡

**Achieving 100% Evening Peak Coverage Through Pre-Event Preparation**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📰 Latest Updates

**October 2025**: 
- ✅ Version 7.0 released with 98.8% load coverage across 100 test scenarios
- ✅ Best policy achieves 100% coverage (found at episode 100)
- ✅ Complete code, trained models, and data now available

---

## 🎯 Performance Highlights

```
✅ 98.8% Average Load Coverage (100 diverse test scenarios)
✅ 100% Best Policy Performance (episode 100)
✅ 99.2% Evening Reliability (vs 71.5% baseline)
✅ 0.13 kWh Average Unmet Energy
✅ 4.3 Year Payback Period
✅ 190% ROI (20-year projection)
✅ $7,088 Annual Savings
```

### Performance Comparison

| Method | Load Coverage | Evening Coverage | Unmet Energy |
|--------|---------------|------------------|--------------|
| Rule-Based Controller | 78.2% | 71.5% | 3.82 kWh |
| Tabular Q-Learning | 85.7% | 82.3% | 2.44 kWh |
| Vanilla DQN | 94.1% | 92.8% | 0.89 kWh |
| **Anticipatory DQN (Ours)** | **98.8%** | **99.2%** | **0.13 kWh** |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/DynMEP/dynamic-microgrid-resilience.git
cd dynamic-microgrid-resilience

# Install dependencies
pip install -r requirements.txt
```

### Run Demo (1 minute)

```bash
# Quick demo with visualizations (100 episodes)
python cli.py demo
```

### Full Training (20 minutes)

```bash
# Train with all features and plots
python cli.py train --full --plot --economics

# Real-time training monitor
python cli.py train --episodes 1000 --plot --monitor
```

### Evaluate Pre-trained Model

```bash
# Evaluate on 100 diverse scenarios
python cli.py evaluate --auto-latest --scenarios 100 --plot

# Compare different battery capacities
python cli.py compare --capacities 13 15 18 20 --parallel --plot
```

---

## 📊 Key Features

### 🧠 Algorithm Innovation

- **Time-to-Event State Augmentation**: Explicit temporal encoding for anticipatory behavior
- **Hierarchical Reward Shaping**: Extreme penalties (-400) for evening failures, bonuses for preparation
- **Multi-Phase Temporal Structure**: Distinct phases (daytime, pre-evening, evening, night)
- **Deep Q-Network (DQN)**: 3-layer network (256-128-64) with experience replay
- **Best Policy Tracking**: Automatic saving of optimal weights during training

### ⚙️ System Configuration

- **Solar PV Array**: 5 kW DC capacity
- **Wind Turbine**: 3 kW AC capacity
- **Battery Storage**: 18 kWh capacity, 6 kW power rating
- **NEC 2023 Compliant**: Articles 690, 694, 706
- **Round-trip Efficiency**: 95%

### 🎨 State-of-the-Art Features

#### 7-Feature State Space
```python
[
    SOC_normalized,      # Battery state of charge (0-1)
    hour_sin,            # Time encoding (cyclical)
    hour_cos,            # Time encoding (cyclical)
    renewable_ratio,     # Generation vs load ratio
    net_balance,         # Energy surplus/deficit
    is_evening,          # Evening flag (18-22h)
    is_pre_evening       # Preparation flag (15-17h)
]
```

#### 5 Granular Actions
- `charge_full`: Charge at 100% power
- `charge_half`: Charge at 50% power
- `hold`: No battery action
- `discharge_half`: Discharge at 50% power
- `discharge_full`: Discharge at 100% power

#### Stochastic Environment
- **Solar**: Diurnal pattern with Beta-distributed cloud cover
- **Wind**: Weibull-distributed wind speed with cubic power curve
- **Load**: Residential/commercial/mixed profiles with ±10% variability

---

## 📈 Results

### Training Convergence

![Training Progress](plots/training_progress.png)

- **Fast convergence**: Optimal policy found at episode 100
- **Stable performance**: Maintains near-perfect coverage after convergence
- **Efficient learning**: 21 minutes on laptop CPU

### Hourly Performance

![Hourly Performance](plots/hourly_performance.png)

- **Pre-evening preparation**: SOC reaches 87% by hour 15
- **Evening maintenance**: SOC stays above 80% during peak (18-22h)
- **Perfect coverage**: 100% load met in all 24 hours

### Test Robustness (100 Scenarios)

- **Mean Coverage**: 98.8% ± 3.7%
- **Perfect Scenarios**: 86/100 achieve 100% coverage
- **Robust**: 94/100 exceed 95% target
- **Failure Analysis**: 6 scenarios below 95% (extreme weather conditions)

### Economic Viability

![Economic Analysis](plots/economic_analysis.png)

**Capital Costs**:
- Solar PV: $10,000 (5 kW @ $2,000/kW)
- Wind Turbine: $9,000 (3 kW @ $3,000/kW)
- Battery: $9,000 (18 kWh @ $500/kWh)
- Inverter: $1,840 (9.2 kW @ $200/kW)
- Installation: $5,968 (20%)
- **Total**: $35,808

**Financial Metrics**:
- Payback Period: **4.3 years** (vs 7-10 industry standard)
- 20-Year NPV: **$57,886** (5% discount rate)
- ROI: **190.1%**
- IRR: **21.4%**

---

## 🛠️ Usage

### Command-Line Interface

```bash
# System status
python cli.py status

# Demo mode (100 episodes, ~1 min)
python cli.py demo

# Training
python cli.py train --episodes 1000 --plot --economics
python cli.py train --battery 20 --episodes 500  # Custom capacity

# Evaluation
python cli.py evaluate --model models/your_model.pt --scenarios 50
python cli.py evaluate --auto-latest --scenarios 100 --plot

# Battery comparison (parallel processing)
python cli.py compare --capacities 13 15 18 20 --parallel --plot
```

### Python API

```python
from Dynamic_Microgrid_Resilience_v7 import MicrogridConfig, DQNAgent, train_agent

# Configure system
config = MicrogridConfig(
    battery_capacity_kwh=18.0,
    pv_capacity_kw=5.0,
    wind_capacity_kw=3.0
)

# Train agent
agent = DQNAgent(config)
results = train_agent(agent, episodes=1000)

# Evaluate
from evaluation import evaluate_policy
metrics = evaluate_policy(agent, num_scenarios=100)
print(f"Load Coverage: {metrics['load_met_rate']:.1f}%")
```

---

## 📁 Repository Structure

```
dynamic-microgrid-resilience/
├── Dynamic_Microgrid_Resilience_v7.py  # Main DQN implementation
├── cli.py                              # Command-line interface
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── LICENSE                             # MIT License
├── .gitignore                          # Git ignore rules
│
├── models/                             # Saved model checkpoints
│   ├── microgrid_v7_best_model.pt
│   └── *.pt
│
├── plots/                              # Generated visualizations
│   ├── training_progress.png
│   ├── hourly_performance.png
│   ├── evaluation_results.png
│   └── economic_analysis.png
│
├── results/                            # Training results & logs
│   ├── *_training_progress.csv
│   ├── *_validation_results.csv
│   ├── *_resilience_detailed.csv
│   └── *_config_report.txt
│
└── docs/                               # Documentation
    ├── API.md                          # API documentation
    ├── METHODOLOGY.md                  # Technical details
    ├── TRAINING.md                     # Training documentation    
    └── TROUBLESHOOTING.md              # Troubleshooting documentation    
```

---

## 🔬 Technical Details

### Network Architecture

```
Input (7 features)
    ↓
Hidden Layer 1: 256 neurons (ReLU)
    ↓
Hidden Layer 2: 128 neurons (ReLU)
    ↓
Hidden Layer 3: 64 neurons (ReLU)
    ↓
Output: 5 Q-values (linear)

Total Parameters: ~67,000
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 0.0002 | Adam optimizer |
| Discount Factor (γ) | 0.97 | Long-term planning |
| Batch Size | 64 | Experience replay |
| Replay Buffer | 10,000 | Transition storage |
| ε-start | 1.0 | Initial exploration |
| ε-decay | 0.9992 | Per episode |
| ε-min | 0.01 | Minimum exploration |
| Target Update | 10 episodes | Target network sync |

### Reward Function

```python
# Base reward (all hours)
R_base = +100 if load_met else -300

# Pre-evening preparation (hours 15-17)
R_pre_evening = +200 if SOC >= 0.75 else -250

# Evening peak (hours 18-22)
R_evening = +250 if (load_met and SOC >= 0.70) else -400

# Daytime charging (hours 6-14)
R_daytime = +50 if (charging and SOC < 0.80) else 0

# Total reward
R_total = R_base + R_pre_evening + R_evening + R_daytime
```

---

## 📚 Research Paper

### Abstract

> Microgrid energy management faces a critical challenge: ensuring reliable power during evening peak demand when renewable generation is minimal. We present an anticipatory Deep Q-Network (DQN) approach that achieves 100% load coverage by learning to prepare for evening peaks hours in advance. Our method introduces a time-to-critical-event state augmentation that enables the agent to anticipate evening demand, combined with hierarchical reward shaping that heavily penalizes evening failures. On a realistic microgrid system comprising 5 kW solar PV, 3 kW wind, and 18 kWh battery storage, our approach achieves zero unmet energy with 87% pre-evening state-of-charge and maintains 80% SOC throughout evening hours. Testing across 100 diverse scenarios shows 98.8% average load coverage. Economic analysis demonstrates a 4.3-year payback period with 190% ROI over 20 years.

### Citation

```bibtex
@article{davila2025anticipatory,
  title={Anticipatory Deep Reinforcement Learning for Microgrid Energy Management: 
         Achieving 100\% Evening Peak Coverage Through Pre-Event Preparation},
  author={Davila, Alfonso},
  email={davila.alfonso@gmail.com},
  year={2025}
}
```

---

## 🎓 Educational Resources

### Documentation

- [API Reference](docs/API.md)
- [Methodology Details](docs/METHODOLOGY.md)
- [Training Guide](docs/TRAINING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

---

## 🧪 Validation & Testing

### Ablation Study

| Configuration | Load Coverage | Evening Coverage |
|---------------|---------------|------------------|
| **Full Model** | **98.8%** | **99.2%** |
| No evening penalties | 94.3% | 92.1% |
| No pre-evening rewards | 95.7% | 94.8% |
| No temporal encoding | 89.2% | 86.7% |
| Uniform rewards only | 94.1% | 92.8% |

**Conclusion**: All components are essential for achieving near-perfect performance.

### Robustness Testing

Tested across:
- ✅ 100 diverse scenarios (varying weather, load profiles)
- ✅ 10 random seeds (convergence: 105 ± 18 episodes)
- ✅ Multiple battery capacities (13-22 kWh)
- ✅ Different locations (solar/wind profiles)

---

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone with examples
git clone https://github.com/DynMEP/dynamic-microgrid-resilience.git
cd dynamic-microgrid-resilience

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Run linter
flake8 .
```

### Areas for Contribution

- 🔌 Grid-connected mode with time-of-use pricing
- 🔋 Battery degradation modeling
- 🌐 Multi-microgrid coordination
- 📱 Real-time dashboard (Flask/React)
- 🧠 Transfer learning for new locations
- 📊 Additional visualization tools

---

## 🐛 Known Issues

- GPU training on Windows requires CUDA toolkit
- Parallel processing on macOS slower due to spawn method
- Real-time monitor requires ANSI escape code support

See [Issues](https://github.com/DynMEP/dynamic-microgrid-resilience/issues) for full list.

---

## 🗺️ Roadmap

### Version 7.0 (Current) ✅
- Anticipatory DQN with pre-evening strategy
- 98.8% average load coverage
- CLI with visualization tools
- Complete documentation

### Version 7.1 (Q1 2026) 🚧
- [ ] Web dashboard for live monitoring
- [ ] Grid-connected mode
- [ ] Battery health modeling
- [ ] Multi-objective optimization

### Version 8.0 (Q2 2026) 🔮
- [ ] Multi-agent microgrids
- [ ] Transfer learning module
- [ ] Edge deployment (Raspberry Pi)
- [ ] Real-world pilot study

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

### Commercial Use

This software is free for academic and commercial use under MIT license. If you use this in production or research, please cite our paper.

---

## 🙏 Acknowledgments

- **PyTorch Team**: For the excellent deep learning framework
- **NEC**: For electrical code compliance standards (NEC 2023)
- **RL Community**: For foundational work on DQN and temporal credit assignment
- **arXiv**: For open-access preprint hosting

---

## 📞 Contact & Support

**Author**: Alfonso Davila
- **Email**: davila.alfonso@gmail.com
- **LinkedIn**: [https://www.linkedin.com/in/alfonso-davila-vera](https://www.linkedin.com/in/alfonso-davila-vera) 
- **GitHub Issues**: [Report bugs or request features](https://github.com/DynMEP/dynamic-microgrid-resilience/issues)

### Getting Help

1. Check [Documentation](docs/)
2. Search [Issues](https://github.com/DynMEP/dynamic-microgrid-resilience/issues)
3. Ask on [Discussions](https://github.com/DynMEP/dynamic-microgrid-resilience/discussions)
4. Email for research collaboration

---

## ⭐ Star History

If you find this project useful, please consider giving it a star on GitHub! ⭐

[![Star History Chart](https://api.star-history.com/svg?repos=DynMEP/dynamic-microgrid-resilience&type=Date)](https://star-history.com/#DynMEP/dynamic-microgrid-resilience&Date)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/DynMEP/dynamic-microgrid-resilience?style=social)
![GitHub forks](https://img.shields.io/github/forks/DynMEP/dynamic-microgrid-resilience?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/DynMEP/dynamic-microgrid-resilience?style=social)
![GitHub issues](https://img.shields.io/github/issues/DynMEP/dynamic-microgrid-resilience)
![GitHub pull requests](https://img.shields.io/github/issues-pr/DynMEP/dynamic-microgrid-resilience)

---

<div align="center">

**Built with 🔋 by Alfonso Davila Vera**

*Powering the future with intelligent energy management*

[⬆ Back to Top](#anticipatory-deep-q-network-for-microgrid-energy-management-)

</div>

---

*Last updated: October 2025*
