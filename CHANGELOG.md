# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Publication-ready preprint `paper/arxiv_preprint_final_v6.pdf` (June 2026): 21 pages, four figures, embedded bibliography.
- ORCID (0009-0001-3521-3802) added to the paper author block.

### Changed
- Reframed contribution as a forecast-free, off-grid formulation; added closest prior art to Related Work (Muriithi & Chowdhury 2022; tariff-aware dual-layer Q-learning).
- Clarified that battery discharge is a standard load-following rule with an SOC floor, not a separate contribution; the gain comes from anticipatory charging.

### Fixed
- Synced repo metadata to the tempered framing: README BibTeX (now `@misc`, 2026, "Near-100%", preprint note), `CITATION.cff` (v7.1.0, 2026-06-07, preprint reference), and METHODOLOGY wording.
- Paper bibliography now compiles (replaced missing `\bibliography{references}` with an embedded `thebibliography`); resolved an undefined citation key and a broken figure reference.

### Planned
- Grid-connected mode with time-of-use pricing
- Battery degradation modeling
- Multi-objective optimization (Pareto frontier)
- Web dashboard for real-time monitoring
- Transfer learning module

## [7.0.0] - 2025-10-19

### 🎉 Major Release: Anticipatory DQN

This release represents a complete reimplementation with anticipatory deep reinforcement learning.

### Added
- **Anticipatory DQN Algorithm**
  - Time-to-event state augmentation with cyclical hour encoding
  - Hierarchical reward shaping with extreme evening penalties (-400)
  - Multi-phase temporal structure (daytime, pre-evening, evening, night)
  - 7-feature enhanced state space
  - 5 granular action space (charge/discharge at 50%/100%, hold)

- **Performance Achievements**
  - 98.8% average load coverage across 100 test scenarios
  - Best policy: 100% coverage (found at episode 100)
  - 99.2% evening reliability (vs 71.5% baseline)
  - 0.13 kWh average unmet energy
  - 86% scenarios achieve perfect 100% coverage

- **Enhanced CLI**
  - `demo` command for quick demonstration
  - `train` command with full configuration options
  - `evaluate` command with auto-latest model loading
  - `compare` command with parallel processing
  - `status` command for system check
  - Real-time training monitor with `--monitor` flag
  - Automatic visualization generation with `--plot` flag
  - Progress bars with tqdm integration

- **Visualization System**
  - Publication-quality plots (300 DPI PNG/PDF)
  - Training progress analysis (4 panels)
  - Hourly performance visualization (4 panels)
  - Evaluation results (load coverage, performance by type)
  - Economic analysis (cost breakdown, 20-year cash flow)

- **Best Policy Tracking**
  - Automatic saving of optimal weights during training
  - Validation every 100 episodes
  - Early stopping when optimal policy found
  - Performance score: 100 * LoadMet - 1000 * Unmet

- **Stochastic Environment**
  - Beta-distributed cloud cover for solar
  - Weibull-distributed wind speed
  - Three load profile types (residential, commercial, mixed)
  - ±10% load variability

- **Documentation**
  - Comprehensive README with badges
  - arXiv paper integration
  - Complete API documentation
  - Usage examples and tutorials
  - Contributing guidelines

- **Economic Analysis**
  - Capital cost breakdown by component
  - 20-year NPV calculation (5% discount rate)
  - ROI and IRR computation
  - Payback period analysis
  - Annual savings projection

- **Testing & Validation**
  - 100-scenario test suite
  - Ablation studies
  - Multi-seed convergence analysis
  - Battery capacity comparison
  - Baseline method comparisons

### Changed
- **Battery Capacity**: Increased from 13 kWh to 18 kWh (+38%)
  - Enables 100% evening coverage
  - Improves economic viability (4.3-year payback vs 4.8)
  
- **State Space**: Expanded from 4 to 7 features
  - Added cyclical hour encoding (sin/cos)
  - Added renewable ratio feature
  - Added net energy balance
  - Added evening period flag
  - Added pre-evening preparation flag

- **Action Space**: Refined from 3 to 5 actions
  - Split charge/discharge into 50%/100% power levels
  - Enables more granular control

- **Reward Function**: Hierarchical multi-phase structure
  - Base: +100 met, -300 unmet
  - Pre-evening: +200 if SOC≥75%, -250 otherwise
  - Evening: +250 bonus, -400 catastrophic penalty
  - Daytime: +50 for charging when SOC<80%

- **Discount Factor**: Increased from 0.95 to 0.97
  - Better long-term planning
  - Improved evening anticipation

- **Learning Rate**: Decreased from 0.001 to 0.0002
  - More stable convergence
  - Better final performance

### Performance Improvements
- **Training Speed**: 21 minutes for 1,100 episodes (laptop CPU)
- **Inference Speed**: <1 ms per decision (real-time capable)
- **Memory Usage**: 450 MB (replay buffer optimized)
- **Convergence**: Optimal policy at episode 100 (10x faster than v3)

### Fixed
- Battery SOC calculation precision issues
- Evening period reward timing
- Validation scenario consistency
- Plot generation memory leaks
- CSV export formatting

### Technical Debt Addressed
- Refactored monolithic code into modular components
- Improved error handling and logging
- Added type hints throughout
- Comprehensive docstrings
- Unit test coverage

## [3.0.0] - 2025-09-15

### Added
- Enhanced Q-Learning with best policy tracking
- 13 kWh battery configuration
- Basic validation framework
- CSV results export

### Changed
- Improved reward structure
- Better exploration strategy
- State space from 3 to 4 features

### Performance
- 92-96% load coverage
- Convergence around episode 500

## [2.0.0] - 2025-08-20

### Added
- Optimized Q-Learning parameters
- 13 kWh battery (increased from 10 kWh)
- Basic plotting capabilities

### Changed
- Learning rate tuning
- Epsilon decay adjustment

### Performance
- 85-90% load coverage
- Better evening performance than v1

## [1.0.0] - 2025-07-10

### Added
- Baseline Q-Learning implementation
- 10 kWh battery system
- Solar + wind generation
- Basic load profiles
- Simple reward function

### Performance
- 70-80% load coverage
- Proof of concept

## Comparison Across Versions

| Version | Battery | Algorithm | Load Coverage | Evening Coverage | Convergence |
|---------|---------|-----------|---------------|------------------|-------------|
| v1.0    | 10 kWh  | Q-Learning | 70-80% | 60-70% | ~1000 eps |
| v2.0    | 13 kWh  | Q-Learning | 85-90% | 75-85% | ~800 eps |
| v3.0    | 13 kWh  | Enhanced Q | 92-96% | 85-90% | ~500 eps |
| **v7.0** | **18 kWh** | **Anticipatory DQN** | **98.8%** | **99.2%** | **~100 eps** |

## Migration Guide

### From v3.0 to v7.0

**Breaking Changes:**
- State space changed from 4 to 7 features
- Action space changed from 3 to 5 actions
- Config class restructured

**Migration Steps:**
```python
# Old (v3.0)
from microgrid_v3 import train_qlearning
results = train_qlearning(episodes=1000)

# New (v7.0)
from Dynamic_Microgrid_Resilience_v7 import MicrogridConfig, DQNAgent, train_agent
config = MicrogridConfig()
agent = DQNAgent(config)
results = train_agent(agent, episodes=1000)
```

**Models:** v3.0 models are not compatible with v7.0. Retrain from scratch.

## Acknowledgments

### Version 7.0 Contributors
- Alfonso Davila - Core algorithm, implementation, documentation

### Research Support
- PyTorch team for the deep learning framework
- NEC for electrical code standards
- RL community for DQN innovations

## Links

- **Repository**: https://github.com/DynMEP/dynamic-microgrid-resilience
- **Issues**: https://github.com/DynMEP/dynamic-microgrid-resilience/issues
- **Discussions**: https://github.com/DynMEP/dynamic-microgrid-resilience/discussions

---

**Note**: Version numbers follow semantic versioning (MAJOR.MINOR.PATCH)
- MAJOR: Incompatible API changes
- MINOR: Backward-compatible functionality
- PATCH: Backward-compatible bug fixes
