# Contributing to Dynamic Microgrid Resilience

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## 🎯 Ways to Contribute

### 1. Report Bugs
- Use the [GitHub Issues](https://github.com/DynMEP/dynamic-microgrid-resilience/issues) page
- Check if the issue already exists
- Include: OS, Python version, error message, minimal reproducible example

### 2. Suggest Features
- Open a [Feature Request](https://github.com/DynMEP/dynamic-microgrid-resilience/issues/new)
- Describe the feature and its use case
- Explain why it would be valuable

### 3. Improve Documentation
- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve API documentation

### 4. Submit Code
- Bug fixes
- New features
- Performance improvements
- Additional tests

## 🔧 Development Setup

### Prerequisites
- Python 3.8+
- Git
- Virtual environment tool (venv or conda)

### Setup Instructions

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/dynamic-microgrid-resilience.git
cd dynamic-microgrid-resilience

# 3. Add upstream remote
git remote add upstream https://github.com/DynMEP/dynamic-microgrid-resilience.git

# 4. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 5. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies

# 6. Create a branch for your changes
git checkout -b feature/your-feature-name
```

## 📝 Code Style

### Python Style Guide
- Follow [PEP 8](https://pep8.org/)
- Use type hints where appropriate
- Maximum line length: 100 characters
- Use descriptive variable names

### Example

```python
def calculate_battery_soc(
    energy_kwh: float,
    capacity_kwh: float,
    min_soc: float = 0.2
) -> float:
    """
    Calculate battery state of charge.
    
    Args:
        energy_kwh: Current energy in battery (kWh)
        capacity_kwh: Maximum battery capacity (kWh)
        min_soc: Minimum allowed SOC (default 0.2)
    
    Returns:
        State of charge as fraction [0, 1]
    """
    soc = energy_kwh / capacity_kwh
    return max(min_soc, min(1.0, soc))
```

### Code Formatting
```bash
# Run black formatter
black .

# Run flake8 linter
flake8 .

# Run isort for imports
isort .
```

## 🧪 Testing

### Run Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_agent.py

# Run with verbose output
pytest -v
```

### Writing Tests
- Add tests for new features
- Maintain test coverage > 80%
- Use descriptive test names
- Include edge cases

Example test:
```python
def test_battery_charging():
    """Test battery charges correctly with available power"""
    config = MicrogridConfig(battery_capacity_kwh=18.0)
    battery = BatterySystem(config)
    
    # Test charging
    initial_soc = battery.get_SOC_percent()
    charged = battery.charge(power_kw=6.0, duration_hours=1.0)
    
    assert charged > 0
    assert battery.get_SOC_percent() > initial_soc
```

## 📤 Submitting Changes

### Pull Request Process

1. **Update Documentation**
   - Update README.md if needed
   - Add docstrings to new functions
   - Update CHANGELOG.md

2. **Run Tests**
   ```bash
   pytest
   flake8 .
   ```

3. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: Add new feature description"
   ```
   
   Use conventional commit messages:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Test changes
   - `refactor:` Code refactoring
   - `perf:` Performance improvements

4. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

5. **Create Pull Request**
   - Go to your fork on GitHub
   - Click "New Pull Request"
   - Fill in the PR template
   - Link related issues

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Tests pass locally
- [ ] New tests added
- [ ] Documentation updated

## Related Issues
Closes #123
```

## 🎨 Areas for Contribution

### High Priority
1. **Grid-connected mode**: Time-of-use pricing integration
2. **Battery degradation**: Capacity fade modeling
3. **Multi-objective optimization**: Pareto frontier for cost vs reliability
4. **Real-time dashboard**: Flask/React web interface

### Medium Priority
1. **Transfer learning**: Adapt to new locations
2. **Multi-agent systems**: Neighborhood microgrids
3. **Additional algorithms**: PPO, SAC, A3C comparisons
4. **Mobile app**: iOS/Android monitoring

### Documentation
1. Video tutorials
2. Interactive Jupyter notebooks
3. Case studies
4. Deployment guides

## 🐛 Bug Report Template

```markdown
**Describe the bug**
Clear description of the bug

**To Reproduce**
Steps to reproduce:
1. Run command '...'
2. See error

**Expected behavior**
What you expected to happen

**Environment:**
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.8.10]
- PyTorch version: [e.g. 2.0.1]

**Additional context**
Error messages, screenshots, etc.
```

## 💡 Feature Request Template

```markdown
**Feature Description**
Clear description of the proposed feature

**Use Case**
Why this feature would be valuable

**Proposed Solution**
How you envision implementing this

**Alternatives Considered**
Other approaches you've thought about
```

## 📋 Code Review Process

### For Contributors
- Respond to reviewer comments promptly
- Make requested changes
- Mark conversations as resolved when addressed

### For Reviewers
- Be constructive and respectful
- Focus on code quality, not style preferences
- Approve when requirements are met

## 🎓 Learning Resources

### Reinforcement Learning
- [Spinning Up in Deep RL](https://spinningup.openai.com/)
- [Deep RL Course (Berkeley)](http://rail.eecs.berkeley.edu/deeprlcourse/)
- [RL Book (Sutton & Barto)](http://incompleteideas.net/book/the-book.html)

### Energy Systems
- [NEC 2023](https://www.nfpa.org/codes-and-standards/all-codes-and-standards/list-of-codes-and-standards/detail?code=70)
- [Microgrid Resources](https://building-microgrid.lbl.gov/)

### PyTorch
- [Official Tutorials](https://pytorch.org/tutorials/)
- [DQN Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

## 🤝 Community Guidelines

### Be Respectful
- Welcome newcomers
- Be patient with questions
- Provide constructive feedback
- Respect different perspectives

### Communication
- Use GitHub Discussions for questions
- Use Issues for bugs and features
- Keep conversations on-topic
- Be clear and concise

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in the paper (for major contributions)

## 📞 Questions?

- **General Questions**: [GitHub Discussions](https://github.com/DynMEP/dynamic-microgrid-resilience/discussions)
- **Bug Reports**: [GitHub Issues](https://github.com/DynMEP/dynamic-microgrid-resilience/issues)
- **Email**: davila.alfonso@gmail.com

Thank you for contributing! 🎉
