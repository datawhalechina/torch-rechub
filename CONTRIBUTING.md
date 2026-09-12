# Contributing Guide

Thank you for your interest in contributing to Torch-RecHub! We welcome all types of contributions, including but not limited to:

- 🐛 Bug reports
- 💡 Feature suggestions
- 📝 Documentation improvements
- 🔧 Code contributions
- 🧪 Test cases
- 📖 Tutorials and examples

## 🚀 Getting Started

### Development Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. Install dependencies and setup environment
uv sync

# 3. Install the package in development mode
uv pip install -e .
```

### Development Workflow

1. **Fork the repository:** Click the "Fork" button in the upper right corner.
2. **Make your changes:** Implement new features or fix bugs.
3. **Format code:** Run code formatting before committing to ensure consistent code style:
   ```bash
   python config/format_code.py
   ```
4. **Commit changes:** `git commit -m "feat: add new feature"` or `fix: fix some issue"` (Following [Conventional Commits](https://www.conventionalcommits.org/) is preferred).
5. **Push to branch:** `git push origin`
6. **Create Pull Request:** Go back to the original repository page, click "New pull request", compare your branch with the `main` branch of the main repository, and submit PR.

## 📋 Code Standards

### Branch Naming

- `feature/feature-name` - for new features
- `fix/bug-description` - for bug fixes
- `docs/documentation-update` - for documentation changes
- `test/test-description` - for test additions

### Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat: add new recommendation model`
- `fix: resolve memory leak in training loop`
- `docs: update installation guide`
- `test: add unit tests for DeepFM model`
- `refactor: optimize data loading pipeline`

### Pull Request Process

1. **Push your branch**
   ```bash
   git push origin your-branch-name
   ```

2. **Create Pull Request**
   - Visit the GitHub repository page
   - Click "New pull request"
   - Select your branch
   - Fill out the PR template

3. **PR Requirements**
   - Clear description of changes
   - Explanation of why changes are needed
   - List related issues (if any)
   - Include testing instructions
   - Add screenshots (if applicable)

## 🧪 Testing Guidelines

### Writing Tests

- **Unit Tests**: Test individual functions or classes
- **Integration Tests**: Test interactions between modules
- **End-to-End Tests**: Test complete workflows

### Test Structure

```
tests/
├── test_models/
│   ├── test_ranking.py
│   ├── test_matching.py
│   └── test_multi_task.py
├── test_trainers/
├── test_utils/
└── test_e2e/
```

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_models/test_ranking.py

# Run with coverage
uv run pytest --cov=torch_rechub
```

### Test Example

```python
import pytest
from torch_rechub.models.ranking import DeepFM

def test_deepfm_forward():
    """Test DeepFM forward pass"""
    model = DeepFM(
        deep_features=deep_features,
        fm_features=fm_features,
        mlp_params={"dims": [128, 64]}
    )
    
    output = model(sample_input)
    assert output.shape == expected_shape
```

## 📝 Documentation

### Documentation Types

- **API Documentation**: Docstrings in code
- **User Guides**: Files in `docs/` directory
- **Tutorials**: Jupyter notebooks in `tutorials/` directory
- **README**: Project introduction and quick start

### Documentation Standards

- Use Markdown format
- Include code examples
- Provide clear step-by-step instructions
- Keep both English and Chinese versions synchronized
- Follow scikit-learn style docstrings (NumPy/SciPy convention) for Python code

### Docstring Example

```python
def train_model(model, data_loader, optimizer):
    """Train a recommendation model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to train.
    data_loader : DataLoader
        Training data loader.
    optimizer : torch.optim.Optimizer
        Optimizer for training.

    Returns
    -------
    float
        Training loss.

    Examples
    --------
    >>> model = DeepFM(features, mlp_params)
    >>> loss = train_model(model, train_loader, optimizer)
    """
    # Implementation here
```

## 🎯 Contribution Ideas

### Good First Issues

- 📖 Improve documentation and comments
- 🧪 Add test cases
- 🐛 Fix simple bugs
- 📝 Translate documentation
- 💡 Add example code
- 🔧 Code formatting and style improvements

### Advanced Contributions

- 🚀 Implement new recommendation algorithms
- ⚡ Performance optimizations
- 🏗️ Architecture improvements
- 📊 Add new evaluation metrics
- 🛠️ Development tools and scripts
- 🔬 Research paper implementations

### Model Implementation Guidelines

When implementing new models:

1. **Follow existing patterns**: Look at existing models for structure
2. **Add comprehensive tests**: Include unit tests and integration tests
3. **Provide examples**: Add usage examples in `examples/` directory
4. **Document thoroughly**: Include docstrings and README updates
5. **Benchmark performance**: Compare with existing implementations

## 📞 Getting Help

If you encounter issues during contribution:

1. **Check existing Issues**: There might be related discussions
2. **Create new Issue**: Describe your problem clearly
3. **Join discussions**: Ask questions in relevant Issues or PRs
4. **Contact maintainers**: Through GitHub or email
5. **Check documentation**: Review the [FAQ](docs/en/community/faq.md) for common questions and troubleshooting

## 🏆 Recognition

We value every contribution! All contributors will be recognized in:

- Contributors list in README
- Release notes acknowledgments
- Project documentation contributor pages
- Special mentions for significant contributions

## 📜 Code of Conduct

Please follow our [Code of Conduct](CODE_OF_CONDUCT.md) to ensure a friendly and inclusive community environment.

## 🔄 Release Process

Project maintainers regularly release new versions:

1. **Version Planning**: Discussed in Issues
2. **Feature Freeze**: Stop adding new features
3. **Testing**: Comprehensive testing phase
4. **Release Preparation**: Update documentation and version numbers
5. **Official Release**: Publish to PyPI

## 📋 Review Process

All contributions go through a review process:

1. **Automated Checks**: CI/CD pipeline runs tests
2. **Code Review**: Maintainers review code quality
3. **Testing**: Verify functionality works as expected
4. **Documentation**: Ensure documentation is updated
5. **Approval**: At least one maintainer approval required

---

Thank you again for your contribution! Every contribution makes Torch-RecHub better. 🎉