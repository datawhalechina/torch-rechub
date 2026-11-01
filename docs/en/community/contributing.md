---
title: Contributing Guide
description: Torch-RecHub project contribution guide, including bug reports, feature requests, and pull request process
---

# Contributing Guide

Thank you for your interest in Torch-RecHub! We welcome all forms of contributions, including but not limited to:

- Bug reports
- Feature suggestions
- Documentation improvements
- Code contributions
- Test cases
- Tutorials and examples

## Quick Start

### Development Environment Setup

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. Create the environment and install development plus all optional dependencies
# uv sync installs the current project editable; no separate pip install -e . is needed
uv sync --all-extras

# 3. Create a working branch
git switch -c docs/update-guide
```

### Development Workflow

1. **Fork the repository**: Click the "Fork" button in the top right corner.
2. **Create a branch and make changes**: Implement new features or fix bugs.
3. **Format code**: Run code formatting before committing to ensure consistent style:
   ```bash
   uv run python config/format_code.py
   ```
4. **Commit changes**: `git commit -m "feat: add new feature"` or `git commit -m "fix: fix some issue"` (following [Conventional Commits](https://www.conventionalcommits.org/) is recommended).
5. **Push to branch**: `git push -u origin your-branch-name`
6. **Create Pull Request**: Return to the original repository page, click "New pull request", compare your branch with the main repository's `main` branch, then submit the PR.

## Code Standards

### Branch Naming

- `feature/feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/update-install-guide` - Documentation updates (this is a Git branch name)
- `test/test-description` - Test additions

### Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat: add new recommendation model`
- `fix: resolve memory leak in training loop`
- `docs: update installation guide`
- `test: add unit tests for DeepFM model`
- `refactor: optimize data loading pipeline`

### Pull Request Process

1. **Push your branch**
   ```bash
   git push -u origin your-branch-name
   ```

2. **Create Pull Request**
   - Visit the GitHub repository page
   - Click "New pull request"
   - Select your branch
   - Fill in the PR template

3. **PR Requirements**
   - Clear description of changes
   - Explain why these changes are needed
   - List related Issues (if any)
   - Include test instructions
   - Add screenshots (if applicable)

## Testing Guide

### Writing Tests

- **Unit tests**: Test individual functions or classes
- **Integration tests**: Test interactions between modules
- **End-to-end tests**: Test complete workflows

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_e2e_ranking.py

# Run with coverage report
uv run pytest --cov=torch_rechub
```

Default tests do not start external services. After starting a local Milvus service, run its integration tests explicitly:

```powershell
$env:RUN_MILVUS_TESTS = "1"
uv run --extra milvus pytest tests/test_serving.py -m integration
```

## Documentation

### Documentation Types

- **API documentation**: Docstrings in code
- **User guides**: Files in `docs/` directory
- **Tutorials**: Jupyter notebooks in `tutorials/` directory
- **README**: Project introduction and quick start

### Documentation Standards

- Use Markdown format
- Include code examples
- Provide clear step-by-step instructions
- Keep English and Chinese versions in sync
- Use the project's existing NumPy/SciPy docstring style, with sections such as `Parameters` and `Returns`

### Build Documentation Locally

```bash
npm ci
npm run docs:build
```

For documentation changes, run `npm run docs:build` at least once and keep corresponding pages under `docs/zh` and `docs/en` synchronized.

## Contribution Ideas

### Beginner-Friendly Tasks

- Improve documentation and comments
- Add test cases
- Fix simple bugs
- Translate documentation
- Add example code
- Code formatting and style improvements

### Advanced Contributions

- Implement new recommendation algorithms
- Performance optimization
- Architecture improvements
- Add new evaluation metrics
- Develop tools and scripts
- Research paper implementations

### Model Implementation Guide

When implementing new models:

1. **Follow existing patterns**: Look at the structure of existing models
2. **Add comprehensive tests**: Include unit and integration tests
3. **Provide examples**: Add usage examples in `examples/` directory
4. **Detailed documentation**: Include docstrings and README updates
5. **Performance benchmarks**: Compare with existing implementations

## Getting Help

If you encounter issues during contribution:

1. **Check existing Issues**: There may be related discussions
2. **Create new Issue**: Clearly describe your problem
3. **Join discussions**: Ask questions in related Issues or PRs
4. **Contact maintainers**: Via GitHub or email

## Recognition

We value every contribution! All contributors will be recognized in:

- Contributors list in README
- Acknowledgments in release notes
- Special mentions for significant contributions

## Code of Conduct

Please follow our [Code of Conduct](https://github.com/datawhalechina/torch-rechub/blob/main/CODE_OF_CONDUCT.md) to ensure a friendly and inclusive community environment.

---

Thank you again for your contribution! Every contribution makes Torch-RecHub better.

