---
title: Installation Guide
description: Detailed installation instructions for Torch-RecHub including stable and development versions
---

# Installation Guide

This document provides detailed installation instructions for Torch-RecHub, including both stable and development versions.

## Requirements

Before installing Torch-RecHub, ensure you have the following prerequisites:

- **Python 3.9+**
- **PyTorch 1.7+** (CUDA-enabled version recommended for GPU acceleration)
- **NumPy**
- **Pandas**
- **SciPy**
- **Scikit-learn**

## Installation Methods

### Stable Release (Recommended for Users)

The easiest way to install Torch-RecHub is via pip:

```bash
pip install torch-rechub
```

### Latest Development Version

To install the latest development version with the most recent features:

```bash
# Install uv first (if not already installed)
pip install uv

# Clone and install
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub
uv sync
```

## Development Setup

If you want to contribute to Torch-RecHub or work with the source code:

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. Install dependencies and setup environment
uv sync

# 3. Install the package in development mode
uv pip install -e .
```

## Verification

To verify that Torch-RecHub is installed correctly, you can run:

```python
import torch_rechub
print(torch_rechub.__version__)
```

Or run a simple example:

```bash
python examples/matching/run_ml_dssm.py
```

## Troubleshooting

### PyTorch Installation

If you need to install PyTorch with specific CUDA support, visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for installation instructions tailored to your system.

### GPU Support

For GPU acceleration, ensure you have:
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- CUDA Toolkit installed
- cuDNN library installed

### Common Issues

If you encounter any installation issues, please:
1. Check the [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. Create a new issue with detailed error messages and system information
3. Refer to the [FAQ](/manual/faq) section

