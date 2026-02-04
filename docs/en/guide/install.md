---
title: Installation Guide
description: Torch-RecHub installation instructions for stable and development versions
---

# Installation Guide

This document provides detailed installation instructions for Torch-RecHub, including stable and development versions.

## System Requirements

Before installing Torch-RecHub, ensure your system meets the following requirements:

- **Python 3.9+**
- **PyTorch 1.7+** (CUDA version recommended for GPU acceleration)
- **NumPy**
- **Pandas**
- **SciPy**
- **Scikit-learn**

## Installation Methods

### Stable Version (Recommended)

The simplest way to install is via pip:

```bash
pip install torch-rechub
```

### Latest Development Version

To install the development version with the latest features:

```bash
# Install uv first (if not already installed)
pip install uv

# Clone and install
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub
uv sync
```

## Development Environment Setup

If you want to contribute to Torch-RecHub or work with the source code:

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. Install dependencies and set up environment
uv sync

# 3. Install package in development mode
uv pip install -e .
```

## Verify Installation

To verify that Torch-RecHub is correctly installed:

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

If you need to install PyTorch with a specific CUDA version, visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for installation instructions tailored to your system.

### GPU Support

For GPU acceleration, ensure you have:
- NVIDIA GPU with compute capability 3.5 or higher
- CUDA Toolkit installed
- cuDNN library installed

### Common Issues

If you encounter any installation issues:
1. Check [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. Create a new Issue with detailed error messages and system information
3. Refer to the FAQ section
