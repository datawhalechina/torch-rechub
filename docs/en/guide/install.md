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
# Install PyTorch matching your device
pip install torch                                                     # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU (CUDA 12.1)
pip install torch torch-npu                                           # NPU (Huawei Ascend, requires torch-npu >= 2.5.1)

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

# Install PyTorch matching your device
uv pip install torch                                                     # CPU
uv pip install torch --index-url https://download.pytorch.org/whl/cu121  # GPU (CUDA 12.1)
uv pip install torch torch-npu                                           # NPU (Huawei Ascend, requires torch-npu >= 2.5.1)

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
# cd into the script directory first (scripts use relative data paths)
cd examples/matching
python run_ml_dssm.py
```

## Troubleshooting

### PyTorch Installation

If you need to install PyTorch with a specific CUDA version, visit the [PyTorch official website](https://pytorch.org/get-started/locally/) for installation instructions tailored to your system.

### GPU Support

For GPU acceleration, ensure you have:
- NVIDIA GPU with compute capability 3.5 or higher
- CUDA Toolkit installed
- cuDNN library installed

### NPU Support (Huawei Ascend)

Torch-RecHub supports Huawei Ascend NPU devices, tested on **Huawei Ascend 910B**.

Please install the Ascend-compatible versions of PyTorch and torch-npu. For version compatibility details, refer to the [Ascend PyTorch repository](https://gitcode.com/Ascend/pytorch).

After installation, import `torch_npu` in your code, then specify the device in the Trainer:

```python
import torch
import torch_npu

trainer = CTRTrainer(model, device='npu:0')
```

### Common Issues

If you encounter any installation issues:
1. Check [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. Create a new Issue with detailed error messages and system information
3. Refer to the FAQ section
