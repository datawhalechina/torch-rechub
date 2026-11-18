---
title: Installation Guide
description: Torch-RecHub installation instructions for stable and development versions
---

# Installation Guide

This document provides detailed installation instructions for Torch-RecHub, including stable and development versions.

## System Requirements

- Python 3.10 or later
- PyTorch `>=2.6,<3`

NumPy (1.26.4+), pandas (2.1+), and scikit-learn (1.4+) are installed with the package. The optional generative models use Transformers (`>=4.57.3,<5`).

## Installation Methods

For CPU training, install the stable release or use the source checkout below. For GPU or NPU training, first install the [PyTorch build for your device](#install-pytorch-for-your-device), then install Torch-RecHub in the same environment.

### Stable Version

```bash
python -m pip install torch-rechub
```

### Optional Features

The default installation includes the core training dependencies. Use the pip commands below for the release package. In a source checkout, run `uv sync --extra <name>`, for example `uv sync --extra visualization`.

| Use case | Command |
|---|---|
| Generative models | `python -m pip install "torch-rechub[generative]"` |
| Streaming Parquet data | `python -m pip install "torch-rechub[bigdata]"` |
| ONNX export and quantization | `python -m pip install "torch-rechub[onnx]"` |
| Model visualization | `python -m pip install "torch-rechub[visualization]"` |
| Experiment tracking | `python -m pip install "torch-rechub[tracking]"` |
| Vector search | `python -m pip install "torch-rechub[annoy]"` (choose `annoy`, `faiss`, or `milvus` for your backend) |
| Testing, formatting, and type checking | `python -m pip install "torch-rechub[dev]"` |
| All optional features | `python -m pip install "torch-rechub[all]"` |

Rendering model graphs also requires the Graphviz `dot` executable on your system path.

You can run a script with an extra using `uv run --extra <name> python <script.py>`. Repeat `--extra` if the script needs several extras.

### Latest Development Version

To install the development version with the latest features:

```bash
# Install uv first (if not already installed)
python -m pip install uv

# Clone and install
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub

# Create .venv, install locked dependencies, and install this project editable
uv sync
```

If you have already installed a device-specific PyTorch build, use `uv pip install -e .` for the last step and run examples with the `python` from that activated environment.

## Development Environment Setup

If you want to contribute to Torch-RecHub or work with the source code:

```bash
# 1. Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. Install dependencies and set up environment
uv sync

# 3. Run tests (uv sync already installs this project editable)
uv run pytest
```

## Install PyTorch for Your Device

For NVIDIA GPUs, choose a CUDA build supported by your GPU and driver. For AMD GPUs, choose a ROCm build for your GPU architecture. For Huawei Ascend NPUs, install and load CANN, then select compatible versions of PyTorch and `torch-npu`. Check the [PyTorch selector](https://pytorch.org/get-started/locally/), [AMD ROCm documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/), or [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md) for your configuration.

Activate your Python environment and choose one command for your device. With uv, you can use `uv pip` in place of `python -m pip`.

```bash
# CPU-only build on Linux or Windows
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU, for example with CUDA 12.6
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# AMD GPU, for example a Ryzen AI Max+ 395 (gfx1151)
python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio

# Huawei Ascend NPU, when the default packages match your CANN version
python -m pip install torch torch-npu
```

To train on the first device, set `device="cuda:0"` in the trainer for NVIDIA or AMD. For Ascend, import `torch_npu` and set `device="npu:0"`.

## Verify Installation

Check the version in the environment where you installed Torch-RecHub.

```bash
python -c "import torch_rechub; print(torch_rechub.__version__)"
```

## Troubleshooting

### Common Issues

If you encounter any installation issues:
1. Check [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. Create a new Issue with detailed error messages and system information
3. Refer to the [FAQ](/community/faq)
