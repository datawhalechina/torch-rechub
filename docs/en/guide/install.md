---
title: Installation
description: Install Torch-RecHub from PyPI or from source and check that PyTorch can use your device.
---

# Installation

Install from source to follow the tutorials with the sample datasets in the repository. To use Torch-RecHub in an existing training project, install the PyPI package and import `torch_rechub`.

## Requirements

- Python 3.10 or later
- PyTorch `>=2.6,<3`

The package installs NumPy (1.26.4 or later), pandas (2.1 or later) and scikit-learn (1.4 or later). Generative models also use the optional Transformers dependency (`>=4.57.3,<5`).

## Install Torch-RecHub

On a CPU, choose either the release package or a source installation. For a GPU or NPU, first install a compatible PyTorch build in the environment where you will run Torch-RecHub. [PyTorch device examples](#install-pytorch-for-your-device)

### In an existing project

```bash
python -m pip install torch-rechub
```

### Optional features

The base installation can train models such as DeepFM and DSSM. Add an extra when you need to draw a diagram, export to ONNX or search an index. The commands below install the release package. In a source checkout, use the extra name with uv, for example `uv sync --extra visualization`.

| Feature | Installation command |
| --- | --- |
| Generative models | `python -m pip install "torch-rechub[generative]"` |
| Parquet streaming | `python -m pip install "torch-rechub[bigdata]"` |
| ONNX export and quantization | `python -m pip install "torch-rechub[onnx]"` |
| Model diagrams | `python -m pip install "torch-rechub[visualization]"` |
| Experiment tracking | `python -m pip install "torch-rechub[tracking]"` |
| Vector search | `python -m pip install "torch-rechub[annoy]"`. Choose `annoy`, `faiss` or `milvus` to match your backend. |
| Tests, formatters and type checking | `python -m pip install "torch-rechub[dev]"` |
| All optional features | `python -m pip install "torch-rechub[all]"` |

To render model diagrams, install Graphviz on the system as well. The `dot` executable must be on your path.

For a single script, you can install its extras as part of the command with `uv run --extra <name> python <script.py>`. Repeat `--extra` when a script needs more than one group.

### From source

The checkout includes example scripts and sample data. These commands download it and create a project environment in `.venv`.

```bash
# Install uv if it is not already available.
python -m pip install uv

# Clone the repository.
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub

# Create .venv, install locked dependencies and install the project in editable mode.
uv sync
```

If your active environment already has a device-specific PyTorch build, use `uv pip install -e .` in place of the final `uv sync`. Run the examples with that environment's `python`.

## Set up a contribution

To submit a pull request, fork the repository on GitHub and clone your fork. Replace `YOUR_USERNAME` below with your GitHub username.

```bash
# 1. Fork the project and clone your fork.
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. Install dependencies in the project environment.
uv sync

# 3. Run the tests against the editable installation.
uv run pytest
```

## Install PyTorch for your device

Choose the build for your hardware and driver.

- **NVIDIA GPU.** Select a CUDA build compatible with the installed driver. [PyTorch installation guide](https://pytorch.org/get-started/locally/)
- **AMD GPU.** Find the ROCm packages for your GPU model. [AMD ROCm documentation](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)
- **Huawei Ascend NPU.** Install and load CANN, then choose the matching PyTorch and `torch-npu` versions. [Ascend compatibility table](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)

The commands below illustrate one setup for each device. Run the one that matches your machine in the active Python environment. With uv, substitute `uv pip` for `python -m pip`.

```bash
# CPU-only build for Linux or Windows
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU example using CUDA 12.6
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# AMD GPU example for the Ryzen AI Max+ 395 (gfx1151)
python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio

# Huawei Ascend NPU, if the default package versions match the installed CANN version
python -m pip install torch torch-npu
```

Use `device="cuda:0"` in the trainer for the first NVIDIA or AMD GPU. For the first Ascend NPU, import `torch_npu` and use `device="npu:0"`.

## Verify the installation

Print the package version from the environment where you installed Torch-RecHub.

```bash
python -c "import torch_rechub; print(torch_rechub.__version__)"
```

If you use an accelerator, choose the command for your device and check that PyTorch can create a tensor on it.

```bash
# NVIDIA CUDA or AMD ROCm
python -c "import torch; print(torch.zeros(1, device='cuda:0').device)"

# Huawei Ascend: load the CANN environment first.
python -c "import torch, torch_npu; print(torch.zeros(1, device='npu:0').device)"
```

If Python cannot find `torch_rechub` after a source installation, run `uv run python` from the repository root to use the project environment. If the device check fails, look up the full error together with your accelerator model, driver or CANN version, and PyTorch version. [FAQ](/community/faq) · [GitHub issues](https://github.com/datawhalechina/torch-rechub/issues)
