---
title: 安装指南
description: Torch-RecHub 详细安装说明，包括稳定版和最新版安装步骤
---

# 安装指南

本文档提供了 Torch-RecHub 的详细安装说明，包括稳定版和最新开发版的安装步骤。

## 系统要求

在安装 Torch-RecHub 之前，请确保您的系统满足以下要求：

- **Python 3.9+**
- **PyTorch 1.10+**（根据硬件选择 CPU、NVIDIA CUDA、AMD ROCm 或华为昇腾 NPU 版本）
- **NumPy**
- **Pandas**
- **SciPy**
- **Scikit-learn**

## 安装方式

PyTorch 与硬件、驱动和运行时版本强相关。安装前建议先查看对应官方适配文档：[NVIDIA CUDA / PyTorch 版本](https://pytorch.org/get-started/previous-versions/)、[Huawei Ascend NPU / PyTorch 版本](https://www.hiascend.com/document/detail/zh/Pytorch/2600/releasenote/docs/zh/release_notes/release_notes.md)、[AMD ROCm / PyTorch 版本](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)。

### 稳定版（推荐用户使用）

最简单的安装方式是通过 pip：

```bash
# 根据设备选择对应的 PyTorch 版本，只需安装其中一种
pip install torch                                                     # CPU
pip install torch --index-url https://download.pytorch.org/whl/cu121  # NVIDIA GPU (CUDA 12.1)
pip install torch torch-npu                                           # Huawei Ascend NPU (需要 torch-npu >= 2.5.1)
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio  # AMD GPU (ROCm, gfx1151 = Ryzen AI Max+ 395/390/385)

pip install torch-rechub
```

### 最新开发版

要安装包含最新功能的开发版本：

```bash
# 首先安装 uv（如果尚未安装）
pip install uv

# 克隆并安装
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub

# 根据设备选择对应的 PyTorch 版本，只需安装其中一种
uv pip install torch                                                     # CPU
uv pip install torch --index-url https://download.pytorch.org/whl/cu121  # NVIDIA GPU (CUDA 12.1)
uv pip install torch torch-npu                                           # Huawei Ascend NPU (需要 torch-npu >= 2.5.1)
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio  # AMD GPU (ROCm, gfx1151 = Ryzen AI Max+ 395/390/385)

uv sync
```

## 开发环境设置

如果您想为 Torch-RecHub 做出贡献或使用源代码：

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. 安装依赖并设置环境
uv sync

# 3. 以开发模式安装包
uv pip install -e .
```

## 验证安装

要验证 Torch-RecHub 是否正确安装，您可以运行：

```python
import torch_rechub
print(torch_rechub.__version__)
```

或运行一个简单的示例：

```bash
# 需要先进入脚本所在目录（脚本使用相对路径加载数据）
cd examples/matching
python run_ml_dssm.py
```

## 故障排除

### PyTorch 安装

如果您需要安装特定 CUDA 版本的 PyTorch，请参考 [NVIDIA CUDA / PyTorch 版本](https://pytorch.org/get-started/previous-versions/)。

### NVIDIA GPU 支持

要获得 NVIDIA GPU 加速，请确保您拥有：
- NVIDIA GPU，计算能力 3.5 或更高
- 已安装 CUDA Toolkit
- 已安装 cuDNN 库

### AMD GPU 支持（ROCm）

如果您使用的是 Ryzen AI Max+ 395/390/385 等 `gfx1151` 设备，可以通过 AMD 的 ROCm wheel 源安装 ROCm 和 PyTorch：

```bash
uv pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio
```

### NPU 支持（华为昇腾）

Torch-RecHub 支持华为昇腾 NPU 设备，测试设备为 **华为昇腾 910B**。

使用前请安装昇腾支持的 PyTorch 和 torch-npu 版本，具体版本对应关系请参考 [Huawei Ascend NPU / PyTorch 版本](https://www.hiascend.com/document/detail/zh/Pytorch/2600/releasenote/docs/zh/release_notes/release_notes.md)。

安装完成后，需要在代码中导入 `torch_npu`，然后在 Trainer 中指定设备即可：

```python
import torch
import torch_npu

trainer = CTRTrainer(model, device='npu:0')
```

### 常见问题

如果您遇到任何安装问题，请：
1. 查看 [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. 创建新的 Issue，并提供详细的错误信息和系统信息
3. 参考 常见问题解答 部分

