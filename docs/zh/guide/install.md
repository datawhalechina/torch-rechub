---
title: 安装指南
description: Torch-RecHub 详细安装说明，包括稳定版和最新版安装步骤
---

# 安装指南

本文档提供了 Torch-RecHub 的详细安装说明，包括稳定版和最新开发版的安装步骤。

## 系统要求

在安装 Torch-RecHub 之前，请确保您的系统满足以下要求：

- **Python 3.9+**
- **PyTorch 1.7+**（推荐使用 CUDA 版本以获得 GPU 加速）
- **NumPy**
- **Pandas**
- **SciPy**
- **Scikit-learn**

## 安装方式

### 稳定版（推荐用户使用）

最简单的安装方式是通过 pip：

```bash
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
python examples/matching/run_ml_dssm.py
```

## 故障排除

### PyTorch 安装

如果您需要安装特定 CUDA 版本的 PyTorch，请访问 [PyTorch 官方网站](https://pytorch.org/get-started/locally/) 获取针对您系统的安装说明。

### GPU 支持

要获得 GPU 加速，请确保您拥有：
- NVIDIA GPU，计算能力 3.5 或更高
- 已安装 CUDA Toolkit
- 已安装 cuDNN 库

### 常见问题

如果您遇到任何安装问题，请：
1. 查看 [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. 创建新的 Issue，并提供详细的错误信息和系统信息
3. 参考 常见问题解答 部分

