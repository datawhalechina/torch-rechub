---
title: 安装指南
description: Torch-RecHub 详细安装说明，包括稳定版和最新版安装步骤
---

# 安装指南

本文档提供了 Torch-RecHub 的详细安装说明，包括稳定版和最新开发版的安装步骤。

## 系统要求

在安装 Torch-RecHub 之前，请确保您的系统满足以下要求：

- **Python 3.10+**
- **PyTorch 2.6+**（根据硬件选择 CPU、NVIDIA CUDA、AMD ROCm 或华为昇腾 NPU 版本）
- **NumPy**
- **Pandas**
- **Scikit-learn**

本项目的开发基线为 Python 3.10+、PyTorch 2.6+、NumPy 1.26.4+、Pandas 2.1+、Scikit-learn 1.4+。生成式可选依赖限定 Transformers 4.57.3 至 5.0 之前；不自动跨到 5.x。版本下限表示选用的 API 基线，并非每一种硬件/依赖组合都已运行验证。RTX 50 系等新显卡仍需选择支持该架构的较新 PyTorch/CUDA 构建，不应为了贴近最低版本降级已可用的环境。

## 安装方式

PyTorch 与操作系统、硬件、驱动和运行时版本强相关。CPU 用户可以直接安装 Torch-RecHub；NVIDIA CUDA、AMD ROCm 或华为昇腾 NPU 用户应先按官方文档选择匹配的 PyTorch 构建：[PyTorch 安装选择器](https://pytorch.org/get-started/locally/)、[AMD ROCm / PyTorch 兼容性](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html)、[Huawei Ascend NPU / PyTorch 版本配套](https://www.hiascend.com/document/detail/zh/Pytorch/2600/releasenote/docs/zh/release_notes/release_notes.md)。硬件安装命令会随版本变化，请以这些官方页面当前生成的命令为准。

### 稳定版（推荐用户使用）

最简单的安装方式是通过 pip：

```bash
# CPU 环境会自动安装项目声明的 PyTorch 依赖
python -m pip install torch-rechub

# CUDA / ROCm / NPU 环境：先按上方官方指引安装 PyTorch，再执行
python -m pip install torch-rechub
```

### 可选功能

默认安装只包含核心训练依赖。按实际用途选择 extra：

| 用途 | 安装命令 |
|---|---|
| 生成式模型 | `python -m pip install "torch-rechub[generative]"` |
| Parquet 流式数据 | `python -m pip install "torch-rechub[bigdata]"` |
| ONNX 导出与量化 | `python -m pip install "torch-rechub[onnx]"` |
| 模型可视化 | `python -m pip install "torch-rechub[visualization]"` |
| 实验跟踪 | `python -m pip install "torch-rechub[tracking]"` |
| 统一向量索引工厂（Annoy + Faiss + Milvus） | `python -m pip install "torch-rechub[annoy,faiss,milvus]"` |
| 全部可选功能 | `python -m pip install "torch-rechub[all]"` |

> 当前 `torch_rechub.serving` 初始化时会同时导入 Annoy、Faiss 和 Milvus 后端。如果要使用 `from torch_rechub.serving import builder_factory`，需安装表中的三个 extra；只安装单一后端目前不足以导入该统一入口。

### 最新开发版

要安装包含最新功能的开发版本：

```bash
# 首先安装 uv（如果尚未安装）
python -m pip install uv

# 克隆并安装
git clone https://github.com/datawhalechina/torch-rechub.git
cd torch-rechub

# 创建 .venv、安装锁定依赖，并以可编辑方式安装当前项目
uv sync
```

CUDA、ROCm 或 NPU 开发环境仍需先按官方兼容性文档确定 PyTorch 与运行时的组合；不要盲目复用其他机器的 wheel 地址。

## 开发环境设置

如果您想为 Torch-RecHub 做出贡献或使用源代码：

```bash
# 1. Fork 并克隆仓库
git clone https://github.com/YOUR_USERNAME/torch-rechub.git
cd torch-rechub

# 2. 安装依赖并设置环境
uv sync

# 3. 运行测试（uv sync 已以可编辑方式安装当前项目）
uv run pytest
```

## 验证安装

要验证 Torch-RecHub 是否正确安装，您可以运行：

```python
import torch_rechub
print(torch_rechub.__version__)
```

如果还需确认 PyTorch 识别到的设备，可运行：

```python
import torch

print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
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

请使用 [PyTorch 安装选择器](https://pytorch.org/get-started/locally/) 按操作系统和 CUDA 环境生成命令。安装后以 `torch.cuda.is_available()` 的返回值验证当前 Python 环境。

### AMD GPU 支持（ROCm）

请先在 [AMD ROCm / PyTorch 兼容性文档](https://rocm.docs.amd.com/en/latest/compatibility/ml-compatibility/pytorch-compatibility.html) 中确认操作系统、GPU 架构、ROCm 和 PyTorch 版本组合，再使用对应的官方安装命令。

### NPU 支持（华为昇腾）

Torch-RecHub 支持华为昇腾 NPU 设备，测试设备为 **华为昇腾 910B**。

使用前请安装昇腾支持的 PyTorch 和 torch-npu 版本，具体版本对应关系请参考 [Huawei Ascend NPU / PyTorch 版本](https://www.hiascend.com/document/detail/zh/Pytorch/2600/releasenote/docs/zh/release_notes/release_notes.md)。

安装完成后，需要在代码中导入 `torch_npu`，然后在 Trainer 中指定设备即可：

```python
import torch_npu
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(model, device='npu:0')
```

### 常见问题

如果您遇到任何安装问题，请：
1. 查看 [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. 创建新的 Issue，并提供详细的错误信息和系统信息
3. 参考 [常见问题解答](/zh/community/faq)

