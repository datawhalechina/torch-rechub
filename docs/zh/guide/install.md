---
title: 安装指南
description: Torch-RecHub 详细安装说明，包括稳定版和最新版安装步骤
---

# 安装指南

本文档提供了 Torch-RecHub 的详细安装说明，包括稳定版和最新开发版的安装步骤。

## 系统要求

- Python 3.10 及以上
- PyTorch `>=2.6,<3`

NumPy（1.26.4+）、pandas（2.1+）和 scikit-learn（1.4+）会随项目安装。生成式模型使用可选依赖 Transformers（`>=4.57.3,<5`）。

## 安装方式

CPU 环境可直接选择稳定版或源码安装。使用 GPU 或 NPU 时，先按[设备示例](#按设备安装-pytorch)准备 PyTorch，再在同一个环境中安装 Torch-RecHub。

### 稳定版（推荐用户使用）

```bash
python -m pip install torch-rechub
```

### 可选功能

默认安装只包含核心训练依赖。按实际用途选择 extra：使用发布包时执行表中的 pip 命令；使用源码时，在仓库根目录运行 `uv sync --extra <name>`，例如 `uv sync --extra visualization`。

| 用途 | 安装命令 |
|---|---|
| 生成式模型 | `python -m pip install "torch-rechub[generative]"` |
| Parquet 流式数据 | `python -m pip install "torch-rechub[bigdata]"` |
| ONNX 导出与量化 | `python -m pip install "torch-rechub[onnx]"` |
| 模型可视化 | `python -m pip install "torch-rechub[visualization]"` |
| 实验跟踪 | `python -m pip install "torch-rechub[tracking]"` |
| 向量检索 | `python -m pip install "torch-rechub[annoy]"`（按后端选 `annoy`、`faiss` 或 `milvus`） |
| 测试、格式化与类型检查工具 | `python -m pip install "torch-rechub[dev]"` |
| 全部可选功能 | `python -m pip install "torch-rechub[all]"` |

渲染模型结构图还需在系统中安装 Graphviz，并确保命令行能找到 `dot`。

运行脚本时，也可用 `uv run --extra <name> python <script.py>` 指定所需依赖组；多个依赖组可以重复传入 `--extra`。

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

已按设备安装 PyTorch 时，最后一步用 `uv pip install -e .` 将源码装到当前已激活的环境中，再用该环境的 `python` 运行示例。

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

## 按设备安装 PyTorch

NVIDIA GPU 按显卡和驱动选择 CUDA 版 PyTorch，AMD GPU 按显卡型号选择 ROCm 版 PyTorch，华为昇腾 NPU 先安装并加载 CANN，再选择配套的 PyTorch 和 `torch-npu`。具体组合可查阅 [PyTorch 安装向导](https://pytorch.org/get-started/locally/)、[AMD ROCm 文档](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/) 和[昇腾版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)。

在已激活的 Python 环境中，按设备选择一条安装命令。使用 uv 时，可将 `python -m pip` 换成 `uv pip`。

```bash
# CPU（Linux 或 Windows 的纯 CPU 版本）
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu

# NVIDIA GPU，以 CUDA 12.6 为例
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126

# AMD GPU，以 Ryzen AI Max+ 395（gfx1151）为例
python -m pip install --index-url https://repo.amd.com/rocm/whl/gfx1151/ "rocm[libraries,devel]" torch torchvision torchaudio

# 华为昇腾 NPU，默认包版本与本机 CANN 匹配时
python -m pip install torch torch-npu
```

使用第一张卡训练时，NVIDIA 和 AMD 在 Trainer 中都写 `device="cuda:0"`，华为昇腾可在导入 `torch_npu` 后使用 `device="npu:0"`。

## 验证安装

在安装 Torch-RecHub 的环境中查看版本。

```bash
python -c "import torch_rechub; print(torch_rechub.__version__)"
```

## 故障排除

### 常见问题

如果您遇到任何安装问题，请：
1. 查看 [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)
2. 创建新的 Issue，并提供详细的错误信息和系统信息
3. 参考 [常见问题解答](/zh/community/faq)

