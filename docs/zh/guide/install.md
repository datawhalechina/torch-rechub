---
title: 安装指南
description: Torch-RecHub 详细安装说明，包括稳定版和最新版安装步骤
---

# 安装指南

跟着教程运行仓库里的样本，建议直接安装源码。如果已有自己的训练项目，安装 PyPI 发布包就可以导入 `torch_rechub`。

## 系统要求

- Python 3.10 及以上
- PyTorch `>=2.6,<3`

NumPy（1.26.4+）、pandas（2.1+）和 scikit-learn（1.4+）会随项目安装。生成式模型使用可选依赖 Transformers（`>=4.57.3,<5`）。

## 安装方式

CPU 环境可直接选择稳定版或源码安装。使用 GPU 或 NPU 时，先按[设备示例](#按设备安装-pytorch)准备 PyTorch，再在同一个环境中安装 Torch-RecHub。

### 在已有项目中安装

```bash
python -m pip install torch-rechub
```

### 可选功能

基础安装可以训练 DeepFM、DSSM 等模型。需要画图、导出 ONNX 或做向量检索时，再加装对应的依赖。下表给的是发布包命令，在源码目录可以写成 `uv sync --extra visualization` 这样的形式。

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

只在某个脚本里用到这些依赖时，可以直接运行 `uv run --extra <name> python <script.py>`。一个脚本用到两组依赖，就写两次 `--extra`。

### 从源码安装

示例脚本和样本数据随仓库提供。按下面的命令下载源码并创建 `.venv`。

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

## 准备提交改动

准备提 PR 时，先在 GitHub 上 fork 仓库，再克隆自己的副本。把下面的 `YOUR_USERNAME` 换成你的用户名。

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

先看机器上装的是什么卡，再选 PyTorch 安装包。

- NVIDIA GPU 需要与驱动兼容的 CUDA 版本。[PyTorch 安装向导](https://pytorch.org/get-started/locally/)
- AMD GPU 先按显卡型号查找对应的 ROCm 安装包。[AMD ROCm 文档](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/)
- 华为昇腾 NPU 先安装并加载 CANN，再按配套关系安装 PyTorch 和 `torch-npu`。[昇腾版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)

下面各给一个安装示例。在已激活的 Python 环境中，选与你设备相符的一条。使用 uv 时，可将 `python -m pip` 换成 `uv pip`。

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

使用加速卡时，在同一环境中选择对应的命令，确认 PyTorch 能在设备上创建张量。

```bash
# NVIDIA CUDA 或 AMD ROCm
python -c "import torch; print(torch.zeros(1, device='cuda:0').device)"

# 华为昇腾，先加载 CANN 环境
python -c "import torch, torch_npu; print(torch.zeros(1, device='npu:0').device)"
```

源码安装后如果提示找不到 `torch_rechub`，先在仓库根目录用 `uv run python` 重试，确认正在使用项目的 `.venv`。设备检查若报错，记下显卡型号、驱动或 CANN 版本、PyTorch 版本和完整报错，再到项目中查找相同情况。[常见问题](/zh/community/faq) · [GitHub Issues](https://github.com/datawhalechina/torch-rechub/issues)

