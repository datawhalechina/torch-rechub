---
title: 模型结构图
description: 在 Jupyter 查看模型各层的输入输出形状，或将结构图保存为文件
---

# 模型结构图

想知道 DeepFM 的特征在哪里拼接，或者 DSSM 两个塔的输出维度是否一致，可以在训练前先画一张图。`visualize_model` 会按模型的特征定义生成一小批输入，用 torchview 执行一次前向计算。顺着图上的连线就能看到各层的输入和输出形状。

## 安装依赖

在源码目录安装绘图依赖。

```bash
uv sync --extra visualization
```

通过 PyPI 安装项目时，可使用 `pip install "torch-rechub[visualization]"`。

绘图还依赖 Graphviz 的 `dot` 命令。在当前操作系统上选一条安装命令。

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

## 快速开始

以 DeepFM 为例。下面定义两个连续特征和两个类别特征。FM 要对类别特征的向量做交互，因此这里把它们的嵌入维度都设成 16。

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model
from torch_rechub.basic.features import DenseFeature, SparseFeature

# 定义特征
dense_features = [DenseFeature("age"), DenseFeature("income")]
sparse_features = [
    SparseFeature("city", vocab_size=100, embed_dim=16),
    # FM 交互中的稀疏特征需使用相同 embedding 维度
    SparseFeature("gender", vocab_size=3, embed_dim=16)
]

# 创建模型
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2}
)
```

在 Jupyter 中运行下面一行，会在当前单元格显示网络结构。

```python
graph = visualize_model(model, depth=4)
```

在终端中生成文件时，指定 `save_path`，随后打开保存的 PDF 即可。

```python
graph = visualize_model(model, depth=4, save_path="deepfm_arch.pdf")
```

## 核心函数

### visualize_model

图里只会出现这批输入实际经过的层。如果想查看模型的其他分支，可以换一批 `input_data` 再画。`depth` 越大，展开的子模块越多。

默认在 CPU 上绘图。模型已经在 GPU 上时，函数会复制模型到 CPU，绘图后原模型仍在 GPU 上。模型较大时，可传入它所在的设备（例如 `device="cuda:0"`），省去这次复制。画完后，各层会恢复原先的训练或评估状态。

```python
from torch_rechub.utils.visualization import visualize_model

graph = visualize_model(
    model,                    # PyTorch 模型
    input_data=None,          # 输入数据（可选，自动生成）
    batch_size=2,             # 自动生成输入的 batch size
    seq_length=10,            # 序列特征的长度
    depth=3,                  # 可视化深度
    show_shapes=True,         # 是否显示张量形状
    expand_nested=True,       # 是否展开嵌套模块
    save_path=None,           # 保存路径
    graph_name="model",       # 图名称
    device="cpu",             # 设备
    dpi=300,                  # 输出分辨率
)
```

**参数说明：**

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| `model` | nn.Module | 要可视化的 PyTorch 模型 | 必需 |
| `input_data` | dict / Tensor / tuple / list | 字典整体传给模型，tuple/list 按顺序展开，None 使用自动生成的输入 | None |
| `batch_size` | int | 自动生成输入时的 batch size | 2 |
| `seq_length` | int | 序列特征的长度 | 10 |
| `depth` | int | 可视化深度，-1 表示显示所有层 | 3 |
| `show_shapes` | bool | 是否在边上显示张量形状 | True |
| `expand_nested` | bool | 是否展开嵌套的 nn.Module | True |
| `save_path` | str | 保存路径，支持 .pdf/.svg/.png | None |
| `dpi` | int | 输出图像分辨率 | 300 |

### 在 Jupyter 中切换显示格式

前面的调用默认显示 PNG。如果已经拿到了 `graph`，可以在另一个单元格中以 SVG 查看同一张图。

```python
from torch_rechub.utils.visualization import display_graph

display_graph(graph, format="svg");
```

## 通过训练器绘图

在 DSSM 教程里，完成 `trainer = MatchTrainer(...)` 后就可以运行下面这行，无需先训练。`CTRTrainer` 和 `MTLTrainer` 的调用方式相同。[DSSM 教程](/zh/tutorials/models/matching/dssm)

```python
trainer.visualization(save_path="dssm_arch.png", depth=3)
```

`SeqTrainer` 可以用 `seq_length` 控制图中示例序列的长度。

```python
seq_trainer.visualization(seq_length=20, save_path="sequence.svg")
```

如果自定义模型的 `forward` 接收 `tokens` 和 `time_diffs` 两个参数，可以传入这两个实际张量。

```python
visualize_model(model, input_data=(tokens, time_diffs), save_path="custom.png")
```

## 输出格式

| 格式 | 扩展名 | 适用场景 |
| --- | --- | --- |
| PDF | .pdf | 论文、报告（矢量图，可缩放） |
| SVG | .svg | 网页展示（矢量图） |
| PNG | .png | 通用图片格式 |

## 常见问题

### 为什么装了 Python 包仍然画不出图？

Python 包负责生成图的描述，Graphviz 的 `dot` 命令负责渲染。在启动 Jupyter 的终端运行 `dot -V`。如果提示找不到命令，按本页的安装步骤装好 Graphviz，将其 `bin` 目录加入 `PATH`，再重启终端和 Jupyter。

### 如何调整图片大小？

保存 PNG 时用 `dpi` 调整分辨率。需要在报告中放大图像时，可将 `save_path` 的扩展名改为 `.svg` 或 `.pdf`。

