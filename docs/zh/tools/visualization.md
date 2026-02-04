---
title: 可视化监控
description: Torch-RecHub 训练过程可视化监控
---

# 可视化监控

Torch-RecHub 提供了模型架构可视化功能，帮助开发者直观地理解模型结构和计算流程。

## 为什么选择 torchview

Torch-RecHub 采用 [torchview](https://github.com/mert-kurttutan/torchview) 作为可视化后端，而非其他常见方案（如 torchviz、netron），**最核心的原因是：torchview 是唯一支持复杂字典输入的可视化工具**。

推荐系统模型的输入通常是包含多种特征类型的字典：

```python
# 推荐模型的典型输入格式
x = {
    "user_id": tensor([1, 2, 3]),           # 稀疏特征
    "age": tensor([0.5, 0.3, 0.8]),         # 稠密特征
    "hist_items": tensor([[1,2,3], ...]),   # 序列特征
}
model(x)  # 字典作为输入
```

其他可视化工具（torchviz、netron 等）仅支持简单的 Tensor 输入，无法处理这种字典形式的复杂输入结构。

> **提示**：如果你已将模型导出为 ONNX 格式，也可以使用 [Netron](https://netron.app/) 在线查看模型结构。详见 [ONNX 导出文档](/zh/serving/onnx)。

| 特性 | torchview | torchviz | netron |
|------|-----------|----------|--------|
| **支持字典输入** | ✅ | ❌ | ❌ (需先导出 ONNX) |
| 基于前向传播追踪 | ✅ | ❌ (基于 autograd) | ❌ (静态解析) |
| 支持动态控制流 | ✅ | ❌ | ❌ |
| 显示张量形状 | ✅ | ❌ | ✅ |
| 可调节展示深度 | ✅ | ❌ | ❌ |
| 嵌套模块展开 | ✅ | ❌ | 部分 |

**其他优势**：
- **前向追踪**：通过 hook 机制追踪前向传播，准确捕获注意力机制、多塔结构等动态结构
- **层级控制**：通过 `depth` 参数灵活控制展示粒度
- **形状可视化**：直观显示各层输入输出的张量形状

## 安装依赖

可视化功能需要安装额外依赖：

```bash
pip install torch-rechub[visualization]
```

同时需要安装系统级 graphviz：

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

## 快速开始

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model

# 创建模型
model = DeepFM(
    deep_features=deep_features,
    fm_features=fm_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2}
)

# 可视化模型（在 Jupyter 中自动显示）
graph = visualize_model(model, depth=4)

# 保存为 PDF
visualize_model(model, save_path="model_arch.pdf", dpi=300)
```

## 核心函数

### visualize_model

生成模型的计算图可视化。

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
| `input_data` | dict | 输入数据字典，如果为 None 则自动生成 | None |
| `batch_size` | int | 自动生成输入时的 batch size | 2 |
| `seq_length` | int | 序列特征的长度 | 10 |
| `depth` | int | 可视化深度，-1 表示显示所有层 | 3 |
| `show_shapes` | bool | 是否在边上显示张量形状 | True |
| `expand_nested` | bool | 是否展开嵌套的 nn.Module | True |
| `save_path` | str | 保存路径，支持 .pdf/.svg/.png | None |
| `dpi` | int | 输出图像分辨率 | 300 |

### display_graph

在 Jupyter 中显示计算图。

```python
from torch_rechub.utils.visualization import display_graph

# 获取计算图
graph = visualize_model(model, depth=4)

# 在 Jupyter 中显示
display_graph(graph, format='png')
```

## 使用示例

### 排序模型可视化

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model
from torch_rechub.basic.features import DenseFeature, SparseFeature

# 定义特征
dense_features = [DenseFeature("age"), DenseFeature("income")]
sparse_features = [
    SparseFeature("city", vocab_size=100, embed_dim=16),
    SparseFeature("gender", vocab_size=3, embed_dim=8)
]

# 创建模型
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2}
)

# 可视化
visualize_model(model, depth=4, save_path="deepfm_arch.pdf")
```

### 召回模型可视化

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.utils.visualization import visualize_model

model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64]},
    item_params={"dims": [256, 128, 64]}
)

visualize_model(model, depth=3, save_path="dssm_arch.png", dpi=300)
```

### 通过 Trainer 可视化

训练器也提供了可视化方法：

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(model)
trainer.fit(train_dl, val_dl)

# 可视化模型
trainer.visualization(save_path="model.pdf", depth=4)
```

## 输出格式

支持多种输出格式：

| 格式 | 扩展名 | 适用场景 |
| --- | --- | --- |
| PDF | .pdf | 论文、报告（矢量图，可缩放） |
| SVG | .svg | 网页展示（矢量图） |
| PNG | .png | 通用图片格式 |

## 常见问题

### Q: 提示 graphviz 未安装？

确保同时安装了 Python 包和系统包：

```bash
pip install graphviz
# 以及系统级安装（见上文）
```

### Q: 在 VSCode 中图片不显示？

尝试设置输出格式为 PNG：

```python
import graphviz
graphviz.set_jupyter_format('png')
```

### Q: 如何调整图片大小？

使用 `dpi` 参数控制分辨率，或使用返回的 graph 对象调整：

```python
graph = visualize_model(model, depth=4)
graph.resize_graph(scale=1.5)
```

