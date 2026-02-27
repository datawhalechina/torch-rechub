---
title: DeepFM 使用示例
description: DeepFM 模型完整使用教程 —— 从数据准备到模型训练与评估
---

# DeepFM 使用示例

## 1. 模型简介与适用场景

DeepFM（Deep Factorization Machine）是由华为诺亚方舟实验室在 IJCAI'2017 上提出的模型，将因子分解机（FM）与深度神经网络结合，能够**同时捕获低阶和高阶特征交互**，且无需手动特征工程。

**论文**: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

### 模型结构

<div align="center">
  <img src="/img/models/deepfm_arch.png" alt="DeepFM Model Architecture" width="600"/>
</div>

- **FM 部分**：通过二阶交互捕获特征间的组合关系
- **Deep 部分**：通过多层全连接网络捕获高阶非线性特征交互
- **共享 Embedding**：FM 和 Deep 共享底层 Embedding，减少参数

### 适用场景

- 点击率 (CTR) 预测
- 广告推荐排序
- 需要同时利用低阶和高阶特征交互的业务场景
- 工业界基准模型，适合作为起步模型

---

## 2. 数据准备与预处理

本示例使用 **Criteo** 广告点击数据集。原始数据包含 13 个连续特征（I1-I13）和 26 个类别特征（C1-C26）。

### 2.1 加载数据

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature

# 加载 Criteo 采样数据
data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")
print(f"数据量: {data.shape[0]} 条, 特征数: {data.shape[1] - 1}")
```

### 2.2 特征处理

```python
# 区分连续特征和类别特征
dense_features = [f for f in data.columns if f.startswith("I")]
sparse_features = [f for f in data.columns if f.startswith("C")]

# 填充缺失值
data[sparse_features] = data[sparse_features].fillna("0")
data[dense_features] = data[dense_features].fillna(0)

# 连续特征: 归一化到 [0, 1]
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# 类别特征: LabelEncoder 编码
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
```

### 2.3 定义特征

```python
# 定义 DenseFeature 和 SparseFeature
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

# 提取标签
y = data["label"]
del data["label"]
x = data
```

### 2.4 创建 DataLoader

```python
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],  # 训练集:验证集:测试集 = 7:1:2
    batch_size=2048
)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.ranking import DeepFM

model = DeepFM(
    deep_features=dense_feas + sparse_feas,  # Deep 部分使用全部特征
    fm_features=sparse_feas,       # FM 部分使用的特征
    mlp_params={
        "dims": [256, 128],        # MLP 隐藏层维度
        "dropout": 0.2,            # dropout 比率
        "activation": "relu"       # 激活函数
    }
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `deep_features` | `list[Feature]` | Deep 部分的特征列表，通常包含全部特征 | 连续特征 + 类别特征 |
| `fm_features` | `list[Feature]` | FM 部分的特征列表，通常包含 SparseFeature | 全部类别特征 |
| `mlp_params.dims` | `list[int]` | MLP 每层神经元数量 | `[256, 128]` 或 `[256, 128, 64]` |
| `mlp_params.dropout` | `float` | Dropout 比率，防止过拟合 | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | 激活函数 (`relu`, `prelu`, `sigmoid`) | `"relu"` |

---

## 4. 训练过程与代码示例

### 4.1 创建训练器并训练

```python
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={
        "lr": 1e-3,               # 学习率
        "weight_decay": 1e-3       # L2 正则化
    },
    n_epoch=50,                    # 最大训练轮数
    earlystop_patience=10,         # 早停耐心值
    device="cpu",                  # 使用 "cuda:0" 进行 GPU 训练
    model_path="./saved/deepfm"    # 模型保存路径
)

# 开始训练
ctr_trainer.fit(train_dl, val_dl)
```

### 4.2 训练日志说明

训练过程中会输出每个 epoch 的训练损失和验证集 AUC：

```
epoch: 0, train loss: 0.5234
epoch: 0, val auc: 0.7156
epoch: 1, train loss: 0.4987
epoch: 1, val auc: 0.7321
...
```

---

## 5. 模型评估与结果分析

### 5.1 测试集评估

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 5.2 预期性能参考

在 Criteo 数据集上的典型 AUC 范围：

| 数据规模 | 预期 AUC |
|----------|----------|
| Sample (1万条) | 0.70 ~ 0.75 |
| Full (4500万条) | 0.79 ~ 0.81 |

> **注意**: 实际性能取决于数据集大小、特征工程和超参数设置。

---

## 6. 参数调优建议

### 6.1 超参数调优优先级

1. **学习率** (`lr`): 最重要的超参数，建议从 `1e-3` 开始搜索 `[1e-4, 5e-4, 1e-3, 5e-3]`
2. **Embedding 维度** (`embed_dim`): 影响模型容量，建议 `8 ~ 32`
3. **MLP 层数和维度**: `[256, 128]` 是较好的起点，可尝试 `[512, 256, 128]`
4. **Dropout**: `0.1 ~ 0.3`，数据量较小时增大

### 6.2 调优技巧

```python
# 使用学习率调度器
ctr_trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=100,
    earlystop_patience=10,
    device="cuda:0",
    model_path="./saved/deepfm",
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 2, "gamma": 0.8}  # 每2个epoch衰减
)
```

---

## 7. 常见问题与解决方案

### Q1: 训练损失不下降？
- 检查学习率是否过大或过小
- 确认数据预处理是否正确（缺失值处理、归一化）
- 检查标签分布是否极度不平衡

### Q2: 过拟合（训练 AUC 高，验证 AUC 低）？
- 增加 dropout 比率（0.2 → 0.5）
- 增大 weight_decay（1e-3 → 1e-2）
- 减少 MLP 层数或维度
- 使用更多训练数据

### Q3: 如何选择 deep_features 和 fm_features?
- 通常做法：**全部特征给 Deep 部分，sparse 特征给 FM 部分**
- 也可以只用 dense 特征给 Deep，但效果通常不如全部特征

### Q4: GPU 内存不足？
- 减小 `batch_size`（如 2048 → 512）
- 减小 `embed_dim`（如 16 → 8）
- 减小 MLP 维度

---

## 8. 模型可视化

Torch-RecHub 内置了基于 `torchview` 的模型结构可视化工具，可以生成模型的计算图。

### 安装依赖

```bash
pip install torch-rechub[visualization]
# 系统级依赖:
# Ubuntu: sudo apt-get install graphviz
# macOS: brew install graphviz
# Windows: choco install graphviz
```

### 可视化 DeepFM 模型

```python
from torch_rechub.utils.visualization import visualize_model

# 自动生成输入并可视化（在 Jupyter 中直接显示内嵌图像）
graph = visualize_model(model, depth=4)

# 保存为高清 PNG（适合论文/文档）
visualize_model(model, save_path="deepfm_architecture.png", dpi=300)

# 保存为 PDF
visualize_model(model, save_path="deepfm_architecture.pdf")
```

### DeepFM 架构图

![DeepFM 模型架构图](/img/models/deepfm_arch.png)

> `visualize_model` 会自动从模型中提取特征信息并生成 dummy input，无需手动构造。支持自定义 `depth`（展开的层数）和 `batch_size`。

---

## 9. ONNX 导出

将训练好的模型导出为 ONNX 格式，用于跨框架部署（如 ONNX Runtime、TensorRT、OpenVINO）。

### 导出模型

```python
from torch_rechub.utils.onnx_export import ONNXExporter

exporter = ONNXExporter(model, device="cpu")

# 导出 DeepFM 模型
exporter.export("deepfm.onnx", verbose=True)

# 查看输入信息
info = exporter.get_input_info()
print(info)
```

### 使用 ONNX Runtime 推理

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("deepfm.onnx")

# 查看模型输入
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

# 构造输入并推理
input_feed = {}
for inp in session.get_inputs():
    if "int" in inp.type.lower():
        input_feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.int64)
    else:
        input_feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)

output = session.run(None, input_feed)
print(f"Output shape: {output[0].shape}")
```

> 排序模型（DeepFM / WideDeep / DCN）的 ONNX 导出是完整模型导出，不涉及 Tower 分离。双塔模型（DSSM / YoutubeDNN）支持 `mode="user"/"item"` 分别导出。

---

## 完整代码

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)

    # 1. 加载数据
    data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

    # 2. 特征处理
    dense_features = [f for f in data.columns if f.startswith("I")]
    sparse_features = [f for f in data.columns if f.startswith("C")]

    data[sparse_features] = data[sparse_features].fillna("0")
    data[dense_features] = data[dense_features].fillna(0)

    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 3. 定义特征
    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [
        SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
        for name in sparse_features
    ]

    y = data["label"]
    del data["label"]
    x = data

    # 4. 创建 DataLoader
    dg = DataGenerator(x, y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        split_ratio=[0.7, 0.1], batch_size=2048
    )

    # 5. 创建模型
    model = DeepFM(
        deep_features=dense_feas + sparse_feas,        fm_features=sparse_feas,
        mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
    )

    # 6. 训练
    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=50,
        earlystop_patience=10,
        device="cpu",
        model_path="./saved/deepfm"
    )
    trainer.fit(train_dl, val_dl)

    # 7. 评估
    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")

    # 8. 可视化（可选）
    # from torch_rechub.utils.visualization import visualize_model
    # visualize_model(model, save_path="deepfm_arch.png", dpi=300)

    # 9. ONNX 导出（可选）
    # from torch_rechub.utils.onnx_export import ONNXExporter
    # exporter = ONNXExporter(model)
    # exporter.export("deepfm.onnx", verbose=True)


if __name__ == "__main__":
    main()
```

