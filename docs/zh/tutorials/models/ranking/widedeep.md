---
title: Wide&Deep 使用示例
description: Wide&Deep 模型完整使用教程 —— 从数据准备到模型训练与评估
---

# Wide&Deep 使用示例

## 1. 模型简介与适用场景

Wide&Deep 是 Google 在 DLRS'2016 上提出的经典推荐模型，结合了线性模型（Wide 部分）的记忆能力和深度神经网络（Deep 部分）的泛化能力，是工业界应用最广泛的基线模型之一。

**论文**: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

### 模型结构

- **Wide 部分**：线性模型，捕获特征间的共现关系（记忆）
- **Deep 部分**：多层 MLP，通过 Embedding 学习特征的泛化表示
- **联合训练**：Wide 和 Deep 的输出相加后通过 Sigmoid 得到最终预测

### 适用场景

- 推荐系统基线模型
- 需要同时利用记忆和泛化能力的场景
- 快速验证数据特征有效性

---

## 2. 数据准备与预处理

本示例使用 Criteo 广告点击数据集，数据准备流程与 DeepFM 相同。

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# 加载数据
data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

# 区分特征类型
dense_features = [f for f in data.columns if f.startswith("I")]
sparse_features = [f for f in data.columns if f.startswith("C")]

# 缺失值处理
data[sparse_features] = data[sparse_features].fillna("0")
data[dense_features] = data[dense_features].fillna(0)

# 连续特征归一化
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# 类别特征编码
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# 定义特征
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

y = data["label"]
del data["label"]
x = data

# 创建 DataLoader
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1], batch_size=2048
)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.ranking import WideDeep

model = WideDeep(
    wide_features=dense_feas,      # Wide 部分: 通常使用连续特征
    deep_features=sparse_feas,     # Deep 部分: 通常使用类别特征
    mlp_params={
        "dims": [256, 128],        # MLP 隐藏层维度
        "dropout": 0.2,
        "activation": "relu"
    }
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `wide_features` | `list[Feature]` | Wide 部分的特征列表 | 连续特征 / 交叉特征 |
| `deep_features` | `list[Feature]` | Deep 部分的特征列表 | 类别特征 |
| `mlp_params.dims` | `list[int]` | MLP 各层维度 | `[256, 128]` |
| `mlp_params.dropout` | `float` | Dropout 比率 | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | 激活函数 | `"relu"` |

> **Wide 与 Deep 的特征划分**:
> - **Wide 部分**适合放入低维连续特征（如年龄、价格）或手工交叉特征
> - **Deep 部分**适合放入高维类别特征（如用户ID、城市），通过 Embedding 降维

---

## 4. 训练过程与代码示例

```python
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=50,
    earlystop_patience=10,
    device="cpu",                   # GPU: "cuda:0"
    model_path="./saved/widedeep"
)

trainer.fit(train_dl, val_dl)
```

---

## 5. 模型评估与结果分析

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 预期性能

| 数据规模 | 预期 AUC |
|----------|----------|
| Sample (1万条) | 0.68 ~ 0.73 |
| Full (4500万条) | 0.78 ~ 0.80 |

Wide&Deep 作为基线模型，AUC 通常略低于 DeepFM，但训练速度更快。

---

## 6. 参数调优建议

1. **Wide 和 Deep 特征划分**是最关键的选择：
   - 经验法则：连续特征 → Wide，类别特征 → Deep
   - 也可以让两部分共用全部特征

2. **MLP 结构**：WideDeep 的 Deep 部分不宜过深，`[256, 128]` 通常足够

3. **学习率**：推荐从 `1e-3` 开始搜索

---

## 7. 常见问题与解决方案

### Q1: Wide 部分和 Deep 部分应该使用相同特征吗？
建议分开使用：Wide 部分侧重记忆能力，适合低维特征；Deep 部分侧重泛化，适合高维类别特征。但也可以让两部分使用相同特征，模型仍然有效。

### Q2: WideDeep 和 DeepFM 哪个好？
一般来说 DeepFM 优于 WideDeep，因为 FM 部分自动学习二阶交叉，而 WideDeep 的 Wide 部分只做线性变换。但 WideDeep 结构更简单、训练更快。

### Q3: 如何添加交叉特征到 Wide 部分？
可以在数据预处理阶段手动构造交叉特征 (如 `city_x_gender`)，然后作为新的 SparseFeature 加入 `wide_features`.

---

## 完整代码

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import WideDeep
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)

    data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")
    dense_features = [f for f in data.columns if f.startswith("I")]
    sparse_features = [f for f in data.columns if f.startswith("C")]

    data[sparse_features] = data[sparse_features].fillna("0")
    data[dense_features] = data[dense_features].fillna(0)

    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [
        SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
        for name in sparse_features
    ]

    y = data["label"]
    del data["label"]
    x = data

    dg = DataGenerator(x, y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        split_ratio=[0.7, 0.1], batch_size=2048
    )

    model = WideDeep(
        wide_features=dense_feas,
        deep_features=sparse_feas,
        mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
    )

    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=50,
        earlystop_patience=10,
        device="cpu",
        model_path="./saved/widedeep"
    )
    trainer.fit(train_dl, val_dl)

    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
```
