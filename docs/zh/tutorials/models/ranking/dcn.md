---
title: DCN / DCNv2 使用示例
description: Deep & Cross Network 完整使用教程 —— 从数据准备到模型训练与评估
---

# DCN / DCNv2 使用示例

## 1. 模型简介与适用场景

**DCN**（Deep & Cross Network）是 Google 在 ADKDD'2017 上提出的模型，通过交叉网络（Cross Network）**显式学习高阶特征交叉**，同时保持线性计算复杂度。**DCNv2** 是其增强版本，在 WWW'2021 发表，进一步提高了表达能力。

**论文**:
- DCN: [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
- DCNv2: [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)

### 核心区别

| 特性 | DCN | DCNv2 |
|------|-----|-------|
| 交叉方式 | 向量级交叉 | 矩阵级交叉 |
| 表达能力 | 中等 | 更强 |
| 参数数量 | 较少 | 较多 |

### 适用场景

- 需要显式特征交叉的 CTR 预测任务
- 特征维度较高的场景
- 对计算效率有要求的场景

---

## 2. 数据准备与预处理

使用 Criteo 数据集，数据准备流程与 DeepFM 相同。

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

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
```

---

## 3. 模型配置与参数说明

### 3.1 DCN

```python
from torch_rechub.models.ranking import DCN

model_dcn = DCN(
    features=dense_feas + sparse_feas,  # 全部特征
    n_cross_layers=3,                    # Cross Network 层数
    mlp_params={
        "dims": [256, 128],
    }
)
```

> **注意**: DCN 使用统一的 `features` 参数（不像 DeepFM 需要分别指定 deep 和 fm 特征），Cross Network 和 Deep Network 共享同一组特征。

### 3.2 DCNv2

```python
from torch_rechub.models.ranking import DCNv2

model_dcnv2 = DCNv2(
    features=dense_feas + sparse_feas,
    n_cross_layers=3,
    mlp_params={
        "dims": [256, 128],
        "dropout": 0.2,
        "activation": "relu"
    }
)
```

### 3.3 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 全部特征列表 (Dense + Sparse) | 所有特征 |
| `n_cross_layers` | `int` | Cross Network 的层数 | 2 ~ 4 |
| `mlp_params.dims` | `list[int]` | Deep Network 各层维度 | `[256, 128]` |
| `mlp_params.dropout` | `float` | Dropout 比率 | 0.1 ~ 0.3 |

---

## 4. 训练过程与代码示例

```python
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)

# 使用 DCN 或 DCNv2
model = model_dcn  # 或 model_dcnv2

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=50,
    earlystop_patience=10,
    device="cpu",
    model_path="./saved/dcn"
)

trainer.fit(train_dl, val_dl)
```

---

## 5. 模型评估与结果分析

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### DCN vs DCNv2 性能对比

| 模型 | Criteo Sample AUC | 训练速度 |
|------|-------------------|----------|
| DCN  | 0.70 ~ 0.74       | 快       |
| DCNv2 | 0.71 ~ 0.75      | 中等     |

DCNv2 通常比 DCN 高 0.5% ~ 1.0% AUC，但训练时间略长。

---

## 6. 参数调优建议

1. **Cross 层数** (`n_cross_layers`): 核心超参数
   - 层数越多，能学到的特征交叉阶数越高
   - 但过多层数容易过拟合，建议 2~4 层

2. **选择 DCN 还是 DCNv2**:
   - 数据量较小 → DCN（参数少，不易过拟合）
   - 数据量较大 → DCNv2（表达力更强）

3. **MLP 与 Cross 的平衡**:
   - MLP 层数和 Cross 层数不宜同时过大
   - 推荐组合: Cross=3 + MLP=`[256, 128]`

---

## 7. 常见问题与解决方案

### Q1: DCN 和 DCNv2 的区别是什么？
DCN 使用向量级的特征交叉（参数量 O(d)），DCNv2 使用矩阵级的特征交叉（参数量 O(d²)），后者表达能力更强但参数更多。

### Q2: n_cross_layers 设多少合适？
经验上 2~4 层效果最佳。交叉层太多容易过拟合，且边际增益递减。

### Q3: Cross Network 是否可以替代手工的特征交叉？
Cross Network 的设计目的就是自动学习特征交叉，理论上可以替代手工交叉特征。但在某些业务场景下，结合手工特征工程可能进一步提升效果。

---

## 完整代码

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DCN, DCNv2
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator


def main(use_dcnv2=False):
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

    # 选择模型
    ModelClass = DCNv2 if use_dcnv2 else DCN
    model = ModelClass(
        features=dense_feas + sparse_feas,
        n_cross_layers=3,
        mlp_params={"dims": [256, 128]}
    )

    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=50,
        earlystop_patience=10,
        device="cpu",
        model_path=f"./saved/{'dcnv2' if use_dcnv2 else 'dcn'}"
    )
    trainer.fit(train_dl, val_dl)

    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main(use_dcnv2=False)  # 改为 True 使用 DCNv2
```
