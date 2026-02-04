---
title: 排序模型教程
description: Torch-RecHub 排序模型教程，包括 Wide & Deep、DeepFM、DIN 和 DCN-V2 等模型及特征工程技巧
---

# 排序模型教程

本教程将介绍如何使用 Torch-RecHub 中的各种排序模型。我们将使用 Criteo 数据集作为示例。

## 数据准备

首先，我们需要准备数据并进行特征处理：

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# 加载数据
df = pd.read_csv("criteo_sample.csv")

# 特征列定义
sparse_features = [f'C{i}' for i in range(1, 27)]
dense_features = [f'I{i}' for i in range(1, 14)]
```

## Wide & Deep 模型

Wide & Deep 模型结合了记忆和泛化能力：

```python
from torch_rechub.models.ranking import WideDeep
from torch_rechub.trainers import CTRTrainer

# 定义特征
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=df[name].nunique(), embed_dim=16) for name in sparse_features]

# 模型配置
model = WideDeep(
    wide_features=sparse_feas,
    deep_features=sparse_feas + dense_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# 训练配置
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)

# 训练模型
trainer.fit(train_dl, val_dl)
```

## DeepFM 模型

DeepFM 模型通过因子分解机和深度网络建模特征交互：

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

# 模型配置
model = DeepFM(
    deep_features=sparse_feas + dense_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# 训练配置
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)
```

## DIN (Deep Interest Network)

DIN 模型通过注意力机制建模用户兴趣：

```python
from torch_rechub.models.ranking import DIN
from torch_rechub.basic.features import SequenceFeature

# 定义序列特征
history_feas = [SequenceFeature("hist_item_id", vocab_size=item_num, embed_dim=16, pooling=None)]
target_feas = [SparseFeature("item_id", vocab_size=item_num, embed_dim=16)]

# 模型配置
model = DIN(
    features=sparse_feas + dense_feas,
    history_features=history_feas,
    target_features=target_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "dice"},
    attention_mlp_params={"dims": [64, 32], "activation": "dice"}
)

# 训练配置
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## DCN-V2 模型

DCN-V2 通过交叉网络显式建模特征交互：

```python
from torch_rechub.models.ranking import DCNv2

# 模型配置
model = DCNv2(
    features=sparse_feas + dense_feas,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# 训练配置
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## 模型评估

使用常见的排序指标进行评估：

```python
# 评估模型
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

## 特征工程技巧

1. 特征预处理
```python
# 类别特征编码
from sklearn.preprocessing import LabelEncoder
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

# 数值特征归一化
from sklearn.preprocessing import MinMaxScaler
for feat in dense_features:
    scaler = MinMaxScaler()
    df[feat] = scaler.fit_transform(df[feat].values.reshape(-1, 1))
```

2. 特征交叉
```python
# 手动特征交叉
df['cross_feat'] = df['feat1'].astype(str) + '_' + df['feat2'].astype(str)
```

## 高级应用

1. 自定义损失函数
```python
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # 实现 Focal Loss
        pass
```

2. 学习率调度
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

trainer = CTRTrainer(
    model=model,
    scheduler_fn=CosineAnnealingLR,
    scheduler_params={"T_max": 10}
)
```

## 注意事项

1. 合理处理缺失值和异常值
2. 注意特征工程的重要性
3. 选择合适的评估指标
4. 关注模型的可解释性
5. 平衡模型复杂度和效率
6. 处理样本不平衡问题

