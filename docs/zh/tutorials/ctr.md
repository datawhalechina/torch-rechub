---
title: 排序模型教程
description: Torch-RecHub 排序模型教程，包括 Wide & Deep、DeepFM、DIN 和 DCN-V2 等模型及特征工程技巧
---

# 排序模型教程

本教程将介绍如何使用 Torch-RecHub 中的各种排序模型。我们将使用 Criteo 和 Avazu 数据集作为示例。

## 数据准备

首先，我们需要准备数据并进行特征处理：

```python
import pandas as pd
import numpy as np
from rechub.utils import DataGenerator
from rechub.models import *
from rechub.trainers import *

# 加载数据
df = pd.read_csv("criteo_sample.csv")

# 特征列定义
sparse_features = ['C1', 'C2', 'C3', ..., 'C26']
dense_features = ['I1', 'I2', 'I3', ..., 'I13']
features = sparse_features + dense_features
```

## Wide & Deep 模型

Wide & Deep 模型结合了记忆和泛化能力：

```python
# 模型配置
model = WideDeep(
    wide_features=sparse_features,
    deep_features=features,
    hidden_units=[256, 128, 64],
    dropout_rates=[0.1, 0.1, 0.1])

# 训练配置
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10,
                 device='cuda:0')

# 训练模型
trainer.fit(train_dataloader, val_dataloader)
```

## DeepFM 模型

DeepFM 模型通过因子分解机和深度网络建模特征交互：

```python
# 模型配置
model = DeepFM(
    features=features,
    hidden_units=[256, 128, 64],
    dropout_rates=[0.1, 0.1, 0.1],
    embedding_dim=16)

# 训练配置
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10)
```

## DIN (Deep Interest Network)

DIN 模型通过注意力机制建模用户兴趣：

```python
# 生成行为序列特征
behavior_features = ['item_id', 'category_id']
seq_features = generate_seq_feature(df,
                                  user_col='user_id',
                                  item_col='item_id',
                                  time_col='timestamp',
                                  item_attribute_cols=['category_id'])

# 模型配置
model = DIN(
    features=features,
    behavior_features=behavior_features,
    attention_units=[80, 40],
    hidden_units=[256, 128, 64],
    dropout_rate=0.1)

# 训练配置
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10)
```

## DCN-V2 模型

DCN-V2 通过交叉网络显式建模特征交互：

```python
# 模型配置
model = DCNV2(
    features=features,
    cross_num=3,
    hidden_units=[256, 128, 64],
    dropout_rates=[0.1, 0.1, 0.1],
    cross_parameterization='matrix')  # 或 'vector'

# 训练配置
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10)
```

## 模型评估

使用常见的排序指标进行评估：

```python
# 评估模型
auc = evaluate_auc(model, test_dataloader)
log_loss = evaluate_logloss(model, test_dataloader)
print(f"AUC: {auc:.4f}")
print(f"LogLoss: {log_loss:.4f}")
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
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # 实现 Focal Loss
        pass

trainer = Trainer(model=model,
                 loss_fn=FocalLoss(alpha=0.25, gamma=2))
```

2. 学习率调度
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

trainer = Trainer(model=model,
                 scheduler='cosine',  # 使用余弦退火调度
                 scheduler_params={'T_max': 10})
```

## 注意事项

1. 合理处理缺失值和异常值
2. 注意特征工程的重要性
3. 选择合适的评估指标
4. 关注模型的可解释性
5. 平衡模型复杂度和效率
6. 处理样本不平衡问题

