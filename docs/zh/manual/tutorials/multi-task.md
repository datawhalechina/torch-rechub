---
title: 多任务学习教程
description: Torch-RecHub 多任务学习模型教程，包括 SharedBottom、ESMM、MMoE 和 PLE 等模型示例
---

# 多任务学习教程

本教程将介绍如何使用 Torch-RecHub 中的多任务学习模型。我们将使用阿里巴巴的电商数据集作为示例。

## 数据准备

首先，我们需要准备多任务学习的数据：

```python
import pandas as pd
import numpy as np
from rechub.utils import DataGenerator
from rechub.models import *
from rechub.trainers import *

# 加载数据
df = pd.read_csv("ali_ccp_data.csv")

# 特征定义
user_features = ['user_id', 'age', 'gender', 'occupation']
item_features = ['item_id', 'category_id', 'shop_id', 'brand_id']
features = user_features + item_features

# 多任务标签
tasks = ['click', 'conversion']  # CTR 和 CVR 任务
```

## SharedBottom 模型

最基础的多任务学习模型，底层网络共享参数：

```python
# 模型配置
model = SharedBottom(
    features=features,
    hidden_units=[256, 128],
    task_hidden_units=[64, 32],
    num_tasks=2,
    task_types=['binary', 'binary'])

# 训练配置
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)

# 训练模型
trainer.fit(train_dataloader, val_dataloader)
```

## ESMM (Entire Space Multi-Task Model)

解决样本选择偏差的多任务模型：

```python
# 模型配置
model = ESMM(
    features=features,
    hidden_units=[256, 128, 64],
    tower_units=[32, 16],
    embedding_dim=16)

# 训练配置
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)
```

## MMoE (Multi-gate Mixture-of-Experts)

通过专家机制实现任务间的软参数共享：

```python
# 模型配置
model = MMoE(
    features=features,
    expert_units=[256, 128],
    num_experts=8,
    num_tasks=2,
    expert_activation='relu',
    gate_activation='softmax')

# 训练配置
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)
```

## PLE (Progressive Layered Extraction)

通过分层提取更好地建模任务关系：

```python
# 模型配置
model = PLE(
    features=features,
    expert_units=[256, 128],
    num_experts=4,
    num_layers=3,
    num_shared_experts=2,
    task_types=['binary', 'binary'])

# 训练配置
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)
```

## 任务权重优化

### GradNorm

使用 GradNorm 算法动态调整任务权重：

```python
# 配置 GradNorm
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    task_weights_strategy='gradnorm',
    gradnorm_alpha=1.5)
```

### MetaBalance

使用 MetaBalance 优化器平衡任务梯度：

```python
from rechub.utils import MetaBalance

# 配置 MetaBalance 优化器
optimizer = MetaBalance(
    model.parameters(),
    relax_factor=0.7,
    beta=0.9)

trainer = MTLTrainer(
    model=model,
    optimizer=optimizer)
```

## 模型评估

针对不同任务使用相应的评估指标：

```python
# 评估模型
results = evaluate_multi_task(model, test_dataloader)
for task, metrics in results.items():
    print(f"Task: {task}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"LogLoss: {metrics['logloss']:.4f}")
```

## 高级应用

1. 自定义任务损失权重
```python
trainer = MTLTrainer(
    model=model,
    task_weights=[1.0, 0.5])  # 设置固定任务权重
```

2. 获取共享层和任务特定层
```python
from rechub.utils import shared_task_layers

shared_params, task_params = shared_task_layers(model)
```

3. 任务特定的学习率
```python
trainer = MTLTrainer(
    model=model,
    task_specific_lr={'click': 0.001, 'conversion': 0.0005})
```

## 注意事项

1. 选择合适的多任务学习架构
2. 注意任务之间的相关性
3. 处理任务间的数据不平衡
4. 合理设置任务权重
5. 监控每个任务的训练进度
6. 防止任务间的负迁移
7. 考虑计算资源的限制

