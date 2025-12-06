---
title: 召回模型教程
description: Torch-RecHub 召回模型使用教程，包括 DSSM、GRU4Rec 和 MIND 等模型的实际示例
---

# 召回模型教程

本教程将介绍如何使用 Torch-RecHub 中的各种召回模型。我们将使用 MovieLens 数据集作为示例。

## 数据准备

首先，我们需要准备数据。MovieLens 数据集包含用户对电影的评分信息：

```python
import pandas as pd
import numpy as np
from rechub.utils import DataGenerator
from rechub.models import *
from rechub.trainers import *

# 加载数据
df = pd.read_csv("movielens.csv")
user_features = ['user_id', 'age', 'gender', 'occupation']
item_features = ['movie_id', 'genre', 'year']
```

## 基础双塔模型 (DSSM)

DSSM 是最基础的双塔模型，分别对用户和物品进行建模：

```python
# 模型配置
model = DSSM(user_features=user_features,
             item_features=item_features,
             hidden_units=[64, 32, 16],
             dropout_rates=[0.1, 0.1, 0.1])

# 训练配置
trainer = MatchTrainer(model=model,
                      mode=0,  # point-wise 训练
                      optimizer_params={'lr': 0.001},
                      n_epochs=10)

# 训练模型
trainer.fit(train_dataloader, val_dataloader)
```

## 序列推荐模型 (GRU4Rec)

GRU4Rec 通过 GRU 网络建模用户的行为序列：

```python
# 生成序列特征
seq_features = generate_seq_feature(df,
                                  user_col='user_id',
                                  item_col='movie_id',
                                  time_col='timestamp',
                                  item_attribute_cols=['genre'])

# 模型配置
model = GRU4Rec(item_num=item_num,
                hidden_size=64,
                num_layers=2,
                dropout_rate=0.1)

# 训练配置
trainer = MatchTrainer(model=model,
                      mode=1,  # pair-wise 训练
                      optimizer_params={'lr': 0.001},
                      n_epochs=10)
```

## 多兴趣模型 (MIND)

MIND 模型可以捕捉用户的多样化兴趣：

```python
# 模型配置
model = MIND(item_num=item_num,
            num_interests=4,
            hidden_size=64,
            routing_iterations=3)

# 训练配置
trainer = MatchTrainer(model=model,
                      mode=2,  # list-wise 训练
                      optimizer_params={'lr': 0.001},
                      n_epochs=10)
```

## 模型评估

使用常见的召回指标进行评估：

```python
# 计算召回率和命中率
recall_score = evaluate_recall(model, test_dataloader, k=10)
hit_rate = evaluate_hit_rate(model, test_dataloader, k=10)
print(f"Recall@10: {recall_score:.4f}")
print(f"HitRate@10: {hit_rate:.4f}")
```

## 向量检索

训练好的模型可以用于生成用户和物品的向量表示，进行快速检索：

```python
# 使用 Annoy 进行向量检索
from rechub.utils import Annoy

# 构建索引
item_vectors = model.get_item_vectors()
annoy = Annoy(metric='angular')
annoy.fit(item_vectors)

# 查询相似物品
user_vector = model.get_user_vector(user_id=1)
similar_items = annoy.query(user_vector, n=10)
```

## 高级技巧

1. 温度系数调节
```python
trainer = MatchTrainer(model=model,
                      temperature=0.2,  # 添加温度系数
                      mode=2)
```

2. 负样本采样
```python
from rechub.utils import negative_sample

neg_samples = negative_sample(items_cnt_order,
                            ratio=5,
                            method_id=1)  # Word2Vec式采样
```

3. 模型保存与加载
```python
# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model.load_state_dict(torch.load('model.pth'))
```

## 注意事项

1. 选择合适的训练模式（point-wise/pair-wise/list-wise）
2. 注意序列特征的长度和填充方式
3. 根据实际场景调整负样本比例
4. 合理设置 batch_size 和学习率
5. 使用 L2 正则化防止过拟合

