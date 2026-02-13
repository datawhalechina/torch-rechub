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
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input

# 加载数据
df = pd.read_csv("movielens.csv")

# 定义用户特征
user_features = [
    SparseFeature("user_id", vocab_size=user_num, embed_dim=16),
    SparseFeature("gender", vocab_size=3, embed_dim=16),
    SparseFeature("age", vocab_size=10, embed_dim=16),
    SparseFeature("occupation", vocab_size=25, embed_dim=16)
]
# 添加用户历史序列特征
user_features += [
    SequenceFeature("hist_movie_id", vocab_size=item_num, embed_dim=16,
                    pooling="mean", shared_with="movie_id")
]

# 定义物品特征
item_features = [
    SparseFeature("movie_id", vocab_size=item_num, embed_dim=16),
    SparseFeature("cate_id", vocab_size=cate_num, embed_dim=16)
]
```

## 基础双塔模型 (DSSM)

DSSM 是最基础的双塔模型，分别对用户和物品进行建模：

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

# 模型配置
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"}
)

# 训练配置
trainer = MatchTrainer(
    model=model,
    mode=0,  # point-wise 训练
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)

# 训练模型
trainer.fit(train_dl, val_dl)
```

## 序列推荐模型 (GRU4Rec)

GRU4Rec 通过 GRU 网络建模用户的行为序列：

```python
from torch_rechub.models.matching import GRU4Rec
from torch_rechub.basic.features import SequenceFeature

# 定义序列特征
history_features = [SequenceFeature("hist_movie_id", vocab_size=item_num, embed_dim=16, pooling=None)]
neg_item_feature = [SparseFeature("neg_items", vocab_size=item_num, embed_dim=16)]

# 模型配置
model = GRU4Rec(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64], "num_layers": 2, "dropout": 0.2},
    temperature=1.0
)

# 训练配置
trainer = MatchTrainer(
    model=model,
    mode=2,  # list-wise 训练
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## 多兴趣模型 (MIND)

MIND 模型可以捕捉用户的多样化兴趣：

```python
from torch_rechub.models.matching import MIND

# 模型配置
model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    temperature=1.0,
    interest_num=4
)

# 训练配置
trainer = MatchTrainer(
    model=model,
    mode=2,  # list-wise 训练
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## 模型评估

使用常见的召回指标进行评估：

```python
# 评估模型
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")

# 获取用户/物品向量用于离线评估
user_embeddings = trainer.inference_embedding(model, "user", user_dl, model_path="./")
item_embeddings = trainer.inference_embedding(model, "item", item_dl, model_path="./")
```

## 向量检索

训练好的模型可以用于生成用户和物品的向量表示，进行快速检索：

```python
from torch_rechub.utils.match import Annoy

# 构建 Annoy 索引
annoy = Annoy(n_trees=10)
annoy.fit(item_embeddings)

# 查询相似物品
similar_items, scores = annoy.query(user_embeddings[0], topk=10)
print(f"Top 10 similar items: {similar_items}")
```

## 高级技巧

1. 温度系数调节
```python
# 温度系数影响相似度分数的分布
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,  # 较小的温度使分布更尖锐
    user_params={"dims": [256, 128, 64]},
    item_params={"dims": [256, 128, 64]}
)
```

2. 正则化配置
```python
trainer = MatchTrainer(
    model=model,
    mode=0,
    regularization_params={
        "embedding_l2": 1e-5,
        "dense_l2": 1e-5
    }
)
```

3. 模型保存与加载
```python
import torch

# 保存模型
torch.save(model.state_dict(), "model.pth")

# 加载模型
model.load_state_dict(torch.load("model.pth"))
```

## 注意事项

1. 选择合适的训练模式（point-wise/pair-wise/list-wise）
2. 注意序列特征的长度和填充方式
3. 根据实际场景调整负样本比例
4. 合理设置 batch_size 和学习率
5. 使用 L2 正则化防止过拟合

