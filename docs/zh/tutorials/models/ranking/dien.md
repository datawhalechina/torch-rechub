---
title: DIEN 使用示例
description: Deep Interest Evolution Network (DIEN) 模型完整使用教程 —— 捕捉用户兴趣的动态演化
---

# DIEN 使用示例

## 1. 模型简介

DIEN (Deep Interest Evolution Network) 是阿里妈妈在 AAAI'2019 提出的模型，是 DIN 的进化版。

**论文**: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672)

- **Interest Extractor Layer**: GRU 对行为序列建模，辅助损失用正负样本对监督每步隐状态（论文 Eq.7）
- **Interest Evolution Layer**: AUGRU 将注意力嵌入更新门，对全序列 softmax 归一化后演化（论文 Eq.14-16）
- **辅助损失**: $L_{aux} = -\frac{1}{N}\sum[\log\sigma(h_t \cdot e^+_{t+1}) + \log(1-\sigma(h_t \cdot e^-_{t+1}))]$
- **Padding 处理**: padding 位（index=0）不参与 GRU/AUGRU 计算；空历史样本保持零隐状态

---

## 2. 关键约定

| 约定 | 说明 |
|------|------|
| padding index | 序列用 0 填充（`generate_seq_feature` 默认），所有序列特征和 target 特征都需显式设 `padding_idx=0` |
| shared_with | `history_features` 和 `neg_history_features` 的 `shared_with` 必须指向 **target feature 的名字**，不能指向 history feature，因为 `EmbeddingLayer` 只把 `shared_with=None` 的特征注册为 embedding root |
| loss_mode | `CTRTrainer` 需设 `loss_mode=False`，因为 `forward` 返回 `(prediction, aux_loss)` |

---

## 3. 数据准备

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_rechub.utils.data import generate_seq_feature, df_to_dict

raw = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# 复现 generate_seq_feature 内部的编码，用于构建 item2cate 映射
enc_data = raw.copy()
for feat in enc_data:
    le = LabelEncoder()
    enc_data[feat] = le.fit_transform(enc_data[feat]) + 1
enc_data = enc_data.astype('int32')
item2cate = enc_data[['item_id','cate_id']].drop_duplicates().set_index('item_id')['cate_id'].to_dict()
n_items, n_users, n_cates = enc_data['item_id'].max(), enc_data['user_id'].max(), enc_data['cate_id'].max()

train, val, test = generate_seq_feature(
    data=raw, user_col="user_id", item_col="item_id",
    time_col="time", item_attribute_cols=["cate_id"]
)
```

---

## 4. 负样本构造

```python
import random
import numpy as np

def build_neg_history(split, hist_item_col, item2cate, n_items):
    """逐时刻采样负 item，再通过 item2cate 查出对应 cate，保证属性一致。"""
    seqs = split[hist_item_col]
    neg_items = np.zeros_like(seqs)
    neg_cates = np.zeros_like(seqs)
    for i, row in enumerate(seqs):
        for t, item in enumerate(row):
            if item == 0:
                continue
            neg = item
            while neg == item:
                neg = random.randint(1, n_items)
            neg_items[i, t] = neg
            neg_cates[i, t] = item2cate.get(neg, 1)
    return neg_items, neg_cates

train, val, test = df_to_dict(train), df_to_dict(val), df_to_dict(test)
train_y, val_y, test_y = train.pop("label"), val.pop("label"), test.pop("label")

for split in [train, val, test]:
    neg_items, neg_cates = build_neg_history(split, "hist_item_id", item2cate, n_items)
    split["neg_hist_item_id"] = neg_items
    split["neg_hist_cate_id"] = neg_cates
```

---

## 5. 特征定义

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature

features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]

# padding_idx=0 必须设在 target_features 上，因为 embedding 表由它创建
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0),
]

history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id", padding_idx=0),
]
neg_history_features = [
    SequenceFeature("neg_hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
    SequenceFeature("neg_hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id", padding_idx=0),
]
```

---

## 6. 创建模型与训练

```python
import os
from torch_rechub.models.ranking import DIEN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

os.makedirs("./saved/dien", exist_ok=True)

dg = DataGenerator(train, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val, y_val=val_y, x_test=test, y_test=test_y, batch_size=4096
)

model = DIEN(
    features=features,
    history_features=history_features,
    neg_history_features=neg_history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    alpha=0.2,
)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=5,
    earlystop_patience=2,
    device="cpu",
    model_path="./saved/dien",
    loss_mode=False,  # forward 返回 (prediction, aux_loss)
)
trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

---

## 7. 常见问题

### Q1: DIEN 和 DIN 的核心区别？
DIN 用 Activation Unit 做加权求和（静态），DIEN 用 GRU+AUGRU 建模兴趣的时序演化（动态），并引入辅助损失监督 GRU 中间状态。

### Q2: shared_with 为什么要指向 target feature 而不是 history feature？
`EmbeddingLayer` 只把 `shared_with=None` 的特征注册进 `embed_dict`。`history_features` 本身已经 `shared_with="target_*"`，所以 `embed_dict` 里没有 `hist_*` 这个 key，`neg_history_features` 的 `shared_with` 必须直接指向 target feature。

### Q3: 为什么 target_features 也要设 padding_idx=0？
`history_features` 和 `neg_history_features` 共享 target feature 的 embedding 表。只有在 target feature 上设 `padding_idx=0`，embedding 表的 row 0 才真正是受保护的零向量。

### Q4: 序列长度建议？
DIEN 计算随序列长度线性增长（GRU 逐步展开），通常限制在 20~50。

---

## 完整代码

完整可运行示例见 [examples/ranking/run_dien.py](../../../../../examples/ranking/run_dien.py)。
