---
title: DIEN 使用示例
description: 用商品交互序列训练 DIEN，并构造辅助损失需要的负历史
---

# DIEN 使用示例

## 1. 模型简介

DIEN 按先后顺序读取用户的行为历史，每读一步就更新一次兴趣表示。训练时，它还会比较用户下一步实际交互的商品和采样得到的负例。这个额外的训练目标称为辅助损失。

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
| shared_with | `history_features` 和 `neg_history_features` 的 `shared_with` 必须指向 **target feature 的名字**，不能指向 history feature，因为 在这段代码里，`target_item_id` 创建商品嵌入表，两列历史 ID 都要到这张表查向量。`hist_item_id` 已经指向 `target_item_id`，因此负历史也直接写 `shared_with="target_item_id"`。

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

主任务的 `label` 表示是否点击当前候选商品。这里另外构造的 `neg_hist_item_id` 对应历史序列的各个位置，用于训练辅助任务。负商品的类目从 `item2cate` 查，保证商品和类目配对。

```python
from examples.ranking.run_dien import build_neg_history

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
DIN 根据当前候选给历史记录分配权重。DIEN 还按时间顺序更新用户状态，并用下一步的真实商品和负样本来训练中间状态。

### Q2: shared_with 为什么要指向 target feature 而不是 history feature？
`EmbeddingLayer` 只把 `shared_with=None` 的特征注册进 `embed_dict`。`history_features` 本身已经 `shared_with="target_*"`，所以 `embed_dict` 里没有 `hist_*` 这个 key，`neg_history_features` 的 `shared_with` 必须直接指向 target feature。

### Q3: 为什么 target_features 也要设 padding_idx=0？
`history_features` 和 `neg_history_features` 共享 target feature 的 embedding 表。只有在 target feature 上设 `padding_idx=0`，embedding 表的 row 0 才真正是受保护的零向量。

### Q4: 序列长度建议？
GRU 会按顺序读取每一次历史行为。选择截断长度时，可以先看训练集中每位用户有多少次交互，再用几个候选长度比较验证 AUC 和训练耗时。

---

## 完整代码

完整可运行示例见 [examples/ranking/run_dien.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/run_dien.py)。
