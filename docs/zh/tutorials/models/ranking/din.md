---
title: DIN 使用示例
description: Deep Interest Network (DIN) 模型完整使用教程 —— 擅长捕捉用户历史行为中的多样化兴趣
---

# DIN 使用示例

## 1. 模型简介与适用场景

DIN (Deep Interest Network) 是阿里妈妈在 KDD'2018 提出的经典推荐模型。针对电商场景中用户兴趣的**多样性**和**局部激活**特征（用户当前的点击通常只与其历史行为中的一小部分相关），DIN 引入了**目标注意力机制 (Target Attention/Activation Unit)**，根据当前候选物品（Target Item），动态计算用户历史行为序列的权重，从而自适应地捕捉用户的动态兴趣。

**论文**: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

### 模型结构

<div align="center">
  <img src="/img/models/din_arch.png" alt="DIN Model Architecture" width="600"/>
</div>

- **Base 模型**: 类似于一个标准的 Embedding + MLP 结构
- **Activation Unit**: DIN 的核心。利用目标物品特征与用户历史序列特征计算 Attention 分数（权重），将历史序列聚合成一个固定维度的表示。
- **Dice 激活函数**: 论文提出的一种数据依赖的激活函数，能自适应地调整修正点，优于 PReLU。

### 适用场景

- CTR 预估（点击率预测）
- 电商推荐排序
- 拥有丰富且较长的**用户行为序列**数据（如浏览历史、点击历史等）
- 候选物品种类繁多，用户兴趣分散的场景

---

## 2. 数据准备与预处理

本示例使用 **Amazon Electronics** 样本数据集。该数据集包含了用户对商品的交互行为以及时间戳。主要构建了用户的商品历史序列、类目历史序列。

### 2.1 加载和构建序列数据

```python
import numpy as np
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

# 加载数据
data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# 自动生成历史序列特征
# 该函数会根据 time_col 排序，为每个样本生成 hist_item_id 等序列
train, val, test = generate_seq_feature(
    data=data, 
    user_col="user_id", 
    item_col="item_id", 
    time_col="time", 
    item_attribute_cols=["cate_id"] # 同时生成 item 属性的历史序列
)

# 获取特征词表大小
n_users = data["user_id"].max()
n_items = data["item_id"].max()
n_cates = data["cate_id"].max()
```

### 2.2 定义特征列表

DIN 的特征分为三类：`features`（包含目标物品特征和用户特征）、`target_features`（与 features 相同）、`history_features`（历史序列特征）。`target_features` 与 `history_features` 必须**一一对应**计算 Attention。

```python
# 1. 特征列表（目标物品 + 用户属性）
features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
]
target_features = features

# 2. 历史行为序列特征 (History Features)
# 注意：shared_with 参数必须指向对应的 target 特征，确保它们共享 Embedding 空间
history_features = [
    SequenceFeature(
        "hist_item_id",
        vocab_size=n_items + 1,
        embed_dim=8,
        pooling="concat",     # 必须使用 concat，交给 Activation Unit 处理
        shared_with="target_item_id"
    ),
    SequenceFeature(
        "hist_cate_id",
        vocab_size=n_cates + 1,
        embed_dim=8,
        pooling="concat",     # 必须使用 concat
        shared_with="target_cate_id"
    )
]
```

### 2.3 构建输入字典和 DataLoader

```python
# 将 DataFrame 转换为模型可接受的 dict 格式
train_dict = df_to_dict(train)
val_dict = df_to_dict(val)
test_dict = df_to_dict(test)

train_y, val_y, test_y = train_dict.pop("label"), val_dict.pop("label"), test_dict.pop("label")

# 创建 DataLoader
dg = DataGenerator(train_dict, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val_dict, y_val=val_y, 
    x_test=test_dict, y_test=test_y, 
    batch_size=4096
)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.ranking import DIN

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={
        "dims": [256, 128]
    },
    attention_mlp_params={
        "dims": [256, 128]
    }
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 目标物品特征 + 用户特征，同时作为 `target_features` 传入 | |
| `history_features` | `list[Feature]` | 历史序列特征，必须是 `SequenceFeature` 且 pooling 为 `"concat"` | |
| `target_features` | `list[Feature]` | 与 `features` 相同，用于与历史做 Attention | |
| `mlp_params` | `dict` | 顶层预测 MLP 的参数（`activation` 已内置为 `dice`，无需传入） | `{"dims": [256, 128]}` |
| `attention_mlp_params` | `dict` | 目标注意力网络 (Activation Unit) 的参数（默认 `activation="dice"`, `use_softmax=False`） | `{"dims": [256, 128]}` |

> **关键提醒**: `history_features` 和 `target_features` 应当在物理意义上成对出现（如 `hist_item_id` 对应 `target_item_id`），并且必须通过 `shared_with` 共享 Embedding 表！

---

## 4. 训练过程与代码示例

```python
from torch_rechub.trainers import CTRTrainer

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={
        "lr": 1e-3,
        "weight_decay": 1e-3
    },
    n_epoch=5,
    earlystop_patience=2,
    device="cpu", # 或 "cuda:0"
    model_path="./saved/din"
)

# 开始训练
ctr_trainer.fit(train_dl, val_dl)
```

---

## 5. 模型评估与结果分析

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

---

## 6. 参数调优建议

1. **激活函数选择**：DIN 论文提出了 `Dice` 激活函数，通常比默认的 `ReLU` 或 `PReLU` 在大规模稀疏数据上表现更好。
2. **Attention Softmax**：`attention_mlp_params` 中的 `use_softmax=False` 是论文中的设计（允许聚合后的向量长度动态变化，表示兴趣强度的总和）。
3. **序列长度限制**：历史序列过长会增加计算延迟。线上通常截断最近的 20~50 次交互记录。
4. **共享 Embedding**：确保历史 ID 和目标 ID 使用了同一个 LookUp Table，这是 Attention 起作用的物理基础。

---

## 7. 常见问题与解决方案

### Q1: `SequenceFeature` 为什么要用 `pooling="concat"`?
因为 DIN 需要自己处理序列融合，它需要拿到 `[batch, seq_len, embed_dim]` 这样 3D 格式的张量丢给 Activation Unit，而不是先做平均 (`mean` pooling)。

### Q2: 报错 `dimension mismatch` 或张量大小不对？
检查 `target_features` 和 `history_features` 传入模型的顺序和数量是否严格对应。模型内部是基于 zip 的逻辑将第 $i$ 个 `history_feature` 与第 $i$ 个 `target_feature` 进行 Attention 计算的。

---

## 8. 模型可视化

```python
from torch_rechub.utils.visualization import visualize_model

# 自动生成计算图并保存
visualize_model(model, save_path="din_architecture.png", dpi=300)
```

---

## 9. ONNX 导出

```python
from torch_rechub.utils.onnx_export import ONNXExporter

exporter = ONNXExporter(model, device="cpu")

# 导出 DIN，支持动态 batch 和 sequence lengths
exporter.export("din.onnx", dynamic_batch=True)
```

---

## 完整代码

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature


def main():
    torch.manual_seed(2022)

    # 1. 加载数据
    data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

    # 2. 自动生成历史序列
    train, val, test = generate_seq_feature(
        data=data, user_col="user_id", item_col="item_id", 
        time_col='time', item_attribute_cols=["cate_id"]
    )

    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()

    # 3. 定义特征
    features = [
        SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
        SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
        SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
    ]
    target_features = features
    history_features = [
        SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat", shared_with="target_item_id"),
        SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat", shared_with="target_cate_id")
    ]

    # 4. 构建数据字典和 DataLoader
    train, val, test = df_to_dict(train), df_to_dict(val), df_to_dict(test)
    train_y, val_y, test_y = train.pop("label"), val.pop("label"), test.pop("label")

    dg = DataGenerator(train, train_y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=val, y_val=val_y, x_test=test, y_test=test_y, batch_size=4096
    )

    # 5. 构建 DIN 模型
    model = DIN(
        features=features,
        history_features=history_features,
        target_features=target_features,
        mlp_params={"dims": [256, 128]},
        attention_mlp_params={"dims": [256, 128]}
    )

    # 6. 训练与评估
    ctr_trainer = CTRTrainer(
        model, 
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3}, 
        n_epoch=2, earlystop_patience=4, device="cpu", model_path="./saved/din/"
    )
    ctr_trainer.fit(train_dl, val_dl)
    
    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
    print(f'Test AUC: {auc:.4f}')

if __name__ == '__main__':
    main()
```
