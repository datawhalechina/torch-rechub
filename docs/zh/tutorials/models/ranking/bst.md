---
title: BST 使用示例
description: Behavior Sequence Transformer (BST) 模型完整使用教程 —— 用 Transformer 建模用户行为序列
---

# BST 使用示例

## 1. 模型简介与适用场景

BST (Behavior Sequence Transformer) 是阿里巴巴在 2019 年提出的模型，将 **Transformer** 的 Self-Attention 机制引入推荐系统，通过多头注意力捕捉用户行为序列中**任意两个物品间的依赖关系**，而非像 DIN 那样仅关注目标物品与历史物品的关系。

**论文**: [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874)

### 模型结构

> **注意**: 由于 BST 内部使用 Transformer 动态计算，torchview 暂时无法自动追踪其计算图，因此未提供架构可视化图。

- **Embedding Layer**: 将用户特征、物品特征和行为序列编码为 Embedding
- **Transformer Encoder**: 对行为序列 + 目标物品拼接后做 Self-Attention
- **MLP Layer**: 将 Transformer 输出和其他特征拼接后，经 MLP 输出预测分数

### 适用场景

- CTR 预估
- 行为序列较长且物品间存在复杂依赖的场景
- 追求更好的序列建模能力（相比 DIN/DIEN 等 RNN-based 方法）

---

## 2. 数据准备与预处理

BST 的数据准备流程与 DIN/DIEN 一致，使用 Amazon Electronics 数据集。

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

train, val, test = generate_seq_feature(
    data=data, user_col="user_id", item_col="item_id",
    time_col="time", item_attribute_cols=["cate_id"]
)

n_users = data["user_id"].max()
n_items = data["item_id"].max()
n_cates = data["cate_id"].max()

# 特征定义（与 DIN 相同的模式）
features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
]
target_features = features
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id"),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id")
]

# DataLoader
train_dict, val_dict, test_dict = df_to_dict(train), df_to_dict(val), df_to_dict(test)
train_y = train_dict.pop("label")
val_y = val_dict.pop("label")
test_y = test_dict.pop("label")

dg = DataGenerator(train_dict, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val_dict, y_val=val_y, x_test=test_dict, y_test=test_y, batch_size=4096
)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.ranking import BST

model = BST(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    nhead=8,         # 多头注意力头数
    dropout=0.2,     # Transformer 内部 dropout
    num_layers=1     # Transformer Encoder 层数
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 目标物品特征 + 用户特征，同时作为 `target_features` 传入 | |
| `history_features` | `list[Feature]` | 历史行为序列 (pooling=`"concat"`) | |
| `target_features` | `list[Feature]` | 与 `features` 相同 | |
| `mlp_params` | `dict` | 顶层 MLP 参数（`activation` 已内置为 `leakyrelu`，无需传入） | `{"dims": [256, 128]}` |
| `nhead` | `int` | Transformer 多头注意力头数 | 4 或 8 |
| `dropout` | `float` | Transformer 内部 dropout | 0.1 ~ 0.3 |
| `num_layers` | `int` | Transformer Encoder 层数 | 1 ~ 3 |

> **注意**: `embed_dim` 必须能被 `nhead` 整除。例如 `embed_dim=8` 时，`nhead` 可以为 1, 2, 4, 8。

---

## 4. 训练过程与代码示例

```python
from torch_rechub.trainers import CTRTrainer

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=5,
    earlystop_patience=2,
    device="cpu",
    model_path="./saved/bst"
)

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

1. **Transformer 层数** (`num_layers`): 1 层通常已经足够好，增加层数可能导致过拟合
2. **多头数** (`nhead`): 需要能整除 `embed_dim`，通常 4 或 8
3. **Dropout**: Transformer 容易过拟合，推荐 0.2~0.3
4. **序列长度**: BST 的计算复杂度是 $O(n^2)$（Self-Attention），序列过长会显著增加延迟

---

## 7. 常见问题与解决方案

### Q1: BST 和 DIN 的核心区别？
- DIN 用 Target Attention 只关注目标物品与历史的关系
- BST 用 Self-Attention 能捕捉历史物品之间的相互关系（如买了手机壳 → 买了贴膜）

### Q2: embed_dim 和 nhead 不匹配导致报错？
Transformer 要求 `embed_dim % nhead == 0`。如果特征 `embed_dim=8`，则 `nhead` 只能是 1, 2, 4, 8。

### Q3: BST 线上推理速度如何？
Self-Attention 计算量为 $O(n^2d)$，短序列（<50）延迟可接受。长序列建议截断或使用 DIN 替代。

---

## 8. 模型可视化

```python
from torch_rechub.utils.visualization import visualize_model
visualize_model(model, save_path="bst_architecture.png", dpi=300)
```

---

## 9. ONNX 导出

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("bst.onnx", dynamic_batch=True)
```

---

## 完整代码

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import BST
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature


def main():
    torch.manual_seed(2022)

    data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")
    train, val, test = generate_seq_feature(
        data=data, user_col="user_id", item_col="item_id",
        time_col="time", item_attribute_cols=["cate_id"]
    )
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()

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

    train_dict, val_dict, test_dict = df_to_dict(train), df_to_dict(val), df_to_dict(test)
    train_y, val_y, test_y = train_dict.pop("label"), val_dict.pop("label"), test_dict.pop("label")

    dg = DataGenerator(train_dict, train_y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=val_dict, y_val=val_y, x_test=test_dict, y_test=test_y, batch_size=4096
    )

    model = BST(
        features=features, history_features=history_features, target_features=target_features,
        mlp_params={"dims": [256, 128]}, nhead=8, dropout=0.2, num_layers=1
    )

    ctr_trainer = CTRTrainer(
        model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=2, earlystop_patience=4, device="cpu", model_path="./saved/bst/"
    )
    ctr_trainer.fit(train_dl, val_dl)

    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
```
