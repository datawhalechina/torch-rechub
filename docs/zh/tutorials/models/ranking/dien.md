---
title: DIEN 使用示例
description: Deep Interest Evolution Network (DIEN) 模型完整使用教程 —— 捕捉用户兴趣的动态演化
---

# DIEN 使用示例

## 1. 模型简介与适用场景

DIEN (Deep Interest Evolution Network) 是阿里妈妈在 AAAI'2019 提出的模型，是 DIN 的进化版。DIEN 引入了**兴趣抽取层 (Interest Extractor)**（GRU）和**兴趣演化层 (Interest Evolution)**（AUGRU），能够建模用户兴趣随时间的**动态演化过程**，而不仅仅是对历史行为做加权求和。

**论文**: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672)

### 模型结构

> **注意**: 由于 DIEN 内部使用 GRU 动态计算，torchview 暂时无法自动追踪其计算图，因此未提供架构可视化图。

- **Interest Extractor Layer**: 使用 GRU 对用户行为序列建模，提取每一步的兴趣状态
- **Auxiliary Loss**: 利用下一时刻的真实点击行为作为监督信号，辅助训练 GRU
- **Interest Evolution Layer**: 使用 AUGRU (Attention-based Update Gate GRU) 结合目标物品的注意力，建模与目标物品相关的兴趣演化

### 适用场景

- CTR 预估（点击率预测）
- 用户兴趣随时间有明显变化的场景（如电商、新闻推荐）
- 拥有丰富的带时序的用户行为序列数据

---

## 2. 数据准备与预处理

DIEN 的数据准备与 DIN 基本一致，同样使用 Amazon Electronics 数据集，但额外需要提供 `history_labels`（历史行为标签，标记每一步是否点击）。

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

# 加载数据
data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# 自动生成历史序列特征
train, val, test = generate_seq_feature(
    data=data, user_col="user_id", item_col="item_id",
    time_col="time", item_attribute_cols=["cate_id"]
)

n_users = data["user_id"].max()
n_items = data["item_id"].max()
n_cates = data["cate_id"].max()
```

### 定义特征

```python
# 特征列表（目标物品 + 用户属性）
features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
]
target_features = features

# 历史行为序列特征
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id"),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id")
]
```

### 构建 DataLoader

```python
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
from torch_rechub.models.ranking import DIEN

# history_labels: 标记历史行为序列中每一步是否为正反馈（1=点击，0=未点击）
# 注意：该参数是模型级别的固定配置，而非逐样本的标签
# 这里简单使用全 1 表示所有历史行为都是正反馈（简化示例）
# 实际场景中应根据业务数据设置合理的正负反馈标记
history_labels = [1] * 50  # 与序列长度一致

model = DIEN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    history_labels=history_labels,
    alpha=0.2  # 辅助损失权重
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 目标物品特征 + 用户特征，同时作为 `target_features` 传入 | |
| `history_features` | `list[Feature]` | 历史行为序列，pooling 必须为 `"concat"` | |
| `target_features` | `list[Feature]` | 与 `features` 相同 | |
| `mlp_params` | `dict` | 顶层 MLP 参数（`activation` 已内置为 `dice`，无需传入） | `{"dims": [256, 128]}` |
| `history_labels` | `list` | 历史序列中每步的点击标签 (0/1) | 长度与 seq_len 一致 |
| `alpha` | `float` | 辅助损失的权重系数 | 0.1 ~ 0.5 |

> **关键**: DIEN 的 `forward()` 返回 `(prediction, auxiliary_loss)`，训练器会自动处理辅助损失。

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
    model_path="./saved/dien"
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

1. **辅助损失权重** (`alpha`): 过大会导致主任务训练不充分，过小辅助信号太弱。建议从 `0.2` 开始调整
2. **激活函数**: MLP 默认使用 `dice`，与 DIN 一致
3. **序列长度**: DIEN 的计算随序列长度线性增长（GRU），通常限制在 20~50

---

## 7. 常见问题与解决方案

### Q1: DIEN 和 DIN 的核心区别？
- DIN 用 Activation Unit 做加权求和（静态），DIEN 用 GRU+AUGRU 建模兴趣的**时序演化**（动态）
- DIEN 额外引入辅助损失来监督 GRU 的中间状态

### Q2: history_labels 如何获取？
在实际场景中，`history_labels` 来自用户的真实行为反馈（点击/未点击）。在训练数据中，可以根据用户的实际交互记录生成。

---

## 8. 模型可视化

```python
from torch_rechub.utils.visualization import visualize_model
visualize_model(model, save_path="dien_architecture.png", dpi=300)
```

---

## 9. ONNX 导出

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("dien.onnx", dynamic_batch=True)
```

---

## 完整代码

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIEN
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
    train_y = train_dict.pop("label")
    val_y = val_dict.pop("label")
    test_y = test_dict.pop("label")

    dg = DataGenerator(train_dict, train_y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=val_dict, y_val=val_y, x_test=test_dict, y_test=test_y, batch_size=4096
    )

    history_labels = [1] * 50
    model = DIEN(
        features=features, history_features=history_features,
        target_features=target_features, mlp_params={"dims": [256, 128]},
        history_labels=history_labels, alpha=0.2
    )

    ctr_trainer = CTRTrainer(
        model, optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=2, earlystop_patience=4, device="cpu", model_path="./saved/dien/"
    )
    ctr_trainer.fit(train_dl, val_dl)

    auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
```
