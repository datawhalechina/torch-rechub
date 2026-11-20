---
title: BST 使用示例
description: 用 BST 把候选商品接入用户的行为序列，并计算点击概率
---

# BST 使用示例

## 1. 模型简介与适用场景

BST 把候选商品接在用户的浏览历史末尾，一起送入 Transformer。最后一个位置的输出就包含了候选商品和这段历史的关系。模型再结合用户特征，给这次候选打一个点击分数。

**论文**: [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874)

### 模型结构

例如，若输入有 3 个历史位置，候选商品就占第 4 个位置。Transformer 可以在这些位置之间计算注意力，补齐的历史位置会被遮住。

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

# features 只是用户画像/上下文；历史与目标特征另行配对
features = [
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
]
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8)
]
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
    nhead=8,         # 多头注意力头数；embed_dim 必须能被它整除
    dropout=0.2,     # Transformer 内部 dropout
    num_layers=1     # Transformer Encoder 层数
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 用户画像和上下文特征，不含历史与目标特征 | |
| `history_features` | `list[Feature]` | 历史行为序列 (pooling=`"concat"`) | |
| `target_features` | `list[Feature]` | 当前候选物品特征，其 `embed_dim` 总和必须等于历史特征的总和 | |
| `mlp_params` | `dict` | 顶层预测网络的隐藏层宽度等参数。Transformer 内部的前馈层使用 LeakyReLU | `{"dims": [256, 128]}` |
| `nhead` | `int` | Transformer 多头注意力头数 | 4 或 8 |
| `dropout` | `float` | Transformer 内部 dropout | 0.1 ~ 0.3 |
| `num_layers` | `int` | Transformer Encoder 层数 | 1 ~ 3 |

> **注意**: Transformer 的实际维度是所有历史特征 `embed_dim` 的总和。本例为 `8 + 8 = 16`，因此 `nhead=8` 可用。该总和必须能被 `nhead` 整除，并且必须等于目标特征的维度总和。

---

## 4. 训练过程与代码示例

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("./saved/bst", exist_ok=True)

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

本例用一层 Transformer。调整 `nhead` 时先算出每个位置拼接后的向量宽度。例如本例是 16 维，设 `nhead=8`，每个头分到 2 维。调整历史长度时要给序列末尾的候选商品额外留一个位置。

---

## 7. 常见问题与解决方案

### Q1: BST 和 DIN 的核心区别？
- DIN 用 Target Attention 只关注目标物品与历史的关系
- BST 用 Self-Attention 能捕捉历史物品之间的相互关系（如买了手机壳 → 买了贴膜）

### Q2: 特征维度和 nhead 不匹配导致报错？
Transformer 要求 `sum(history_feature.embed_dim) % nhead == 0`，而且历史特征与目标特征的维度总和要相等。本例两个历史特征各为 8 维，Transformer 维度是 16。

位置编码最多接收 `max_seq_len` 个位置，输入会把目标接在历史末尾。若历史长度为 50，`max_seq_len` 至少设为 51。

### Q3: BST 线上推理速度如何？
一次前向计算需要比较序列中的各个位置，序列越长，注意力的计算开销越大。部署前用实际批大小、历史长度和设备测推理耗时。

---

## 8. 模型可视化

安装项目的 `visualization` extra 和 Graphviz 后，可以在 Jupyter 查看计算图，也可以将图保存为 PNG。

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
import os
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import BST
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/bst", exist_ok=True)

    data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")
    train, val, test = generate_seq_feature(
        data=data, user_col="user_id", item_col="item_id",
        time_col="time", item_attribute_cols=["cate_id"]
    )
    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()

    features = [
        SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
    ]
    target_features = [
        SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
        SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8)
    ]
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
