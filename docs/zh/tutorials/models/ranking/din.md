---
title: DIN 使用示例
description: 用当前候选商品为每条历史行为分配注意力权重
---

# DIN 使用示例

## 1. 模型简介与适用场景

同一个用户既看过耳机，也看过显示器。现在候选商品是一副耳机，哪些历史记录对点击预测更有帮助？DIN 用候选商品给每条历史记录算一个权重，再把历史汇总成向量。换一个候选，历史记录的权重也会跟着变。

**论文**: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

### 模型结构

<div align="center">
  <img src="/img/models/din_arch.png" alt="DIN Model Architecture" width="600"/>
</div>

图中的 Activation Unit 接收一个候选向量和一串历史向量，输出与该候选相关的历史表示。最后再结合候选商品本身和用户特征计算点击概率。

---

## 2. 数据准备与预处理

Amazon Electronics 样本里有用户 ID、商品 ID、商品类目和交互时间。下面按时间生成样本，每一行都保留候选商品之前的商品历史和类目历史。

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

DIN 的特征分为三类：`features`（用户画像和上下文）、`target_features`（当前候选物品）、`history_features`（历史序列）。`target_features` 与 `history_features` 必须在数量、顺序和 `embed_dim` 上**一一对应**，模型内部按下标配对计算 Attention。

```python
# 1. 用户画像 / 上下文特征（不要重复放 target 特征）
features = [
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8),
]

# 2. 当前候选物品特征
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0),
]

# 3. 历史行为序列特征 (History Features)
# 注意：shared_with 参数必须指向对应的 target 特征，确保它们共享 Embedding 空间
history_features = [
    SequenceFeature(
        "hist_item_id",
        vocab_size=n_items + 1,
        embed_dim=8,
        pooling="concat",     # 必须使用 concat，交给 Activation Unit 处理
        shared_with="target_item_id", padding_idx=0
    ),
    SequenceFeature(
        "hist_cate_id",
        vocab_size=n_cates + 1,
        embed_dim=8,
        pooling="concat",     # 必须使用 concat
        shared_with="target_cate_id", padding_idx=0
    )
]
```

对历史 `[3, 0, 0]`，有效位置是 `[1, 0, 0]`。`shared_with` 让候选与历史使用同一张表，`padding_idx=0` 让注意力忽略补齐的位置。

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
    history_features=history_features,  # 历史序列
    target_features=target_features,    # 当前候选物品，会和历史逐步计算注意力
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
| `features` | `list[Feature]` | 用户画像和上下文特征，不含历史与目标特征 | |
| `history_features` | `list[Feature]` | 历史序列特征，必须是 `SequenceFeature` 且 pooling 为 `"concat"` | |
| `target_features` | `list[Feature]` | 当前候选物品特征，与 `history_features` 按下标配对 | |
| `mlp_params` | `dict` | 顶层预测 MLP 的参数（`activation` 已内置为 `dice`，无需传入） | `{"dims": [256, 128]}` |
| `attention_mlp_params` | `dict` | 目标注意力网络 (Activation Unit) 的参数（默认 `activation="dice"`, `use_softmax=False`） | `{"dims": [256, 128]}` |

商品 ID 要和商品 ID 配对，类目也要和类目配对。两个字段即使碰巧都是 8 维，若把商品和类目顺序放反，注意力计算比较的就是两种不同含义的向量。

---

## 4. 训练过程与代码示例

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("./saved/din", exist_ok=True)

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

本例中 `use_softmax=False`，注意力网络算出的权重直接参与加权求和。若改为 `True`，模型会在有效历史位置上归一化权重。比较两种设置时请保留相同的历史长度和训练数据。

---

## 7. 常见问题与解决方案

### Q1: `SequenceFeature` 为什么要用 `pooling="concat"`?
DIN 要为每个历史位置分别打分。`pooling="concat"` 保留 `[batch, seq_len, embed_dim]` 里的时间维度，注意力网络才能逐个比较候选和历史商品。

### Q2: 报错 `dimension mismatch` 或张量大小不对？
先核对两个列表的长度和顺序。如果历史的第一个字段是 `hist_item_id`，目标的第一个字段就应是 `target_item_id`，并使用相同的嵌入维度。

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
import os
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/din", exist_ok=True)

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
        SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)
    ]
    target_features = [
        SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0),
        SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0)
    ]
    history_features = [
        SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat", shared_with="target_item_id", padding_idx=0),
        SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat", shared_with="target_cate_id", padding_idx=0)
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
