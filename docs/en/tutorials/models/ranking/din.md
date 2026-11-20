---
title: DIN Tutorial
description: Use the current candidate to weight each action in the user's history
---

# DIN Tutorial

## 1. Model Overview and Use Cases

Imagine a shopper who has browsed both headphones and monitors. When the candidate is a new pair of headphones, which past interactions help predict the click? DIN scores each historical action using the candidate and combines the weighted history into one vector. A different candidate can give the same history different weights.

**Paper**: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

### Model Architecture

<div align="center">
  <img src="/img/models/din_arch.png" alt="DIN Model Architecture" width="600"/>
</div>

The Activation Unit takes a candidate vector and a sequence of history vectors. It outputs a representation of the history for that particular candidate. The model then combines it with the candidate and user features to predict a click probability.

---

## 2. Data Preparation and Preprocessing

The Amazon Electronics sample includes user IDs, item IDs, categories and interaction times. Build examples in chronological order, retaining the items and categories the user saw before each candidate.

### 2.1 Load Data and Build Sequences

```python
import numpy as np
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

# Load data
data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# Generate historical sequence features automatically
# The function sorts by time_col and creates hist_item_id and similar sequences for each sample
train, val, test = generate_seq_feature(
    data=data,
    user_col="user_id",
    item_col="item_id",
    time_col="time",
    item_attribute_cols=["cate_id"] # Also generate historical sequences for item attributes
)

# Get feature vocabulary sizes
n_users = data["user_id"].max()
n_items = data["item_id"].max()
n_cates = data["cate_id"].max()
```

### 2.2 Define Feature Lists

DIN divides its features into three groups: `features` (user-profile and context features), `target_features` (the current candidate item), and `history_features` (historical sequences). `target_features` and `history_features` must correspond **one-to-one** in number, order, and `embed_dim`; the model pairs them by index when computing attention.

```python
# 1. User-profile/context features (do not repeat target features here)
features = [
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8),
]

# 2. Current candidate-item features
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0),
]

# 3. Historical behavior sequence features (History Features)
# Note: shared_with must point to the corresponding target feature so they share an embedding space
history_features = [
    SequenceFeature(
        "hist_item_id",
        vocab_size=n_items + 1,
        embed_dim=8,
        pooling="concat",     # Must use concat so the Activation Unit can process the sequence
        shared_with="target_item_id", padding_idx=0
    ),
    SequenceFeature(
        "hist_cate_id",
        vocab_size=n_cates + 1,
        embed_dim=8,
        pooling="concat",     # Must use concat
        shared_with="target_cate_id", padding_idx=0
    )
]
```

For the history `[3, 0, 0]`, the valid-position mask is `[1, 0, 0]`. `shared_with` lets the candidate and history use the same embedding table, and `padding_idx=0` tells the attention layer to ignore the padding.

### 2.3 Build Input Dictionaries and DataLoaders

```python
# Convert DataFrames to the dictionary format accepted by the model
train_dict = df_to_dict(train)
val_dict = df_to_dict(val)
test_dict = df_to_dict(test)

train_y, val_y, test_y = train_dict.pop("label"), val_dict.pop("label"), test_dict.pop("label")

# Create DataLoaders
dg = DataGenerator(train_dict, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val_dict, y_val=val_y,
    x_test=test_dict, y_test=test_y,
    batch_size=4096
)
```

---

## 3. Model Configuration and Parameter Reference

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import DIN

model = DIN(
    features=features,
    history_features=history_features,  # Historical sequences
    target_features=target_features,    # Current candidate item, attended against the history step by step
    mlp_params={
        "dims": [256, 128]
    },
    attention_mlp_params={
        "dims": [256, 128]
    }
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Suggested Value |
|------|------|------|--------|
| `features` | `list[Feature]` | User-profile and context features, excluding history and target features | |
| `history_features` | `list[Feature]` | Historical sequence features; must be `SequenceFeature` instances with `pooling="concat"` | |
| `target_features` | `list[Feature]` | Current candidate-item features, paired with `history_features` by index | |
| `mlp_params` | `dict` | Top-level prediction MLP parameters (`activation` is fixed internally to `dice`, so do not pass it) | `{"dims": [256, 128]}` |
| `attention_mlp_params` | `dict` | Target-attention network (Activation Unit) parameters (defaults: `activation="dice"`, `use_softmax=False`) | `{"dims": [256, 128]}` |

Match item IDs with item IDs and categories with categories. Even when both fields have width 8, swapping their order makes attention compare vectors with different meanings.

---

## 4. Training Process and Code Example

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
    device="cpu", # Or "cuda:0"
    model_path="./saved/din"
)

# Start training
ctr_trainer.fit(train_dl, val_dl)
```

---

## 5. Model Evaluation and Result Analysis

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

---

## 6. Tuning Suggestions

With `use_softmax=False`, the attention weights enter the weighted sum directly. With `True`, the model normalizes the weights over valid history positions. Use the same training data and history length to compare the two settings.

---

## 7. FAQ and Troubleshooting

### Q1: Why must `SequenceFeature` use `pooling="concat"`?
DIN scores each history position separately. `pooling="concat"` preserves the time axis of `[batch, seq_len, embed_dim]` so attention can compare the candidate with each past item.

### Q2: Why do I get a `dimension mismatch` error or an unexpected tensor size?
Check that the lists have the same length and matching order. If the first history field is `hist_item_id`, the first target should be `target_item_id` with the same embedding width.

---

## 8. Model Visualization

```python
from torch_rechub.utils.visualization import visualize_model

# Generate and save the computation graph automatically
visualize_model(model, save_path="din_architecture.png", dpi=300)
```

---

## 9. ONNX Export

```python
from torch_rechub.utils.onnx_export import ONNXExporter

exporter = ONNXExporter(model, device="cpu")

# Export DIN with dynamic batch and sequence lengths
exporter.export("din.onnx", dynamic_batch=True)
```

---

## Complete Example

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

    # 1. Load data
    data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

    # 2. Generate historical sequences automatically
    train, val, test = generate_seq_feature(
        data=data, user_col="user_id", item_col="item_id",
        time_col='time', item_attribute_cols=["cate_id"]
    )

    n_users, n_items, n_cates = data["user_id"].max(), data["item_id"].max(), data["cate_id"].max()

    # 3. Define features
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

    # 4. Build data dictionaries and DataLoaders
    train, val, test = df_to_dict(train), df_to_dict(val), df_to_dict(test)
    train_y, val_y, test_y = train.pop("label"), val.pop("label"), test.pop("label")

    dg = DataGenerator(train, train_y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=val, y_val=val_y, x_test=test, y_test=test_y, batch_size=4096
    )

    # 5. Build the DIN model
    model = DIN(
        features=features,
        history_features=history_features,
        target_features=target_features,
        mlp_params={"dims": [256, 128]},
        attention_mlp_params={"dims": [256, 128]}
    )

    # 6. Train and evaluate
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
