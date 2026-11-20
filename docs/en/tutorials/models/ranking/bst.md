---
title: BST Tutorial
description: Append a candidate item to a user's history and predict a click with BST
---

# BST Tutorial

## 1. Model Overview and Use Cases

BST appends the candidate item to the user's browsing history and passes the combined sequence through a Transformer. The last position then represents the candidate in the context of that history. The model combines this information with user features to score the click.

**Paper**: [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874)

### Model Architecture

For example, with three history positions the candidate occupies position four. The Transformer can attend across these positions while ignoring padding in the history.

---

## 2. Data Preparation and Preprocessing

BST uses the same data preparation workflow as DIN/DIEN, based on the Amazon Electronics dataset.

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

# features contains only user-profile/context features; history and target features are paired separately
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

## 3. Model Configuration and Parameter Reference

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import BST

model = BST(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    nhead=8,         # Number of attention heads; embed_dim must be divisible by it
    dropout=0.2,     # Dropout inside the Transformer
    num_layers=1     # Number of Transformer Encoder layers
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Suggested Value |
|------|------|------|--------|
| `features` | `list[Feature]` | User-profile and context features, excluding history and target features | |
| `history_features` | `list[Feature]` | Historical behavior sequences (`pooling="concat"`) | |
| `target_features` | `list[Feature]` | Current candidate-item features; the sum of their `embed_dim` values must equal that of the history features | |
| `mlp_params` | `dict` | Hidden widths and other settings for the top prediction network. The Transformer's feed-forward layers use LeakyReLU. | `{"dims": [256, 128]}` |
| `nhead` | `int` | Number of Transformer attention heads | 4 or 8 |
| `dropout` | `float` | Dropout inside the Transformer | 0.1–0.3 |
| `num_layers` | `int` | Number of Transformer Encoder layers | 1–3 |

> **Note**: The Transformer's actual dimension is the sum of `embed_dim` across all history features. In this example it is `8 + 8 = 16`, so `nhead=8` is valid. This sum must be divisible by `nhead` and must equal the total dimension of the target features.

---

## 4. Training Process and Code Example

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

## 5. Model Evaluation and Result Analysis

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

---

## 6. Tuning Suggestions

This example uses one Transformer layer. When choosing `nhead`, first work out the width of the concatenated embeddings at each position. Here the width is 16 and `nhead=8`, so each head gets two dimensions. When changing the history length, leave a position at the end for the candidate item.

---

## 7. FAQ and Troubleshooting

### Q1: What is the core difference between BST and DIN?
- DIN uses target attention and focuses only on the relationship between the target item and the history
- BST uses self-attention to capture relationships among historical items themselves (for example, buying a phone case → buying a screen protector)

### Q2: Why does a feature-dimension and `nhead` mismatch cause an error?
The Transformer requires `sum(history_feature.embed_dim) % nhead == 0`, and the total dimensions of the history and target features must be equal. In this example, the two history features each have 8 dimensions, so the Transformer dimension is 16.

The position encoding supports up to `max_seq_len` positions, including the candidate at the end of the history. For 50 history positions, set `max_seq_len` to at least 51.

### Q3: How fast is BST for online inference?
A forward pass compares positions across the sequence. Longer histories make attention more expensive, so measure inference time on the target device at the batch size and history length you plan to use.

---

## 8. Model Visualization

Install the project's `visualization` extra and Graphviz to view the graph in Jupyter or save it as a PNG.

```python
from torch_rechub.utils.visualization import visualize_model
visualize_model(model, save_path="bst_architecture.png", dpi=300)
```

---

## 9. ONNX Export

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("bst.onnx", dynamic_batch=True)
```

---

## Complete Example

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
