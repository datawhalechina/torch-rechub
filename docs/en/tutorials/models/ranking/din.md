---
title: DIN Tutorial
description: "Complete Deep Interest Network tutorial: target-aware sequence modeling for CTR prediction"
---

# DIN Tutorial

## 1. Model Overview and Use Cases

DIN (Deep Interest Network) is a classic recommendation model proposed by Alibaba at KDD 2018. It focuses on the **diversity** and **local activation** of user interests: only part of a user's history is relevant to the current candidate item. DIN introduces a **target attention mechanism (Activation Unit)** to dynamically weight historical behaviors according to the current target item.

**Paper**: [Deep Interest Network for Click-Through Rate Prediction](https://arxiv.org/abs/1706.06978)

### Model Architecture

<div align="center">
  <img src="/img/models/din_arch.png" alt="DIN Model Architecture" width="600"/>
</div>

- **Base model**: standard embedding + MLP backbone
- **Activation Unit**: computes attention scores between the target item and each historical behavior
- **Dice activation**: a data-adaptive activation used in the original paper

### Suitable Scenarios

- CTR prediction
- E-commerce ranking
- Scenarios with rich and long user behavior sequences
- Candidate sets where user interest is diverse and highly item-dependent

---

## 2. Data Preparation and Preprocessing

This example uses the sampled **Amazon Electronics** dataset and builds user history sequences for item ids and category ids.

### 2.1 Load Data and Build Sequence Features

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

# Load data
data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# Automatically generate historical sequence features ordered by time.
train, val, test = generate_seq_feature(
    data=data,
    user_col="user_id",
    item_col="item_id",
    time_col="timestamp",
    item_attribute_cols=["cate_id"],
    min_item=0,
    shuffle=True,
)

# Get vocabulary sizes used by sparse and sequence features.
user_num = data["user_id"].max() + 1
item_num = data["item_id"].max() + 1
cate_num = data["cate_id"].max() + 1
```

### 2.2 Define Feature Lists

```python
# 1. Base features (target item + user profile features)
# DIN pairs target features with history features, so the target item
# directly affects the attention output.
features = [
    SparseFeature("user_id", vocab_size=user_num, embed_dim=16),
    SparseFeature("gender", vocab_size=data["gender"].max() + 1, embed_dim=8),
    SparseFeature("age", vocab_size=data["age"].max() + 1, embed_dim=8),
    SparseFeature("occupation", vocab_size=data["occupation"].max() + 1, embed_dim=8),
    SparseFeature("zip", vocab_size=data["zip"].max() + 1, embed_dim=8),
]

target_features = [
    SparseFeature("item_id", vocab_size=item_num, embed_dim=16),
    SparseFeature("cate_id", vocab_size=cate_num, embed_dim=16),
]

# 2. History sequence features
# `shared_with` must point to the corresponding target feature so they
# share the same embedding space.
history_features = [
    SequenceFeature("hist_item_id", vocab_size=item_num, embed_dim=16, pooling="concat", shared_with="item_id"),
    SequenceFeature("hist_cate_id", vocab_size=cate_num, embed_dim=16, pooling="concat", shared_with="cate_id"),
]
```

### 2.3 Build Input Dicts and DataLoaders

```python
# Convert DataFrame objects into dict inputs accepted by the model.
x_train, y_train = df_to_dict(train), train["label"].values
x_val, y_val = df_to_dict(val), val["label"].values
x_test, y_test = df_to_dict(test), test["label"].values

dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=2048,
)
```

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import DIN

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "dice"},
    attention_mlp={"dims": [32, 16], "dropout": 0.2, "activation": "dice"},
)
```

### 3.2 Parameter Details

- `history_features` and `target_features` must be paired one by one.
- `SequenceFeature(..., pooling="concat")` is required here because DIN needs the full sequence, not a pooled vector.
- `attention_mlp` controls the capacity of the activation unit.
- `dice` often works better than plain `relu` in this family of models.

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("./saved/din", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=3,
    earlystop_patience=2,
    device="cpu",  # change to "cuda:0" for GPU
    model_path="./saved/din",
)

# Start training
trainer.fit(train_dl, val_dl)
```

## 5. Evaluation and Result Analysis

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"DIN test AUC: {auc:.4f}")
```

- DIN is usually stronger than plain MLP-based CTR models when sequence information is important.
- If your sequence is too short or too noisy, the gain over simple baselines may be limited.

## 6. Tuning Suggestions

- Tune `attention_mlp` before enlarging the final MLP.
- Try sequence lengths such as `20`, `50`, or `100` depending on your data.
- If training is unstable, reduce batch size or hidden dimensions first.

## 7. FAQ and Troubleshooting

### Q1: Why must `SequenceFeature` use `pooling="concat"`?

Because DIN needs the full behavior sequence so the activation unit can compute attention weights for each historical item.

### Q2: Why do I get a dimension mismatch error?

The most common reason is that `history_features` and `target_features` are not aligned, or `shared_with` does not point to the corresponding target feature.

## 8. Model Visualization

```python
from torch_rechub.utils.visualize import visualize_model

# Automatically generate the graph and save it.
visualize_model(model, save_dir="./visualization", model_name="din")
```

## 9. ONNX Export

```python
# Export DIN with dynamic batch and dynamic sequence lengths.
trainer.export_onnx(
    "./saved/din/din.onnx",
    data_loader=test_dl,
    dynamic_batch=True,
)
```

## Full Example

The code blocks above can be combined directly. For a full sequence-ranking pipeline using the same Amazon Electronics data processing flow, see [examples/ranking/run_amazon_electronics.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/run_amazon_electronics.py).
