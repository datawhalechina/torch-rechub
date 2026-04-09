---
title: DCN / DCNv2 Tutorial
description: "Complete Deep & Cross Network tutorial: from data preparation to training and evaluation"
---

# DCN / DCNv2 Tutorial

## 1. Model Overview and Use Cases

**DCN** (Deep & Cross Network) was proposed by Google at ADKDD 2017. It uses a dedicated cross network to **explicitly learn high-order feature interactions** while keeping linear-time complexity. **DCNv2**, published at WWW 2021, is an enhanced version with stronger representation power.

**Papers**:
- DCN: [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
- DCNv2: [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)

### Key Differences

| Aspect | DCN | DCNv2 |
|------|-----|-------|
| Cross operation | Vector-wise cross | Matrix-wise cross |
| Representation power | Moderate | Stronger |
| Parameter count | Smaller | Larger |

### Suitable Scenarios

- CTR prediction tasks that need explicit feature crossing
- Scenarios with many sparse features
- Online systems that care about both accuracy and inference efficiency

---

## 2. Data Preparation and Preprocessing

This example uses the sampled **Criteo** dataset. The preprocessing pipeline is the same as in the [ranking tutorial](/tutorials/ctr).

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

dense_features = [f for f in data.columns if f.startswith("I")]
sparse_features = [f for f in data.columns if f.startswith("C")]

# Fill missing values before normalization / label encoding.
data[sparse_features] = data[sparse_features].fillna("-1")
data[dense_features] = data[dense_features].fillna(0)

# Normalize dense features to [0, 1].
data[dense_features] = MinMaxScaler().fit_transform(data[dense_features])

# Encode categorical features as consecutive integer ids.
for f in sparse_features:
    data[f] = LabelEncoder().fit_transform(data[f])

features = [DenseFeature(name) for name in dense_features] + [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

target = data["label"].values.tolist()
train, val, test = data[:800], data[800:900], data[900:]

dg = DataGenerator(train, val, test)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x=features,
    y=target,
    batch_size=256,
)
```

## 3. Model Configuration and Parameter Notes

### 3.1 DCN

```python
from torch_rechub.models.ranking import DCN

model = DCN(
    features=features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### 3.2 DCNv2

```python
from torch_rechub.models.ranking import DCNv2

model = DCNv2(
    features=features,
    cross_num=3,
    dnn_hidden_units=(256, 128),
    low_rank=32,
    num_experts=4,
)
```

### 3.3 Parameter Details

- `n_cross_layers` / `cross_num`: controls how many cross layers are stacked. More layers can model more complex interactions, but may overfit on small data.
- `mlp_params` / `dnn_hidden_units`: controls the deep branch capacity.
- `low_rank` and `num_experts` in DCNv2: improve expressiveness while keeping parameter growth manageable.

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("./saved/dcn", exist_ok=True)

# Use either DCN or DCNv2 here.
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",  # change to "cuda:0" if GPU is available
    model_path="./saved/dcn",
)

trainer.fit(train_dl, val_dl)
```

## 5. Evaluation and Result Analysis

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### DCN vs DCNv2 Performance Comparison

- If your feature interactions are relatively simple, DCN is usually a strong and efficient baseline.
- If your data is larger and interaction patterns are more complex, DCNv2 often performs better.

## 6. Tuning Suggestions

- Tune `n_cross_layers` / `cross_num` first. A common range is `2 ~ 4`.
- Use a slightly larger deep branch when the dataset is large enough.
- If DCNv2 is unstable on a small dataset, reduce `low_rank`, `num_experts`, or hidden sizes first.

## 7. FAQ and Troubleshooting

### Q1: What is the main difference between DCN and DCNv2?

DCN uses vector-wise feature crossing, while DCNv2 uses a richer matrix-based formulation and usually has stronger modeling power.

### Q2: How many cross layers should I use?

Start from `2` or `3`. More layers are not always better, especially on small sampled datasets.

### Q3: Can the Cross Network replace manual feature engineering?

It can learn many useful interaction patterns automatically, but business-specific crosses can still help in some production scenarios.

## Full Example

The code blocks above form a complete runnable example. For a full end-to-end training script using the same Criteo pipeline, see [examples/ranking/run_criteo.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/run_criteo.py).
