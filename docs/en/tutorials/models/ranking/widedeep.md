---
title: Wide&Deep Tutorial
description: "Complete Wide&Deep usage tutorial: from data preparation to training and evaluation"
---

# Wide&Deep Tutorial

## 1. Model Overview and Use Cases

Wide&Deep is the classic recommendation model proposed by Google at DLRS 2016. It combines the memorization ability of a linear model (the Wide branch) and the generalization ability of a deep neural network (the Deep branch), and is one of the most widely used industrial baselines.

**Paper**: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

### Model Architecture

- **Wide branch**: a linear model that captures feature co-occurrence patterns (memorization)
- **Deep branch**: a multi-layer MLP that learns generalized feature representations through embeddings
- **Joint training**: the outputs of the Wide and Deep branches are added together and passed through a Sigmoid for the final prediction

### Suitable Scenarios

- recommendation system baseline
- scenarios that need both memorization and generalization
- fast validation of whether the available features are useful

---

## 2. Data Preparation and Preprocessing

This example uses the Criteo ad click dataset. The data preparation pipeline is the same as the one used in DeepFM.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# Load data
data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

# Split feature types
dense_features = [f for f in data.columns if f.startswith("I")]
sparse_features = [f for f in data.columns if f.startswith("C")]

# Missing value handling
data[sparse_features] = data[sparse_features].fillna("0")
data[dense_features] = data[dense_features].fillna(0)

# Normalize dense features
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# Encode sparse features
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

# Define features
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

y = data["label"]
del data["label"]
x = data

# Create DataLoaders
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1], batch_size=2048
)
```

---

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import WideDeep

model = WideDeep(
    wide_features=dense_feas,      # the Wide branch is more about "memorization", so dense or manual cross features fit well here
    deep_features=sparse_feas,     # the Deep branch is more about "generalization", so sparse categorical combinations fit well here
    mlp_params={
        "dims": [256, 128],        # hidden dimensions of the MLP
        "dropout": 0.2,
        "activation": "relu"
    }
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Suggested Value |
|------|------|------|--------|
| `wide_features` | `list[Feature]` | feature list for the Wide branch | dense features / cross features |
| `deep_features` | `list[Feature]` | feature list for the Deep branch | sparse features |
| `mlp_params.dims` | `list[int]` | hidden dimensions of each MLP layer | `[256, 128]` |
| `mlp_params.dropout` | `float` | dropout ratio | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | activation function | `"relu"` |

> **How to split features between Wide and Deep**
> - the **Wide branch** usually fits low-dimensional dense features or manually designed cross features
> - the **Deep branch** usually fits high-dimensional sparse categorical features through embeddings

---

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/widedeep", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=50,
    earlystop_patience=10,
    device="cpu",                   # GPU: "cuda:0"
    model_path="./saved/widedeep"
)

trainer.fit(train_dl, val_dl)
```

---

## 5. Evaluation and Result Analysis

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### Expected Performance

| Data Scale | Expected AUC |
|----------|----------|
| Sample (10k rows) | 0.68 ~ 0.73 |
| Full (45M rows) | 0.78 ~ 0.80 |

As a baseline model, Wide&Deep usually delivers slightly lower AUC than DeepFM, but it often trains faster.

---

## 6. Tuning Suggestions

1. **The split between Wide and Deep features** is the most important choice:
   - a common rule of thumb is: dense features -> Wide, categorical features -> Deep
   - you can also let both branches share all features

2. **MLP structure**: the Deep branch of Wide&Deep usually does not need to be too deep; `[256, 128]` is often enough

3. **Learning rate**: start searching from `1e-3`

---

## 7. FAQ and Troubleshooting

### Q1: Should the Wide branch and the Deep branch use the same features?
It is usually better to split them: the Wide branch focuses on memorization and works well with low-dimensional features, while the Deep branch focuses on generalization and works well with high-dimensional categorical features. That said, using the same features in both branches is also valid.

### Q2: Which is better, Wide&Deep or DeepFM?
In general, DeepFM tends to perform better because the FM branch automatically learns second-order feature interactions, while the Wide branch of Wide&Deep only performs a linear transform. However, Wide&Deep is simpler and often trains faster.

### Q3: How do I add cross features into the Wide branch?
You can manually build cross features during preprocessing, such as `city_x_gender`, and add them as new `SparseFeature` columns in `wide_features`.

---

## Full Example

```python
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import WideDeep
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/widedeep", exist_ok=True)

    data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")
    dense_features = [f for f in data.columns if f.startswith("I")]
    sparse_features = [f for f in data.columns if f.startswith("C")]

    data[sparse_features] = data[sparse_features].fillna("0")
    data[dense_features] = data[dense_features].fillna(0)

    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [
        SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
        for name in sparse_features
    ]

    y = data["label"]
    del data["label"]
    x = data

    dg = DataGenerator(x, y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        split_ratio=[0.7, 0.1], batch_size=2048
    )

    model = WideDeep(
        wide_features=dense_feas,
        deep_features=sparse_feas,
        mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
    )

    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=50,
        earlystop_patience=10,
        device="cpu",
        model_path="./saved/widedeep"
    )
    trainer.fit(train_dl, val_dl)

    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
```
