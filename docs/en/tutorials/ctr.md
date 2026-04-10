---
title: Ranking Tutorial
description: Torch-RecHub ranking tutorial covering WideDeep, DeepFM, DCN, and sequence ranking models
---

# Ranking Tutorial

This tutorial focuses on the common ranking workflow: data preparation, feature definition, trainer usage, evaluation, and common extensions. The base examples use the built-in `Criteo` sample dataset. The sequence-model section uses the `Amazon Electronics` sample dataset.

## 1. Basic Ranking Pipeline (Criteo)

### 1. Data preparation and feature preprocessing

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

df = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

# Most ranking baselines follow the pattern: dense features + sparse categorical features.
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# Keep preprocessing aligned with the official examples to avoid reproduction drift.
df[sparse_features] = df[sparse_features].fillna("-996")
df[dense_features] = df[dense_features].fillna(0)

# Normalize dense features and encode sparse features as integer ids.
scaler = MinMaxScaler()
df[dense_features] = scaler.fit_transform(df[dense_features])

for feat in sparse_features:
    encoder = LabelEncoder()
    df[feat] = encoder.fit_transform(df[feat].astype(str))

# These Feature objects describe how each column should be consumed by the model.
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=df[name].nunique(), embed_dim=16) for name in sparse_features]

x = df.drop(columns=["label"])
y = df["label"]

# DataGenerator splits train / validation / test sets automatically.
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)
```

### 2. Shared training pattern for WideDeep / DeepFM / DCN

```python
import os
from torch_rechub.models.ranking import WideDeep, DeepFM, DCN
from torch_rechub.trainers import CTRTrainer

# Pick one model.
# DeepFM is usually the easiest first ranking pipeline to start with.
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

# WideDeep separates wide features and deep features more explicitly.
# model = WideDeep(
#     wide_features=sparse_feas,
#     deep_features=sparse_feas + dense_feas,
#     mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
# )

# DCN uses an explicit cross network for feature interactions.
# model = DCN(
#     features=dense_feas + sparse_feas,
#     n_cross_layers=3,
#     mlp_params={"dims": [256, 128], "dropout": 0.2},
# )

os.makedirs("./saved/ranking_base", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path="./saved/ranking_base",
)

trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 3. What changes between WideDeep, DeepFM, and DCN?

- `WideDeep`: split features into wide and deep branches
- `DeepFM`: use FM for low-order interactions plus MLP for higher-order patterns
- `DCN`: use cross layers for explicit feature crossing

If you are choosing a starting point:

- easiest baseline: `DeepFM`
- easier to interpret feature crossing: `DCN`
- classic production baseline: `WideDeep`

## 2. Sequence Ranking Models

Sequence ranking models usually require:

- historical behavior features as `SequenceFeature`
- target item features
- sequence preprocessing that matches the model design

### 1. DIN

```python
from torch_rechub.basic.features import SequenceFeature
from torch_rechub.models.ranking import DIN

# DIN attends from the target item to the user's history sequence.
# Target and history should be paired feature-by-feature.
history_features = [
    SequenceFeature("hist_item_id", vocab_size=item_num, embed_dim=16, pooling="concat", shared_with="item_id")
]
target_features = [
    SparseFeature("item_id", vocab_size=item_num, embed_dim=16)
]

model = DIN(
    features=sparse_feas + dense_feas,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "dice"},
    attention_mlp_params={"dims": [64, 32], "activation": "dice"},
)
```

### 2. DIEN

```python
from torch_rechub.models.ranking import DIEN

# DIEN adds interest evolution modeling on top of the history sequence.
# history_labels are usually needed to indicate positive / negative interest transitions.
model = DIEN(
    features=sparse_feas + dense_feas,
    history_features=history_features,
    target_features=target_features,
    gru_type="AUGRU",
    alpha=1.0,
    use_neg=True,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "dice"},
    attention_mlp_params={"dims": [64, 32], "activation": "dice"},
)
```

### 3. BST

```python
from torch_rechub.models.ranking import BST

# BST feeds history + target features into a Transformer encoder.
# Make sure the embedding dimension is divisible by nhead.
model = BST(
    features=sparse_feas + dense_feas,
    history_features=history_features,
    target_features=target_features,
    transformer_params={
        "nhead": 4,
        "dim_feedforward": 128,
        "dropout": 0.2,
        "activation": "relu",
    },
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### 4. Sequence preprocessing reminder

For sequence ranking models, the data pipeline is often more important than the model line itself:

- keep target features and history features aligned
- use `pooling="concat"` when the full sequence tensor is needed
- ensure padding / truncation strategy matches your model assumptions

## 3. Evaluation and Tuning Suggestions

### 1. Evaluation

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 2. Tuning priorities

- `WideDeep`: tune the MLP dimensions and wide/deep feature split
- `DeepFM`: tune `embed_dim` and MLP depth first
- `DCN`: tune `n_cross_layers`
- `DIN / DIEN`: tune history length, attention MLP, and sequence preprocessing
- `BST`: tune `nhead`, Transformer depth, and embedding dimension

## 4. FAQ

### Q1: Why do the examples use local sample data paths instead of remote URLs?

Using the built-in sample datasets makes the tutorial reproducible without depending on external downloads.

### Q2: Why call `os.makedirs(...)` before training?

The trainers do not create `model_path` automatically, so it is safer to create the save directory explicitly.

### Q3: Why evaluate `trainer.model` instead of the original `model` variable?

Because `trainer.model` ensures you are evaluating the best validated checkpoint after training.

### Q4: Where should I go next?

- [DeepFM tutorial](/tutorials/models/ranking/deepfm)
- [WideDeep tutorial](/tutorials/models/ranking/widedeep)
- [DCN tutorial](/tutorials/models/ranking/dcn)
- [DIN tutorial](/tutorials/models/ranking/din)
- [DIEN tutorial](/tutorials/models/ranking/dien)
- [BST tutorial](/tutorials/models/ranking/bst)
