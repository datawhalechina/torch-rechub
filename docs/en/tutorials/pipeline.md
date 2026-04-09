---
title: Multi-Task Learning Tutorial
description: Torch-RecHub multi-task learning tutorial covering Ali-CCP data preparation, MMOE, PLE, and ESMM
---

# Multi-Task Learning Tutorial

This tutorial uses the built-in `Ali-CCP` sample dataset to introduce the actual multi-task training flow in Torch-RecHub. All code snippets assume you are running from the **repository root**.

## 1. Data Preparation

### 1. Load sample data

```python
import pandas as pd

df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

# Concatenate train / val / test first so feature definitions stay consistent.
train_idx = df_train.shape[0]
val_idx = train_idx + df_val.shape[0]

data = pd.concat([df_train, df_val, df_test], axis=0)
# ctcvr_label is often used in ESMM as the third task: click * conversion
data.rename(columns={"purchase": "cvr_label", "click": "ctr_label"}, inplace=True)
data["ctcvr_label"] = data["cvr_label"] * data["ctr_label"]
```

### 2. Build dense and sparse features

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature

# Ali-CCP is mostly sparse features, with a small number of dense columns.
dense_cols = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]
sparse_cols = [
    col for col in data.columns
    if col not in dense_cols and col not in ["cvr_label", "ctr_label", "ctcvr_label"]
]

# In multi-task learning, all tasks share the same bottom input feature set by default.
features = [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] + [
    DenseFeature(col) for col in dense_cols
]

label_cols = ["cvr_label", "ctr_label"]
used_cols = sparse_cols + dense_cols
```

### 3. Build train / validation / test loaders

```python
from torch_rechub.utils.data import DataGenerator

# In multi-task settings, y becomes a 2D label matrix instead of a single label vector.
x_train = {name: data[name].values[:train_idx] for name in used_cols}
y_train = data[label_cols].values[:train_idx]

x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
y_val = data[label_cols].values[train_idx:val_idx]

x_test = {name: data[name].values[val_idx:] for name in used_cols}
y_test = data[label_cols].values[val_idx:]

dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=1024,
)
```

## 2. MMOE

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer

# MMOE: shared experts + task-specific gates
model = MMOE(
    features=features,
    task_types=["classification", "classification"],
    n_expert=8,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}],
)
```

### Training pattern

```python
import os
import torch

torch.manual_seed(2022)
# MTLTrainer does not create model_path automatically.
os.makedirs("./saved/mmoe", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path="./saved/mmoe",
)

mtl_trainer.fit(train_dl, val_dl)
# evaluate() returns a list whose order matches task_types.
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC: {auc}")  # [cvr_auc, ctr_auc]
```

## 3. PLE

```python
from torch_rechub.models.multi_task import PLE

# PLE is often more stable than MMOE when task differences are larger,
# because it separates shared and task-specific experts.
model = PLE(
    features=features,
    task_types=["classification", "classification"],
    n_level=1,
    n_expert_specific=2,
    n_expert_shared=1,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}],
)
```

### Adaptive loss weighting (optional)

```python
# adaptive_params turns on dynamic loss balancing; this example uses UWL.
os.makedirs("./saved/ple", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    adaptive_params={"method": "uwl"},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",
    model_path="./saved/ple",
)

mtl_trainer.fit(train_dl, val_dl)
```

## 4. ESMM

`ESMM` differs from `MMOE / PLE` in two ways:

- it only uses sparse features
- its label order is usually `["cvr_label", "ctr_label", "ctcvr_label"]`

```python
from torch_rechub.models.multi_task import ESMM

item_cols = ["129", "205", "206", "207", "210", "216"]
user_cols = [col for col in sparse_cols if col not in item_cols]

user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]

label_cols = ["cvr_label", "ctr_label", "ctcvr_label"]
x_train = {name: data[name].values[:train_idx] for name in sparse_cols}
y_train = data[label_cols].values[:train_idx]
```

```python
# ESMM estimates CTR / CVR / CTCVR jointly from user and item feature towers.
model = ESMM(
    user_features,
    item_features,
    cvr_params={"dims": [16, 8]},
    ctr_params={"dims": [16, 8]},
)
```

## 5. Trainer Interface

```python
from torch_rechub.trainers import MTLTrainer

trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3},
    regularization_params={"embedding_l2": 0.0, "dense_l2": 0.0},
    adaptive_params=None,   # Optional: {"method": "uwl"} / {"method": "gradnorm"} / {"method": "metabalance"}
    n_epoch=10,
    earlystop_taskid=0,
    earlystop_patience=10,
    device="cpu",
    model_path="./saved/mtl",
)
```

## 6. Evaluation and Tuning Suggestions

### 1. Evaluation output

```python
scores = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(scores)
```

`evaluate()` returns a list ordered by `task_types`, for example:

- `[cvr_auc, ctr_auc]`
- or three task scores in the ESMM case

### 2. What to tune first

- `MMOE`: start with `n_expert`
- `PLE`: start with `n_level / n_expert_specific / n_expert_shared`
- if task imbalance is obvious: try `adaptive_params={"method": "uwl"}`
- if multi-task AUC is unstable: reduce the learning rate first, then shrink expert/tower dimensions

## 7. FAQ

### Q1: Why not use `from torch_rechub.utils import DataGenerator` here?

Because `DataGenerator` lives in `torch_rechub.utils.data`, not in the top-level `torch_rechub.utils` namespace.

### Q2: Why use `n_epoch` instead of `n_epochs`?

The actual parameter name in `MTLTrainer` is `n_epoch`.

### Q3: Why is there no `evaluate_multi_task()` helper?

The framework directly uses `MTLTrainer.evaluate(model, data_loader)`, which returns a list of task scores.

### Q4: Why call `os.makedirs(...)` before training?

`MTLTrainer` does not create `model_path` automatically, so the examples create the save directory explicitly.

### Q5: Where should I go next?

- [MMOE tutorial](/tutorials/models/multi_task/mmoe)
- [PLE tutorial](/tutorials/models/multi_task/ple)
