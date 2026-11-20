---
title: Multi-Task Learning Tutorial
description: Torch-RecHub multi-task learning tutorial covering Ali-CCP data preparation, MMOE, PLE, and ESMM
---

# Multi-Task Learning Tutorial

Each Ali-CCP impression records two outcomes: whether the user clicked and whether they purchased. We'll train MMOE and PLE on the same samples and compare their AUCs for each task. Then we'll train ESMM to see how its labels capture the relationship between clicks and purchases. Run the examples from the repository root.

## 1. Data Preparation

### 1. Load sample data

```python
import pandas as pd

df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

# These samples share a category encoding and already have train, validation and test splits.
# For raw business data, fit the vocabulary on the training split.
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

def make_features():
    return [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] + [
        DenseFeature(col) for col in dense_cols
    ]
```

### 3. Prepare features and labels for the model

`label_cols` determines the output order. We put purchases first and clicks second, so column 0 of `y_train` holds purchase labels and column 1 holds click labels. Read predictions and AUCs in the same order.

```python
from torch_rechub.utils.data import DataGenerator

def make_loaders(feature_cols, label_cols):
    def split(part):
        return {name: part[name].to_numpy() for name in feature_cols}, part[label_cols].to_numpy()

    x_train, y_train = split(data.iloc[:train_idx])
    x_val, y_val = split(data.iloc[train_idx:val_idx])
    x_test, y_test = split(data.iloc[val_idx:])
    return DataGenerator(x_train, y_train).generate_dataloader(
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=256,
    )

label_cols = ["cvr_label", "ctr_label"]
train_dl, val_dl, test_dl = make_loaders(sparse_cols + dense_cols, label_cols)
```

## 2. MMOE

The two tasks share a pool of expert networks. Each task has a gate that learns how much weight to give each expert's output when predicting a click or a purchase.

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer

# MMOE: shared experts + task-specific gates
model = MMOE(
    features=make_features(),
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

PLE has separate experts for clicks and purchases as well as shared experts. Reuse the DataLoaders above and call `make_features()` again so PLE starts with its own embedding weights.

```python
from torch_rechub.models.multi_task import PLE

# Give PLE fresh feature objects. Its experts include both shared and task-specific networks.
model = PLE(
    features=make_features(),
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

This example splits the categorical fields into user features and item features. In addition to the purchase and click labels, compute `ctcvr_label`, which is 1 when the impression led to both a click and a purchase. Use the following column order for training, validation and testing.

```python
from torch_rechub.models.multi_task import ESMM

item_cols = ["129", "205", "206", "207", "210", "216"]
user_cols = [col for col in sparse_cols if col not in item_cols]

user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]

label_cols = ["cvr_label", "ctr_label", "ctcvr_label"]
esmm_train, esmm_val, esmm_test = make_loaders(sparse_cols, label_cols)

model = ESMM(
    user_features, item_features,
    cvr_params={"dims": [16, 8]}, ctr_params={"dims": [16, 8]},
)
os.makedirs("./saved/esmm", exist_ok=True)
esmm_trainer = MTLTrainer(
    model, task_types=["classification"] * 3, n_epoch=2,
    device="cpu", model_path="./saved/esmm",
)
esmm_trainer.fit(esmm_train, esmm_val)
scores = esmm_trainer.evaluate(esmm_trainer.model, esmm_test)
print(dict(zip(label_cols, scores)))
```

For example, a click without a purchase gives `[0, 1, 0]`. The model outputs CVR, CTR and CTCVR in that order. The loss uses CTR and CTCVR, and evaluation returns three scores in the same order as the columns.

## 5. How do I read the evaluation results?

For MMOE and PLE, `evaluate()` returns purchase AUC first and click AUC second. The bundled validation and test sets each contain 50 rows and just one purchase. Moving that one row up or down the ranking can change purchase AUC substantially. Check the positive counts for each task before comparing models on your own data.

To change the number or depth of the experts, see the individual model tutorials.

- [MMOE tutorial](/tutorials/models/multi_task/mmoe)
- [PLE tutorial](/tutorials/models/multi_task/ple)
