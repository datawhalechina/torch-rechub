---
title: MMOE Tutorial
description: Train MMOE on purchase and click labels from Ali-CCP and inspect AUC for each task
---

# MMOE Tutorial

## 1. Model Overview and Use Cases

In Ali-CCP, purchase and click labels belong to the same impression and share the same input fields. MMOE gives both tasks access to several small networks called experts. Each task learns its own weights for combining the experts, so a given expert can contribute differently to purchase and click predictions.

**Paper**: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)

### Model Architecture

The experts process the input, and each task's gate combines their outputs. A separate prediction layer for each task produces its probability. Keep these outputs in the same order as the label columns.

---

## 2. Data Preparation and Preprocessing

This example uses the sampled **Ali-CCP** dataset and jointly predicts **CTR** and **CVR**.

### 2.1 Load and Process Data

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# Load processed Ali-CCP sampled data
df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")
print(f"Training rows: {len(df_train)}, validation rows: {len(df_val)}, test rows: {len(df_test)}")

# Merge them once so feature processing stays consistent
train_idx = df_train.shape[0]
val_idx = train_idx + df_val.shape[0]
data = pd.concat([df_train, df_val, df_test], axis=0)

# Rename labels
data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
```

### 2.2 Define Features and Labels

```python
col_names = data.columns.tolist()

# Separate dense and sparse features
dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
sparse_cols = [
    col for col in col_names
    if col not in dense_cols and col not in ['cvr_label', 'ctr_label']
]

# Define features
features = [
    SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols
] + [
    DenseFeature(col) for col in dense_cols
]

# Task labels in output order: CVR, CTR.
label_cols = ['cvr_label', 'ctr_label']
used_cols = sparse_cols + dense_cols
```

### 2.3 Build Train / Validation / Test Sets

```python
x_train = {name: data[name].values[:train_idx] for name in used_cols}
y_train = data[label_cols].values[:train_idx]

x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
y_val = data[label_cols].values[train_idx:val_idx]

x_test = {name: data[name].values[val_idx:] for name in used_cols}
y_test = data[label_cols].values[val_idx:]

# Build the DataLoaders.
dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=y_test,
    batch_size=1024
)
```

---

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.multi_task import MMOE

model = MMOE(
    features=features,
    task_types=["classification", "classification"],   # Two classification tasks.
    n_expert=8,                                         # One pool of experts shared by both tasks.
    expert_params={"dims": [16]},                       # Expert network widths.
    tower_params_list=[{"dims": [8]}, {"dims": [8]}]    # One output tower per task.
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Example setting |
|------|------|------|--------|
| `features` | `list[Feature]` | All input features | Dense and sparse features |
| `task_types` | `list[str]` | Task type for each output | `"classification"` or `"regression"` |
| `n_expert` | `int` | Number of expert networks | 4–16 |
| `expert_params.dims` | `list[int]` | Hidden widths in each expert | `[16]` or `[32, 16]` |
| `tower_params_list` | `list[dict]` | Tower configuration for each task | `{"dims": [8]}` per tower |

With `n_expert=8`, both tasks have access to all eight experts. There are still only two entries in `tower_params_list`, one for each task's prediction network.

---

## 4. Training Process and Code Example

### 4.1 Train the Model

```python
import os
from torch_rechub.trainers import MTLTrainer

torch.manual_seed(2022)
os.makedirs("./saved/mmoe", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={
        "lr": 1e-3,
        "weight_decay": 1e-4
    },
    n_epoch=50,
    earlystop_patience=30,
    device="cpu",
    model_path="./saved/mmoe"
)

mtl_trainer.fit(train_dl, val_dl)
```

### 4.2 Adaptive loss weights (optional)

Use adaptive weights when the losses for different tasks have very different scales.

```python
import os

os.makedirs("./saved/mmoe", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    adaptive_params={"method": "uwl"},   # Uncertainty Weight Loss
    n_epoch=50,
    earlystop_patience=30,
    device="cpu",
    model_path="./saved/mmoe"
)
```

---

## 5. Evaluation and Result Analysis

### 5.1 Multi-task Evaluation

```python
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC: {auc}")
# Output format: [cvr_auc, ctr_auc]
```

`evaluate()` returns a list with one AUC per task, in `task_types` order.

The validation and test sets each contain 50 rows and only one purchase. Use the sample to check that `evaluate()` returns two AUCs in the expected order. For a model comparison, fix the data splits and check each task's positive and negative counts.

## 6. Tuning Suggestions

### 6.1 Key Tuning Points

Record separate validation AUCs for purchases and clicks. Then change only `n_expert` and run again on the same split. Each additional expert processes every sample, so record the training time as well.

### 6.2 MMOE vs Other Multi-task Models

| Model | How parameters are shared | When to try it |
|------|------------|---------|
| SharedBottom | All tasks share the bottom layers | Closely related tasks |
| MMOE | Gates weight a shared pool of experts | Tasks with competing objectives |
| PLE | Successive layers of shared and task-specific experts | More complex multi-task problems |

---

## 7. FAQ and Troubleshooting

### Q1: One task keeps getting much lower AUC. What should I do?
Check the number of positives first. This validation set contains one purchase, so a lower purchase AUC than click AUC alone tells you little about interference between tasks. Compare the same task's AUC when trained alone and jointly, using the same split.

### Q2: Does `task_types` support regression tasks?
Yes. Set `task_types=["classification", "regression"]`. The trainer uses binary cross-entropy for classification and mean squared error for regression.

### Q3: How do I add more tasks?
Add more entries to `task_types` and `tower_params_list`, and make sure your labels are aligned with the task order.

```python
model = MMOE(
    features=features,
    task_types=["classification", "classification", "regression"],  # Three tasks.
    n_expert=8,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
)
```

### Q4: Why does training get slower when `n_expert` increases?
Compute grows linearly with the number of experts. Try narrower expert networks when comparing accuracy and training cost.

---

## Full Example

```python
import os
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/mmoe", exist_ok=True)

    # 1. Load the data.
    df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
    df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
    df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

    train_idx = df_train.shape[0]
    val_idx = train_idx + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)

    # 2. Define the features.
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [
        col for col in data.columns
        if col not in dense_cols and col not in ['cvr_label', 'ctr_label']
    ]

    features = [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] \
        + [DenseFeature(col) for col in dense_cols]

    label_cols = ['cvr_label', 'ctr_label']
    used_cols = sparse_cols + dense_cols

    # 3. Build the datasets.
    x_train = {name: data[name].values[:train_idx] for name in used_cols}
    y_train = data[label_cols].values[:train_idx]
    x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
    y_val = data[label_cols].values[train_idx:val_idx]
    x_test = {name: data[name].values[val_idx:] for name in used_cols}
    y_test = data[label_cols].values[val_idx:]

    dg = DataGenerator(x_train, y_train)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=1024
    )

    # 4. Create the model.
    model = MMOE(
        features=features,
        task_types=["classification", "classification"],
        n_expert=8,
        expert_params={"dims": [16]},
        tower_params_list=[{"dims": [8]}, {"dims": [8]}]
    )

    # 5. Train.
    mtl_trainer = MTLTrainer(
        model,
        task_types=["classification", "classification"],
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
        n_epoch=50,
        earlystop_patience=30,
        device="cpu",
        model_path="./saved/mmoe"
    )
    mtl_trainer.fit(train_dl, val_dl)

    # 6. Evaluate.
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
    print(f"Test AUC (CVR, CTR): {auc}")


if __name__ == "__main__":
    main()
```
