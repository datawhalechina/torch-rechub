---
title: PLE Tutorial
description: Predict clicks and purchases with shared and task-specific experts in PLE
---

# PLE Tutorial

## 1. Model Overview and Use Cases

Clicks and purchases may depend on different features. PLE lets you reserve some experts for each task. In this example, each task has two experts of its own and can use one shared expert. Each gate combines the outputs of the experts available to its task before the prediction layer.

**Paper**: [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)

### Model Architecture

<div align="center">
  <img src="/img/models/ple_arch.png" alt="PLE Model Architecture" width="600"/>
</div>

One layer in the diagram contains task-specific experts, shared experts and gates that select the experts. `n_level` controls how many of these layers to stack. Start with one layer and keep the same label order as the MMOE tutorial to compare the two outputs directly.

---

## 2. Data Preparation and Preprocessing

This tutorial uses the **Ali-CCP** dataset for click and conversion prediction. Its data-preparation flow is the same as for MMOE.

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# Load the preprocessed Ali-CCP sample data
df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")
print(f"Train: {len(df_train)}, validation: {len(df_val)}, test: {len(df_test)}")

# Merge the data for consistent feature processing
train_idx = df_train.shape[0]
val_idx = train_idx + df_val.shape[0]
data = pd.concat([df_train, df_val, df_test], axis=0)

# Rename the label columns
data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
```

### 2.2 Define Features and Labels

```python
col_names = data.columns.tolist()

# Separate continuous and categorical features
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

# Define multi-task labels (CVR, CTR)
label_cols = ['cvr_label', 'ctr_label']
used_cols = sparse_cols + dense_cols
```

### 2.3 Build the Training, Validation, and Test Sets

```python
x_train = {name: data[name].values[:train_idx] for name in used_cols}
y_train = data[label_cols].values[:train_idx]

x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
y_val = data[label_cols].values[train_idx:val_idx]

x_test = {name: data[name].values[val_idx:] for name in used_cols}
y_test = data[label_cols].values[val_idx:]

# Create DataLoaders
dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=y_test,
    batch_size=2048
)
```

---

## 3. Model Configuration and Parameter Reference

### 3.1 Create the Model

```python
from torch_rechub.models.multi_task import PLE

model = PLE(
    features=features,
    task_types=["classification", "classification"],  # Two classification tasks
    n_level=1,                    # Number of CGC levels
    n_expert_specific=2,          # Experts dedicated to each task
    n_expert_shared=1,            # Shared experts; this is the key difference from MMOE
    expert_params={
        "dims": [16]
    },
    tower_params_list=[
        {"dims": [8]},            # CVR Tower
        {"dims": [8]}             # CTR Tower
    ]
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Recommended Value |
|------|------|------|--------|
| `features` | `list[Feature]` | Feature list | Dense + Sparse |
| `task_types` | `list[str]` | Task-type list | `"classification"` or `"regression"` |
| `n_level` | `int` | Number of CGC (progressive) levels | 1–3 |
| `n_expert_specific` | `int` | Number of task-specific experts per task | 1–4 |
| `n_expert_shared` | `int` | Number of shared experts | 1–2 |
| `expert_params` | `dict` | Expert MLP parameters | `{"dims": [16]}` |
| `tower_params_list` | `list[dict]` | MLP parameters for each Task Tower | `{"dims": [8]}` |

`n_expert_specific=2` gives each task two experts of its own. With two tasks, the layer has four task-specific experts plus one shared expert.

---

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import MTLTrainer

torch.manual_seed(2022)
os.makedirs("./saved/ple", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    adaptive_params={"method": "uwl"},  # Uncertainty Weighting Loss
    n_epoch=20,
    earlystop_patience=5,
    device="cpu",
    model_path="./saved/ple"
)

mtl_trainer.fit(train_dl, val_dl)
```

### Multi-Task Loss-Balancing Methods

When `adaptive_params` is omitted, the trainer weights the task losses equally. The `uwl` setting above learns the task weights during training. Check the purchase and click AUCs separately with either setting.

---

## 5. Model Evaluation and Result Analysis

```python
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC (CVR): {auc[0]:.4f}, Test AUC (CTR): {auc[1]:.4f}")
```

---

## 6. Tuning Recommendations

To compare PLE with MMOE, use the same training split, task order and loss weighting. With a single PLE layer, vary the number of experts per task and record both AUCs to see which task benefits.

## 7. FAQ and Troubleshooting

### Q1: How do PLE and MMOE differ?

In MMOE, both tasks can use every expert. PLE gives each task its own experts and adds a pool of shared experts. Both models can predict clicks and purchases for the same batch of samples.

### Q2: How can I combine classification and regression tasks?
Set `task_types=["classification", "regression"]`. The model uses the appropriate prediction layer for each task. To use the second task's MSE for early stopping, set `earlystop_taskid=1, earlystop_mode="min"` in `MTLTrainer`.

### Q3: How large should `n_level` be?
The example uses one layer. A second layer feeds the first layer's outputs to another set of experts and gates. Compare both tasks on the same validation samples when choosing between them.

---

## 8. Model Visualization

```python
from torch_rechub.utils.visualization import visualize_model
visualize_model(model, save_path="ple_architecture.png", dpi=300)
```

---

## 9. ONNX Export

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("ple.onnx", verbose=True)
```

---

## Complete Example

```python
import os
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import PLE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/ple", exist_ok=True)

    # 1. Load the data
    df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
    df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
    df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

    train_idx = df_train.shape[0]
    val_idx = train_idx + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)

    # 2. Define features
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [
        col for col in data.columns
        if col not in dense_cols and col not in ['cvr_label', 'ctr_label']
    ]

    features = [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] \
        + [DenseFeature(col) for col in dense_cols]

    label_cols = ['cvr_label', 'ctr_label']
    used_cols = sparse_cols + dense_cols

    # 3. Build datasets
    x_train = {name: data[name].values[:train_idx] for name in used_cols}
    y_train = data[label_cols].values[:train_idx]
    x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
    y_val = data[label_cols].values[train_idx:val_idx]
    x_test = {name: data[name].values[val_idx:] for name in used_cols}
    y_test = data[label_cols].values[val_idx:]

    dg = DataGenerator(x_train, y_train)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=2048
    )

    # 4. Create the PLE model
    model = PLE(
        features=features,
        task_types=["classification", "classification"],
        n_level=1, n_expert_specific=2, n_expert_shared=1,
        expert_params={"dims": [16]},
        tower_params_list=[{"dims": [8]}, {"dims": [8]}]
    )

    # 5. Train
    mtl_trainer = MTLTrainer(
        model, task_types=["classification", "classification"],
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        adaptive_params={"method": "uwl"},
        n_epoch=20, earlystop_patience=5, device="cpu", model_path="./saved/ple"
    )
    mtl_trainer.fit(train_dl, val_dl)

    # 6. Evaluate
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
    print(f"Test AUC (CVR): {auc[0]:.4f}, Test AUC (CTR): {auc[1]:.4f}")


if __name__ == "__main__":
    main()
```
