---
title: MMOE Tutorial
description: "Complete Multi-gate Mixture-of-Experts tutorial"
---

# MMOE Tutorial

## 1. Model Overview and Use Cases

MMOE (Multi-gate Mixture-of-Experts), proposed by Google at KDD 2018, is a classic multi-task learning model. It uses **multiple expert networks** and **task-specific gates** so different tasks can combine experts differently, which helps reduce **negative transfer** between tasks.

**Paper**: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)

### Model Architecture

- **Multiple expert networks**: shared bottom experts that learn different representations
- **Multiple gate networks**: one gate per task to combine expert outputs
- **Multiple task towers**: one prediction head per task

### Suitable Scenarios

- Multi-objective optimization such as CTR + CVR
- Tasks that are related but may also conflict
- E-commerce recommendation with click / add-to-cart / purchase style targets

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

# Merge them once so feature processing stays consistent
df = pd.concat([df_train, df_val, df_test], axis=0)

# Rename labels
label_cols = ["conversion", "click"]
```

### 2.2 Define Features and Labels

```python
# Separate dense and sparse features
dense_cols = [c for c in df.columns if c.startswith("I")]
sparse_cols = [c for c in df.columns if c.startswith("C")]

# Define features
features = [DenseFeature(name) for name in dense_cols] + [
    SparseFeature(name, vocab_size=df[name].max() + 1, embed_dim=16)
    for name in sparse_cols
]
```

### 2.3 Build Train / Validation / Test Sets

```python
x_train = df_train[dense_cols + sparse_cols].to_dict("list")
y_train = df_train[label_cols].values
x_val = df_val[dense_cols + sparse_cols].to_dict("list")
y_val = df_val[label_cols].values
x_test = df_test[dense_cols + sparse_cols].to_dict("list")
y_test = df_test[label_cols].values

dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=1024,
)
```

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.multi_task import MMOE

model = MMOE(
    features=features,
    task_types=["classification", "classification"],
    n_expert=8,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}],
)
```

### 3.2 Parameter Details

- `n_expert`: number of shared experts
- `expert_params`: hidden dimensions of expert networks
- `tower_params_list`: task-specific prediction towers

## 4. Training Process and Code Example

### 4.1 Train the Model

```python
import os
from torch_rechub.trainers import MTLTrainer

os.makedirs("./saved/mmoe", exist_ok=True)

trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",
    model_path="./saved/mmoe",
)

trainer.fit(train_dl, val_dl)
```

### 4.2 Adaptive Loss Weighting (Optional)

You can further try adaptive multi-task loss weighting strategies when one task dominates the optimization.

## 5. Evaluation and Result Analysis

### 5.1 Multi-task Evaluation

```python
# Output format: [cvr_auc, ctr_auc]
scores = trainer.evaluate(trainer.model, test_dl)
print(scores)
```

### 5.2 Expected Performance

- MMOE is a strong multi-task baseline when tasks share useful bottom-level information
- If tasks are highly divergent, the gains may be smaller

## 6. Tuning Suggestions

### 6.1 Key Tuning Points

- Tune `n_expert` first
- Increase tower capacity only after the shared experts are reasonable
- Watch task imbalance during training

### 6.2 MMOE vs Other Multi-task Models

- MMOE is usually a good default baseline
- PLE often works better when tasks need stronger separation

## 7. FAQ and Troubleshooting

### Q1: One task keeps getting much lower AUC. What should I do?

Try adaptive loss balancing, task-specific tower tuning, and a smaller shared expert capacity.

### Q2: Does `task_types` support regression tasks?

Yes. You can mix task types as long as the trainer and labels are configured consistently.

### Q3: How do I add more tasks?

Add more entries to `task_types` and `tower_params_list`, and make sure your labels are aligned with the task order.

### Q4: Why does training get slower when `n_expert` increases?

More experts mean more computation and more parameters. Start from a small number and scale only when needed.

## Full Example

The code blocks above form a complete runnable example. Use them together with the Ali-CCP preprocessing flow in [examples/multi_task/run_ali_ccp.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/multi_task/run_ali_ccp.py).
