---
title: PLE Tutorial
description: "Complete Progressive Layered Extraction tutorial"
---

# PLE Tutorial

## 1. Model Overview and Use Cases

PLE (Progressive Layered Extraction), proposed by Tencent at RecSys 2020, is a multi-task learning model designed to address the **seesaw phenomenon**, where improving one task hurts another. It uses **Customized Gate Control (CGC)** with task-specific experts and shared experts, and stacks them progressively.

**Paper**: [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning Model for Personalized Recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)

### Model Architecture

<div align="center">
  <img src="/img/models/ple_arch.png" alt="PLE Model Architecture" width="600"/>
</div>

- **Task-Specific Experts**: one set of experts per task
- **Shared Experts**: experts shared across all tasks
- **Customized Gate (CGC)**: task-specific gates that combine task-specific and shared experts
- **Multi-Level**: supports stacking multiple CGC levels
- **Task Towers**: one prediction tower per task

### Suitable Scenarios

- Multi-objective optimization such as CTR + CVR
- Tasks that are related but still need stronger separation than MMOE provides
- Industrial recommendation systems with multiple business goals

---

## 2. Data Preparation and Preprocessing

PLE uses the same **Ali-CCP** preparation flow as MMOE.

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# Load processed sampled data
df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

# Merge data for consistent feature handling
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
from torch_rechub.models.multi_task import PLE

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

### 3.2 Parameter Details

- `n_level`: number of stacked CGC layers
- `n_expert_specific`: experts owned by each task
- `n_expert_shared`: experts shared across tasks
- `tower_params_list`: task-specific prediction heads

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import MTLTrainer

os.makedirs("./saved/ple", exist_ok=True)

trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    adaptive_params={"method": "uwl"},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",
    model_path="./saved/ple",
)

trainer.fit(train_dl, val_dl)
```

### Multi-task Loss Balancing

PLE is often paired with adaptive task-weighting strategies, especially when tasks have very different difficulty or label sparsity.

## 5. Evaluation and Result Analysis

```python
scores = trainer.evaluate(trainer.model, test_dl)
print(scores)
```

- PLE is usually a stronger choice than MMOE when task separation matters more
- It also has more parameters and can be harder to tune

## 6. Tuning Suggestions

- Tune `n_level`, `n_expert_specific`, and `n_expert_shared` together
- If tasks are very related, a simpler MMOE may already be enough
- If one task is much weaker, consider increasing task-specific experts before increasing shared experts

## 7. FAQ and Troubleshooting

### Q1: How should I choose between PLE and MMOE?

Use MMOE as a simpler baseline. Use PLE when tasks need clearer expert separation or when MMOE suffers from stronger negative transfer.

### Q2: How do I handle mixed classification + regression tasks?

Set `task_types` accordingly and make sure the labels, losses, and evaluation metrics match each task type.

### Q3: How large should `n_level` be?

Start from `1`. Increasing it can help, but it also makes the model heavier and harder to tune.

## 8. Model Visualization

PLE can still be visualized at a high level, although the multi-level CGC structure is more complex than plain feed-forward MTL baselines.

## 9. ONNX Export

```python
trainer.export_onnx("./saved/ple/ple.onnx", data_loader=test_dl, dynamic_batch=True)
```

## Full Example

The code blocks above form a complete runnable example. Use them together with the Ali-CCP preprocessing flow in [examples/multi_task/run_ali_ccp.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/multi_task/run_ali_ccp.py).
