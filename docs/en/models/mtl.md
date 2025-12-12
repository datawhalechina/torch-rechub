---
title: Multi-Task Models
description: Torch-RecHub multi-task learning models
---

# Multi-Task Models

Multi-task learning jointly optimizes related objectives (e.g., CTR, CVR, retention) to improve generalization.

## Model Overview (what to pick)

| Model        | When to use | Highlights |
| ---          | ---         | --- |
| SharedBottom | Tasks highly related | Shared bottom + task-specific towers |
| MMOE         | Task conflict exists | Multi-gate mixture-of-experts per task |
| PLE          | Complex multi-task    | Progressive layered extraction to reduce negative transfer |
| ESMM         | Sample selection bias | Full-space modeling for CVR/CTR with post-click modeling |
| AITM         | Task dependency       | Adaptive information transfer between tasks |

## Quick Usage (example: SharedBottom)

```python
from torch_rechub.models.multi_task import SharedBottom
from torch_rechub.basic.features import SparseFeature, DenseFeature

common_features = [
    SparseFeature("user_id", vocab_size=10000, embed_dim=32),
    DenseFeature("age", embed_dim=1),
]

model = SharedBottom(
    features=common_features,
    task_types=["classification", "classification"],
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},
    ],
)
```

## Parameter Notes (common fields)

- `features`: shared feature list.
- `task_types`: list of task types (`classification`, `regression`).
- `bottom_params`: shared bottom MLP config.
- `tower_params_list`: per-task tower MLP configs.
- (MMOE/PLE) expert/tower configs follow the same dict style (`dims`, `dropout`, `activation`).

## Tips

- Strongly related tasks, simple setup → SharedBottom.  
- Task competition / conflicts → MMOE.  
- Need finer separation and reduced negative transfer → PLE.  
- CVR/CTR with post-click bias → ESMM.  
- Explicit task dependency → AITM.  

