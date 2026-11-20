---
title: Multi-task models
description: Multi-task recommendation models and label order in Torch-RecHub
---

# Multi-task models

One product impression can carry several labels, such as a click and a purchase. A multi-task model reads that row and predicts a separate output column for each target. Choose the column order before training. Use the same order for the labels, task types and metric names. [Ali-CCP multi-task tutorial](/tutorials/pipeline)

## 1. SharedBottom

SharedBottom passes the features through a shared network, then sends its output to a separate tower for each task. It is a simple baseline when comparing several targets.

A task’s loss updates its own tower and the shared bottom network. The order of `task_types` determines how each column is produced. Classification outputs a probability and regression outputs a numeric value.

### Usage

```python
from torch_rechub.models.multi_task import SharedBottom
from torch_rechub.basic.features import SparseFeature, DenseFeature

# Define the features
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

# Create the model
model = SharedBottom(
    features=common_features,
    task_types=["classification", "classification"],  # Two classification tasks
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # First task tower
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Second task tower
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features shared by all tasks | Required |
| task_types | list | Ordered task types: "classification" or "regression" | Required |
| bottom_params | dict | Shared bottom network parameters | Required |
| tower_params_list | list | Network parameters for each task tower | Required |

## 2. ESMM

ESMM models conversion after a click. Its two towers estimate click probability (CTR) and conversion probability given a click (CVR). Their product estimates the probability of both a click and a conversion on an impression (CTCVR).

### Paper

```
Xiao, Jun, et al. "Entire space multi-task model: An effective approach for estimating post-click conversion rate." Proceedings of the 41st international ACM SIGIR conference on research & development in information retrieval. 2018.
```

The towers share the input embeddings and learn their own tower weights. Here ESMM returns three columns in the order `[cvr, ctr, ctcvr]`. `MTLTrainer` uses the second (CTR) and third (CTCVR) columns to compute the training loss.

### Usage

```python
from torch_rechub.models.multi_task import ESMM

# Create the model
model = ESMM(
    user_features=user_features,
    item_features=item_features,
    cvr_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    ctr_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User-side sparse features | Required |
| item_features | list | Item-side sparse features | Required |
| cvr_params | dict | MLP parameters for the CVR tower | Required |
| ctr_params | dict | MLP parameters for the CTR tower | Required |

Pass labels to `MTLTrainer` with shape `[B, 3]` and column order `[cvr, ctr, ctcvr]`. Set all three entries of `task_types` to `"classification"`.

## 3. MMOE

In MMOE, several small networks called experts each process the same input. Each task has its own gating network that decides how much each expert contributes for this example.

### Paper

```
Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

Click prediction and watch-time prediction can share the same experts and still give them different weights. `n_expert` controls how many experts to use. Set `tower_params_list` in task order to configure the output networks.

### Usage

```python
import os
import numpy as np

from torch_rechub.models.multi_task import MMOE

# Create the model
model = MMOE(
    features=common_features,
    task_types=["classification", "regression"],  # One classification task and one regression task
    n_expert=8,  # Number of experts
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # Classification tower parameters
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Regression tower parameters
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features shared by all tasks | Required |
| task_types | list | Ordered task types: "classification" or "regression" | Required |
| n_expert | int | Number of expert networks | Required |
| expert_params | dict | Expert network parameters | Required |
| tower_params_list | list | Network parameters for each task tower | Required |

## 4. PLE

PLE provides dedicated experts for each task along with a pool of shared experts. The click task can use both its own experts and the shared ones. Another task can choose its own combination.

### Paper

```
Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations." Fourteenth ACM Conference on Recommender Systems. 2020.
```

Experts and gates can be stacked in several layers. When comparing PLE and MMOE, examine each task’s validation metric to see whether one task improves at the expense of another.

### Usage

```python
from torch_rechub.models.multi_task import PLE

# Create the model
model = PLE(
    features=common_features,
    task_types=["classification", "regression"],
    n_level=2,  # Number of PLE levels
    n_expert_specific=4,  # Task-specific experts per task at each level
    n_expert_shared=4,  # Shared experts per level
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # Classification tower parameters
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Regression tower parameters
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features shared by all tasks | Required |
| task_types | list | Ordered task types: "classification" or "regression" | Required |
| n_level | int | Number of PLE levels | Required |
| n_expert_specific | int | Task-specific experts per task at each level | Required |
| n_expert_shared | int | Shared experts per level | Required |
| expert_params | dict | Expert network parameters | Required |
| tower_params_list | list | Network parameters for each task tower | Required |

## 5. AITM

AITM is designed for ordered binary targets such as clicking, adding to the cart and purchasing. A later stage can use the preceding stage’s representation, so ordering the tasks is part of configuring the model.

### Paper

```
Tang, Jiaxi, et al. "Learning Task Relationships in Multi-task Learning with Adaptive Information Transfer." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

Each task uses its own bottom network to compute a representation. Attention incorporates the information passed from the previous task. Each tower outputs a probability.

### Usage

```python
from torch_rechub.models.multi_task import AITM

# Create the model
model = AITM(
    features=common_features,
    n_task=2,
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # First task tower
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Second task tower
    ],
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Features shared by all tasks | Required |
| n_task | int | Number of ordered binary classification tasks | Required |
| bottom_params | dict | Bottom MLP parameters for each task | Required |
| tower_params_list | list | Network parameters for each task tower | Required |

AITM uses binary classification for every task and passes information forward in task-column order. Its constructor does not accept `task_types` or `attention_params`.

## 6. Choosing a multi-task model

When several labels describe the same impression, start with SharedBottom and check how well it learns each task. Compare MMOE and PLE to let tasks learn how much information to share. ESMM is suited to conversions that follow clicks. AITM passes information through a sequence of known funnel stages.

## 7. Training and checking labels

The [Ali-CCP tutorial](/tutorials/pipeline) has runnable training and evaluation examples. MMOE and PLE use two label columns, `[cvr, ctr]`. ESMM uses `[cvr, ctr, ctcvr]`, with `ctcvr = ctr * cvr`. Label the scores returned by `evaluate` in the same order to check each task independently.

## 8. Reading multi-task validation results

Suppose the model outputs `[cvr, ctr]`. Printing CTR first would label the conversion AUC as click AUC. Check that validation labels, prediction columns and printed names line up before comparing scores. Conversions are usually rarer than clicks, so also check that the validation set contains both positive and negative examples for each task.

When the two losses have very different scales, `MTLTrainer` can learn loss weights with `adaptive_params={"method": "uwl"}`. Start with the default weights and compare both variants on the same validation set.
