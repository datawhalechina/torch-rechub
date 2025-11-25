---
title: Multi-Task Learning Tutorial
description: Tutorial on multi-task learning models including SharedBottom, ESMM, MMoE, and PLE with examples
---

# Multi-Task Learning Tutorial

This tutorial will introduce how to use multi-task learning models in Torch-RecHub. We'll use Alibaba's e-commerce dataset as an example.

## Data Preparation

First, we need to prepare data for multi-task learning:

```python
import pandas as pd
import numpy as np
from rechub.utils import DataGenerator
from rechub.models import *
from rechub.trainers import *

# Load data
df = pd.read_csv("ali_ccp_data.csv")

# Feature definitions
user_features = ['user_id', 'age', 'gender', 'occupation']
item_features = ['item_id', 'category_id', 'shop_id', 'brand_id']
features = user_features + item_features

# Multi-task labels
tasks = ['click', 'conversion']  # CTR and CVR tasks
```

## SharedBottom Model

The most basic multi-task learning model with shared parameters in bottom layers:

```python
# Model configuration
model = SharedBottom(
    features=features,
    hidden_units=[256, 128],
    task_hidden_units=[64, 32],
    num_tasks=2,
    task_types=['binary', 'binary'])

# Training configuration
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)

# Train model
trainer.fit(train_dataloader, val_dataloader)
```

## ESMM (Entire Space Multi-Task Model)

A multi-task model that addresses sample selection bias:

```python
# Model configuration
model = ESMM(
    features=features,
    hidden_units=[256, 128, 64],
    tower_units=[32, 16],
    embedding_dim=16)

# Training configuration
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)
```

## MMoE (Multi-gate Mixture-of-Experts)

Implements soft parameter sharing between tasks through expert mechanism:

```python
# Model configuration
model = MMoE(
    features=features,
    expert_units=[256, 128],
    num_experts=8,
    num_tasks=2,
    expert_activation='relu',
    gate_activation='softmax')

# Training configuration
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)
```

## PLE (Progressive Layered Extraction)

Better models task relationships through layered extraction:

```python
# Model configuration
model = PLE(
    features=features,
    expert_units=[256, 128],
    num_experts=4,
    num_layers=3,
    num_shared_experts=2,
    task_types=['binary', 'binary'])

# Training configuration
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    n_epochs=10)
```

## Task Weight Optimization

### GradNorm

Use GradNorm algorithm to dynamically adjust task weights:

```python
# Configure GradNorm
trainer = MTLTrainer(
    model=model,
    optimizer_params={'lr': 0.001},
    task_weights_strategy='gradnorm',
    gradnorm_alpha=1.5)
```

### MetaBalance

Use MetaBalance optimizer to balance task gradients:

```python
from rechub.utils import MetaBalance

# Configure MetaBalance optimizer
optimizer = MetaBalance(
    model.parameters(),
    relax_factor=0.7,
    beta=0.9)

trainer = MTLTrainer(
    model=model,
    optimizer=optimizer)
```

## Model Evaluation

Use appropriate evaluation metrics for different tasks:

```python
# Evaluate model
results = evaluate_multi_task(model, test_dataloader)
for task, metrics in results.items():
    print(f"Task: {task}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"LogLoss: {metrics['logloss']:.4f}")
```

## Advanced Applications

1. Custom Task Loss Weights
```python
trainer = MTLTrainer(
    model=model,
    task_weights=[1.0, 0.5])  # Set fixed task weights
```

2. Get Shared and Task-Specific Layers
```python
from rechub.utils import shared_task_layers

shared_params, task_params = shared_task_layers(model)
```

3. Task-Specific Learning Rates
```python
trainer = MTLTrainer(
    model=model,
    task_specific_lr={'click': 0.001, 'conversion': 0.0005})
```

## Important Notes

1. Choose appropriate multi-task learning architecture
2. Pay attention to task correlations
3. Handle data imbalance between tasks
4. Set reasonable task weights
5. Monitor training progress for each task
6. Prevent negative transfer between tasks
7. Consider computational resource constraints

