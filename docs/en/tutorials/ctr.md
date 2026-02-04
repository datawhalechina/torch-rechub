---
title: Ranking Model Tutorial
description: Torch-RecHub ranking model tutorial, including Wide & Deep, DeepFM, DIN, DCN-V2 and feature engineering tips
---

# Ranking Model Tutorial

This tutorial introduces how to use various ranking models in Torch-RecHub. We will use the Criteo dataset as an example.

## Data Preparation

First, we need to prepare the data and perform feature processing:

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# Load data
df = pd.read_csv("criteo_sample.csv")

# Define feature columns
sparse_features = [f'C{i}' for i in range(1, 27)]
dense_features = [f'I{i}' for i in range(1, 14)]
```

## Wide & Deep Model

Wide & Deep model combines memorization and generalization capabilities:

```python
from torch_rechub.models.ranking import WideDeep
from torch_rechub.trainers import CTRTrainer

# Define features
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=df[name].nunique(), embed_dim=16) for name in sparse_features]

# Model configuration
model = WideDeep(
    wide_features=sparse_feas,
    deep_features=sparse_feas + dense_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# Training configuration
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)

# Train model
trainer.fit(train_dl, val_dl)
```

## DeepFM Model

DeepFM model uses factorization machines and deep networks to model feature interactions:

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

# Model configuration
model = DeepFM(
    deep_features=sparse_feas + dense_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# Training configuration
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)
```

## DIN (Deep Interest Network)

DIN model uses attention mechanism to model user interests:

```python
from torch_rechub.models.ranking import DIN
from torch_rechub.basic.features import SequenceFeature

# Define sequence features
history_feas = [SequenceFeature("hist_item_id", vocab_size=item_num, embed_dim=16, pooling=None)]
target_feas = [SparseFeature("item_id", vocab_size=item_num, embed_dim=16)]

# Model configuration
model = DIN(
    features=sparse_feas + dense_feas,
    history_features=history_feas,
    target_features=target_feas,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "dice"},
    attention_mlp_params={"dims": [64, 32], "activation": "dice"}
)

# Training configuration
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## DCN-V2 Model

DCN-V2 explicitly models feature interactions through cross networks:

```python
from torch_rechub.models.ranking import DCNv2

# Model configuration
model = DCNv2(
    features=sparse_feas + dense_feas,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# Training configuration
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## Model Evaluation

Evaluate using common ranking metrics:

```python
# Evaluate model
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

## Feature Engineering Tips

1. Feature Preprocessing
```python
# Categorical feature encoding
from sklearn.preprocessing import LabelEncoder
for feat in sparse_features:
    lbe = LabelEncoder()
    df[feat] = lbe.fit_transform(df[feat])

# Numerical feature normalization
from sklearn.preprocessing import MinMaxScaler
for feat in dense_features:
    scaler = MinMaxScaler()
    df[feat] = scaler.fit_transform(df[feat].values.reshape(-1, 1))
```

2. Feature Crossing
```python
# Manual feature crossing
df['cross_feat'] = df['feat1'].astype(str) + '_' + df['feat2'].astype(str)
```

## Advanced Applications

1. Custom Loss Function
```python
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        # Implement Focal Loss
        pass
```

2. Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

trainer = CTRTrainer(
    model=model,
    scheduler_fn=CosineAnnealingLR,
    scheduler_params={"T_max": 10}
)
```

## Notes

1. Handle missing values and outliers properly
2. Pay attention to the importance of feature engineering
3. Choose appropriate evaluation metrics
4. Focus on model interpretability
5. Balance model complexity and efficiency
6. Handle sample imbalance issues
