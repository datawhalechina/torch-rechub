---
title: Ranking Model Tutorial
description: Tutorial on ranking models including Wide & Deep, DeepFM, DIN, and DCN-V2 with feature engineering tips
---

# Ranking Model Tutorial

This tutorial will introduce how to use various ranking models in Torch-RecHub. We'll use the Criteo and Avazu datasets as examples.

## Data Preparation

First, we need to prepare the data and process features:

```python
import pandas as pd
import numpy as np
from rechub.utils import DataGenerator
from rechub.models import *
from rechub.trainers import *

# Load data
df = pd.read_csv("criteo_sample.csv")

# Feature column definitions
sparse_features = ['C1', 'C2', 'C3', ..., 'C26']
dense_features = ['I1', 'I2', 'I3', ..., 'I13']
features = sparse_features + dense_features
```

## Wide & Deep Model

Wide & Deep model combines memorization and generalization capabilities:

```python
# Model configuration
model = WideDeep(
    wide_features=sparse_features,
    deep_features=features,
    hidden_units=[256, 128, 64],
    dropout_rates=[0.1, 0.1, 0.1])

# Training configuration
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10,
                 device='cuda:0')

# Train model
trainer.fit(train_dataloader, val_dataloader)
```

## DeepFM Model

DeepFM model uses factorization machines and deep networks to model feature interactions:

```python
# Model configuration
model = DeepFM(
    features=features,
    hidden_units=[256, 128, 64],
    dropout_rates=[0.1, 0.1, 0.1],
    embedding_dim=16)

# Training configuration
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10)
```

## DIN (Deep Interest Network)

DIN model uses attention mechanism to model user interests:

```python
# Generate behavior sequence features
behavior_features = ['item_id', 'category_id']
seq_features = generate_seq_feature(df,
                                  user_col='user_id',
                                  item_col='item_id',
                                  time_col='timestamp',
                                  item_attribute_cols=['category_id'])

# Model configuration
model = DIN(
    features=features,
    behavior_features=behavior_features,
    attention_units=[80, 40],
    hidden_units=[256, 128, 64],
    dropout_rate=0.1)

# Training configuration
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10)
```

## DCN-V2 Model

DCN-V2 explicitly models feature interactions through cross network:

```python
# Model configuration
model = DCNV2(
    features=features,
    cross_num=3,
    hidden_units=[256, 128, 64],
    dropout_rates=[0.1, 0.1, 0.1],
    cross_parameterization='matrix')  # or 'vector'

# Training configuration
trainer = Trainer(model=model,
                 optimizer_params={'lr': 0.001},
                 n_epochs=10)
```

## Model Evaluation

Use common ranking metrics for evaluation:

```python
# Evaluate model
auc = evaluate_auc(model, test_dataloader)
log_loss = evaluate_logloss(model, test_dataloader)
print(f"AUC: {auc:.4f}")
print(f"LogLoss: {log_loss:.4f}")
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
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred, target):
        # Implement Focal Loss
        pass

trainer = Trainer(model=model,
                 loss_fn=FocalLoss(alpha=0.25, gamma=2))
```

2. Learning Rate Scheduling
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

trainer = Trainer(model=model,
                 scheduler='cosine',  # Use cosine annealing scheduler
                 scheduler_params={'T_max': 10})
```

## Important Notes

1. Handle missing values and outliers appropriately
2. Pay attention to feature engineering importance
3. Choose appropriate evaluation metrics
4. Focus on model interpretability
5. Balance model complexity and efficiency
6. Handle class imbalance issues

