---
title: Recall Model Tutorial
description: Tutorial on using various recall models including DSSM, GRU4Rec, and MIND with practical examples
---

# Recall Model Tutorial

This tutorial will introduce how to use various recall models in Torch-RecHub. We'll use the MovieLens dataset as an example.

## Data Preparation

First, we need to prepare the data. The MovieLens dataset contains user ratings for movies:

```python
import pandas as pd
import numpy as np
from rechub.utils import DataGenerator
from rechub.models import *
from rechub.trainers import *

# Load data
df = pd.read_csv("movielens.csv")
user_features = ['user_id', 'age', 'gender', 'occupation']
item_features = ['movie_id', 'genre', 'year']
```

## Basic Two-Tower Model (DSSM)

DSSM is the most basic two-tower model, modeling users and items separately:

```python
# Model configuration
model = DSSM(user_features=user_features,
             item_features=item_features,
             hidden_units=[64, 32, 16],
             dropout_rates=[0.1, 0.1, 0.1])

# Training configuration
trainer = MatchTrainer(model=model,
                      mode=0,  # point-wise training
                      optimizer_params={'lr': 0.001},
                      n_epochs=10)

# Train model
trainer.fit(train_dataloader, val_dataloader)
```

## Sequential Recommendation Model (GRU4Rec)

GRU4Rec models user behavior sequences through GRU networks:

```python
# Generate sequence features
seq_features = generate_seq_feature(df,
                                  user_col='user_id',
                                  item_col='movie_id',
                                  time_col='timestamp',
                                  item_attribute_cols=['genre'])

# Model configuration
model = GRU4Rec(item_num=item_num,
                hidden_size=64,
                num_layers=2,
                dropout_rate=0.1)

# Training configuration
trainer = MatchTrainer(model=model,
                      mode=1,  # pair-wise training
                      optimizer_params={'lr': 0.001},
                      n_epochs=10)
```

## Multi-Interest Model (MIND)

The MIND model can capture users' diverse interests:

```python
# Model configuration
model = MIND(item_num=item_num,
            num_interests=4,
            hidden_size=64,
            routing_iterations=3)

# Training configuration
trainer = MatchTrainer(model=model,
                      mode=2,  # list-wise training
                      optimizer_params={'lr': 0.001},
                      n_epochs=10)
```

## Model Evaluation

Use common recall metrics for evaluation:

```python
# Calculate recall rate and hit rate
recall_score = evaluate_recall(model, test_dataloader, k=10)
hit_rate = evaluate_hit_rate(model, test_dataloader, k=10)
print(f"Recall@10: {recall_score:.4f}")
print(f"HitRate@10: {hit_rate:.4f}")
```

## Vector Retrieval

The trained model can be used to generate vector representations of users and items for fast retrieval:

```python
# Use Annoy for vector retrieval
from rechub.utils import Annoy

# Build index
item_vectors = model.get_item_vectors()
annoy = Annoy(metric='angular')
annoy.fit(item_vectors)

# Query similar items
user_vector = model.get_user_vector(user_id=1)
similar_items = annoy.query(user_vector, n=10)
```

## Advanced Techniques

1. Temperature Coefficient Adjustment
```python
trainer = MatchTrainer(model=model,
                      temperature=0.2,  # Add temperature coefficient
                      mode=2)
```

2. Negative Sampling
```python
from rechub.utils import negative_sample

neg_samples = negative_sample(items_cnt_order,
                            ratio=5,
                            method_id=1)  # Word2Vec-style sampling
```

3. Model Saving and Loading
```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model.load_state_dict(torch.load('model.pth'))
```

## Important Notes

1. Choose appropriate training mode (point-wise/pair-wise/list-wise)
2. Pay attention to sequence feature length and padding method
3. Adjust negative sample ratio based on actual scenarios
4. Set appropriate batch_size and learning rate
5. Use L2 regularization to prevent overfitting

