---
title: Matching Model Tutorial
description: Torch-RecHub matching model tutorial, including practical examples of DSSM, GRU4Rec, and MIND models
---

# Matching Model Tutorial

This tutorial introduces how to use various matching models in Torch-RecHub. We will use the MovieLens dataset as an example.

## Data Preparation

First, we need to prepare the data. The MovieLens dataset contains user ratings for movies:

```python
import pandas as pd
import numpy as np
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input

# Load data
df = pd.read_csv("movielens.csv")

# Define user features
user_features = [
    SparseFeature("user_id", vocab_size=user_num, embed_dim=16),
    SparseFeature("gender", vocab_size=3, embed_dim=16),
    SparseFeature("age", vocab_size=10, embed_dim=16),
    SparseFeature("occupation", vocab_size=25, embed_dim=16)
]
# Add user history sequence features
user_features += [
    SequenceFeature("hist_movie_id", vocab_size=item_num, embed_dim=16,
                    pooling="mean", shared_with="movie_id")
]

# Define item features
item_features = [
    SparseFeature("movie_id", vocab_size=item_num, embed_dim=16),
    SparseFeature("cate_id", vocab_size=cate_num, embed_dim=16)
]
```

## Basic Two-Tower Model (DSSM)

DSSM is the most basic two-tower model, modeling users and items separately:

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

# Model configuration
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"}
)

# Training configuration
trainer = MatchTrainer(
    model=model,
    mode=0,  # point-wise training
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=10,
    device="cuda:0"
)

# Train model
trainer.fit(train_dl, val_dl)
```

## Sequential Recommendation Model (GRU4Rec)

GRU4Rec models user behavior sequences through GRU networks:

```python
from torch_rechub.models.matching import GRU4Rec
from torch_rechub.basic.features import SequenceFeature

# Define sequence features
history_features = [SequenceFeature("hist_movie_id", vocab_size=item_num, embed_dim=16, pooling=None)]
neg_item_feature = [SparseFeature("neg_items", vocab_size=item_num, embed_dim=16)]

# Model configuration
model = GRU4Rec(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64], "num_layers": 2, "dropout": 0.2},
    temperature=1.0
)

# Training configuration
trainer = MatchTrainer(
    model=model,
    mode=2,  # list-wise training
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## Multi-Interest Model (MIND)

MIND model can capture users' diverse interests:

```python
from torch_rechub.models.matching import MIND

# Model configuration
model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    temperature=1.0,
    interest_num=4
)

# Training configuration
trainer = MatchTrainer(
    model=model,
    mode=2,  # list-wise training
    optimizer_params={"lr": 0.001},
    n_epoch=10,
    device="cuda:0"
)
```

## Model Evaluation

Evaluate using common retrieval metrics:

```python
# Evaluate model
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")

# Get user/item vectors for offline evaluation
user_embeddings = trainer.inference_embedding(model, "user", user_dl, model_path="./")
item_embeddings = trainer.inference_embedding(model, "item", item_dl, model_path="./")
```

## Vector Retrieval

Trained models can be used to generate user and item vector representations for fast retrieval:

```python
from torch_rechub.utils.match import Annoy

# Build Annoy index
annoy = Annoy(n_trees=10)
annoy.fit(item_embeddings)

# Query similar items
similar_items, scores = annoy.query(user_embeddings[0], topk=10)
print(f"Top 10 similar items: {similar_items}")
```

## Advanced Tips

1. Temperature Coefficient Adjustment
```python
# Temperature coefficient affects the distribution of similarity scores
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,  # Smaller temperature makes distribution sharper
    user_params={"dims": [256, 128, 64]},
    item_params={"dims": [256, 128, 64]}
)
```

2. Regularization Configuration
```python
trainer = MatchTrainer(
    model=model,
    mode=0,
    regularization_params={
        "embedding_l2": 1e-5,
        "dense_l2": 1e-5
    }
)
```

3. Model Save and Load
```python
import torch

# Save model
torch.save(model.state_dict(), "model.pth")

# Load model
model.load_state_dict(torch.load("model.pth"))
```

## Notes

1. Choose appropriate training mode (point-wise/pair-wise/list-wise)
2. Pay attention to sequence feature length and padding method
3. Adjust negative sample ratio based on actual scenarios
4. Set batch_size and learning rate appropriately
5. Use L2 regularization to prevent overfitting
