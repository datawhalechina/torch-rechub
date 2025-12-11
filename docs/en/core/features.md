---
title: Feature Definitions
description: Torch-RecHub feature types
---

# Feature Definitions

Torch-RecHub provides three core feature classes for different data types.

## DenseFeature

Numeric features (e.g., age, income).

```python
from torch_rechub.basic.features import DenseFeature

dense_feature = DenseFeature(name="age", embed_dim=1)
```

Parameters: `name`, `embed_dim` (always 1).

## SparseFeature

Categorical features (e.g., city, gender).

```python
from torch_rechub.basic.features import SparseFeature

sparse_feature = SparseFeature(
    name="city",
    vocab_size=100,
    embed_dim=16,
    shared_with=None,  # share embeddings with another feature if needed
)
```

Parameters: `name`, `vocab_size`, `embed_dim` (auto if None), `shared_with`, `padding_idx`, `initializer`.

## SequenceFeature

Sequence or multi-hot features (e.g., behavior history, tags).

```python
from torch_rechub.basic.features import SequenceFeature

sequence_feature = SequenceFeature(
    name="user_history",
    vocab_size=10000,
    embed_dim=32,
    pooling="mean",  # mean, sum, concat
)
```

Parameters: `name`, `vocab_size`, `embed_dim` (auto if None), `pooling` (mean/sum/concat), `shared_with`, `padding_idx`, `initializer`.

## Usage Example

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature

dense_features = [
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1),
]

sparse_features = [
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    SparseFeature(name="gender", vocab_size=3, embed_dim=8),
]

sequence_features = [
    SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling="mean"),
]

all_features = dense_features + sparse_features + sequence_features
```

