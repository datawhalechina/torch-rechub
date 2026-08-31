---
title: Feature Definitions
description: Torch-RecHub feature types
---

# Feature Definitions

Torch-RecHub provides three core feature classes for different data types.

![Feature types to model inputs](/img/diagrams/feature_types_flow.png)

## DenseFeature

Numeric features (e.g., age, income).

```python
from torch_rechub.basic.features import DenseFeature

dense_feature = DenseFeature(name="age", embed_dim=1)
```

Parameters: `name`, `embed_dim`. Scalar inputs normally use 1; if the field is already a vector, set it to the actual vector width.

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
    padding_idx=0,    # use 0 for sequence padding
)
```

Parameters: `name`, `vocab_size`, `embed_dim` (auto if None), `pooling` (mean/sum/concat), `shared_with`, `padding_idx`, `initializer`. `mean` and `sum` reduce `(batch_size, seq_len, embed_dim)` to `(batch_size, embed_dim)`; `concat` preserves the sequence dimension.

Because `concat` returns sequence-shaped output, do not mix concat-pooled `SequenceFeature` objects with sparse or reduced sequence features in one `EmbeddingLayer` call. Request sequence embeddings separately from ordinary feature embeddings:

```python
input_user = embedding(x, user_features, squeeze_dim=True)  # (B, D)
history_emb = embedding(x, history_features)                 # (B, H, L, D)
single_history_emb = history_emb.squeeze(1)                  # (B, L, D), only when H == 1
```

## Model Input Contract

Feature objects describe how a model interprets fields; they do not encode raw categories or pad sequences for you. Input dictionaries normally follow this contract:

| Feature type | Typical shape | Value contract |
|---|---|---|
| `DenseFeature` | `(batch_size,)` or `(batch_size, embed_dim)` | Convertible to floating point |
| `SparseFeature` | `(batch_size,)` | Integer indices in `[0, vocab_size)` |
| `SequenceFeature` | `(batch_size, seq_len)` | Integer indices padded to a common length before model input |

When sequences are padded with `0`, set `padding_idx=0` explicitly. Otherwise, the default mask treats `-1`, not `0`, as padding.

With `shared_with="item_id"`, the same `EmbeddingLayer` feature list must also contain a feature named `item_id` that creates its own embedding. The shared table determines the effective vocabulary size and embedding dimension.

## Embedding Initialization

`initializer` accepts a callable initializer instance. The built-ins are `RandomNormal`, `RandomUniform`, `XavierNormal`, `XavierUniform`, and `Pretrained`; `SparseFeature` and `SequenceFeature` default to `RandomNormal(0, 0.0001)`.

```python
import torch

from torch_rechub.basic.features import SparseFeature
from torch_rechub.basic.initializers import Pretrained, XavierUniform

random_feature = SparseFeature(
    name="item_id",
    vocab_size=1000,
    embed_dim=32,
    padding_idx=0,
    initializer=XavierUniform(gain=1.0),
)

weights = torch.randn(1000, 32)
pretrained_feature = SparseFeature(
    name="pretrained_item_id",
    vocab_size=1000,
    embed_dim=32,
    padding_idx=0,
    initializer=Pretrained(weights, freeze=False),
)
```

`Pretrained` requires its weight shape to match `vocab_size` and `embed_dim` exactly. `freeze=True` is the default, so that embedding is not updated during training. The built-in random and Xavier initializers zero the `padding_idx` row when one is configured.

## Feature Instances and Embedding Ownership

> **Warning**: `SparseFeature` and `SequenceFeature` cache the `nn.Embedding` created by `get_embedding_layer()`. If the **same Feature instance** is passed to multiple models, those models use the same embedding parameters. Training or loading weights into one model therefore changes the embedding observed by the others.

The two EmbeddingLayer instances below unintentionally share the `city` embedding:

```python
from torch_rechub.basic.features import SparseFeature
from torch_rechub.basic.layers import EmbeddingLayer

features = [SparseFeature(name="city", vocab_size=100, embed_dim=16)]

embedding_a = EmbeddingLayer(features)
embedding_b = EmbeddingLayer(features)

assert embedding_a.embed_dict["city"] is embedding_b.embed_dict["city"]
```

For independent model comparisons, cross-validation, or ensemble training, create new Feature instances for every model:

```python
def build_features():
    return [SparseFeature(name="city", vocab_size=100, embed_dim=16)]

embedding_a = EmbeddingLayer(build_features())
embedding_b = EmbeddingLayer(build_features())

assert embedding_a.embed_dict["city"] is not embedding_b.embed_dict["city"]
```

Copying only the list does not help: `features.copy()` still contains the original Feature instances. This accidental cross-model sharing is different from explicitly using `shared_with` to share an embedding inside one model; the latter is intentional.

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
    SequenceFeature(
        name="user_history",
        vocab_size=10000,
        embed_dim=32,
        pooling="mean",
        padding_idx=0,
    ),
]

all_features = dense_features + sparse_features + sequence_features
```
