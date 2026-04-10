---
title: Retrieval Tutorial
description: Torch-RecHub retrieval tutorial covering DSSM, GRU4Rec, MIND, and vector search
---

# Retrieval Tutorial

This tutorial focuses on the common retrieval workflow: building user/item features, generating sequence samples, training two-tower or sequence retrieval models, exporting embeddings, and running ANN search. All examples assume you are running from the **repository root** and using the built-in `MovieLens-1M` sample dataset.

## 1. Data Preparation

### 1. Load sample data

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

data = pd.read_csv("examples/matching/data/ml-1m/ml-1m_sample.csv")
# For this minimal example, use the first genre as a lightweight categorical feature.
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])

user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]
```

### 2. Encode features and build profiles

```python
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    # Reserve 0 for padding, consistent with the embedding conventions in the framework.
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

# Profile tables store static user/item attributes and will later be merged into the sequence samples.
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates("user_id")
item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")
```

### 3. Generate sequence samples

#### DSSM (point-wise)

```python
# point-wise: each user-item sample gets an independent 0/1 label
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,
    neg_ratio=3,
    min_item=0,
)

# gen_model_input packs user profile, target item, and history sequence into a model-ready dict.
x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train.pop("label")
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
```

#### GRU4Rec / MIND (list-wise)

```python
# list-wise: each positive sample is paired with multiple negatives
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=2,
    neg_ratio=3,
    min_item=0,
)

x_train = gen_model_input(
    df_train,
    user_profile,
    user_col,
    item_profile,
    item_col,
    seq_max_len=50,
    padding="post",
    truncating="post",
)
y_train = [0] * len(df_train)  # In list-wise mode, the first position is the positive sample.
x_test = gen_model_input(
    df_test,
    user_profile,
    user_col,
    item_profile,
    item_col,
    seq_max_len=50,
    padding="post",
    truncating="post",
)
```

## 2. DSSM: Basic Two-Tower Retrieval

### 1. Define features

```python
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
user_features += [
    # In DSSM, the history sequence is compressed into a single fixed-length vector.
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="mean", shared_with="movie_id")
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in item_cols
]

all_item = df_to_dict(item_profile)
test_user = x_test
```

### 2. Train the model

```python
import os
import torch

from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

torch.manual_seed(2022)

# MatchDataGenerator returns three DataLoaders: train, user inference, and item inference.
dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,  # More stable for Windows / notebook environments.
)

# DSSM is the standard two-tower retrieval baseline.
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

os.makedirs("./saved/dssm", exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=0,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path="./saved/dssm",
)

trainer.fit(train_dl)
```

### 3. Export embeddings

```python
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path="./saved/dssm",
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path="./saved/dssm",
)

print(user_embedding.shape, item_embedding.shape)
```

## 3. Sequential Retrieval: GRU4Rec / MIND

Compared with DSSM, these models differ mainly in two ways:

- they need list-wise training samples with `mode=2`
- both history sequences and negative samples should use `SequenceFeature(..., pooling="concat")`

### 1. Define sequence features

```python
user_cols = ["user_id", "gender", "age", "occupation", "zip"]

user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
history_features = [
    # The full sequence tensor is required here and will be modeled inside GRU4Rec / MIND.
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="concat", shared_with="movie_id")
]
item_features = [SparseFeature("movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16)]
neg_item_feature = [
    # In list-wise training, negative items are also kept as a full sequence tensor.
    SequenceFeature("neg_items", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="concat", shared_with="movie_id")
]
```

### 2. GRU4Rec

```python
from torch_rechub.models.matching import GRU4Rec

# GRU4Rec models sequential interest evolution with an RNN.
model = GRU4Rec(
    user_features,
    history_features,
    item_features,
    neg_item_feature,
    user_params={"dims": [128, 64, 16]},
    temperature=0.02,
)
```

### 3. MIND

```python
from torch_rechub.models.matching import MIND

# MIND decomposes a user into multiple interest capsules, which helps when user interests are diverse.
model = MIND(
    user_features,
    history_features,
    item_features,
    neg_item_feature,
    max_length=50,
    temperature=0.02,
)
```

### 4. Training pattern

```python
import os

# For sequence retrieval, num_workers=0 is still recommended, especially on Windows.
dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,
)

os.makedirs("./saved/matching_sequence", exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=2,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path="./saved/matching_sequence",
)

trainer.fit(train_dl)
```

## 4. Vector Search

Torch-RecHub provides wrappers for `Annoy / Faiss / Milvus`. A minimal example:

```python
from torch_rechub.utils.match import Annoy

annoy = Annoy(n_trees=10)
annoy.fit(item_embedding)
similar_items, scores = annoy.query(user_embedding[0], topk=10)
print(similar_items)
```

> If the dependency is missing, install it first, for example: `pip install annoy`.

## 5. FAQ

### Q1: Why does DSSM use `pooling="mean"` while GRU4Rec / MIND use `pooling="concat"`?

- DSSM compresses the history into one fixed vector, so `mean` pooling is enough
- GRU4Rec / MIND need the full sequence tensor, so `concat` is required

### Q2: Why create `./saved/...` before training?

`MatchTrainer` does not create `model_path` automatically, so the examples create the directory explicitly.

### Q3: Why set `num_workers=0` on Windows?

Retrieval pipelines create three DataLoaders at once: training, user inference, and item inference. In Windows and notebook environments, multiple workers are more likely to cause permission or handle issues, so `num_workers=0` is the safer default.

### Q4: Where should I go next?

- [DSSM tutorial](/tutorials/models/matching/dssm)
- [YoutubeDNN tutorial](/tutorials/models/matching/youtube_dnn)
- [MIND tutorial](/tutorials/models/matching/mind)
