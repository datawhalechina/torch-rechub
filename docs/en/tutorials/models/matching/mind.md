---
title: MIND Tutorial
description: "Complete MIND tutorial: multi-interest retrieval model"
---

# MIND Tutorial

## 1. Model Overview and Use Cases

MIND (Multi-Interest Network with Dynamic Routing), proposed by Alibaba at CIKM 2019, is a **multi-interest retrieval model**. Unlike DSSM, which compresses a user into a **single vector**, MIND uses a capsule-network-style **dynamic routing** mechanism to extract **multiple interest vectors** from a user's behavior history and better model diverse interests.

**Paper**: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030v1)

### Model Architecture

> **Note**: MIND contains a dynamic-routing capsule mechanism, so `torchview` cannot fully trace its internal graph automatically.

- **Embedding Layer**: encodes user profile and historical behavior sequence
- **Capsule Network (Dynamic Routing)**: extracts multiple interest vectors from the sequence
- **User Representation**: multiple interest vectors with shape `[batch_size, interest_num, embed_dim]`
- **Training**: list-wise softmax training, similar to YoutubeDNN

### List-wise Forward Output

In `mode=2` list-wise training, `neg_item_feature` provides sampled negative items for each sample, and `item_tower` returns candidate item embeddings:

```text
item_embedding: [batch_size, 1 + n_neg_items, embed_dim]
```

MIND first uses the positive item to select the most relevant user interest as `best_interest_emb`:

```text
best_interest_emb: [batch_size, 1, embed_dim]
```

It then computes one dot-product logit for each candidate item:

```text
y = (best_interest_emb * item_embedding).sum(dim=-1)
y: [batch_size, 1 + n_neg_items]
```

The reduction must be over the embedding dimension `dim=-1`, not over the candidate item dimension. `MatchTrainer(mode=2)` uses `CrossEntropyLoss`, so `y_train = 0` means the first candidate item is the positive item.

### Suitable Scenarios

- Retrieval stage of recommendation systems
- Scenarios where user interests are clearly diverse
- Large-scale candidate retrieval with ANN search

---

## 2. Data Preparation and Preprocessing

This example also uses **MovieLens-1M** and builds list-wise retrieval data with `mode=2`.

```python
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

data = pd.read_csv("examples/matching/data/ml-1m/ml-1m_sample.csv")

for col in ["user_id", "movie_id", "gender", "age", "occupation", "zip"]:
    data[col] = LabelEncoder().fit_transform(data[col])

# mode=2: list-wise training
train, test = generate_seq_feature_match(
    data,
    user_col="user_id",
    item_col="movie_id",
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=0,
    mode=2,
)

train_user_input, train_item_input, y_train = gen_model_input(train, mode=2, seq_max_len=50)
test_user_input, test_item_input, y_test = gen_model_input(test, mode=2, seq_max_len=50)
```

### Define Features

```python
# History sequence features
history_features = [
    SequenceFeature("hist_movie_id", vocab_size=data["movie_id"].max() + 1, embed_dim=16, pooling="concat", shared_with="movie_id"),
]

# Positive item features
item_features = [
    SparseFeature("movie_id", vocab_size=data["movie_id"].max() + 1, embed_dim=16),
]

# Negative item features
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=data["movie_id"].max() + 1, embed_dim=16, pooling="concat", shared_with="movie_id"),
]

user_features = [
    SparseFeature("user_id", vocab_size=data["user_id"].max() + 1, embed_dim=16),
    SparseFeature("gender", vocab_size=data["gender"].max() + 1, embed_dim=8),
]
```

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.matching import MIND

model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    temperature=0.02,
    interest_num=4,
)
```

### 3.2 Parameter Details

- `interest_num`: number of interest vectors extracted for each user
- `max_length`: maximum history sequence length
- `temperature`: logit scaling during list-wise training

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import MatchTrainer

os.makedirs("./saved/mind", exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=2,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-6},
    n_epoch=5,
    device="cpu",
    model_path="./saved/mind",
)

dg = MatchDataGenerator(x=train_user_input, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    x_test=test_user_input,
    y_test=y_test,
    item_dataset=df_to_dict(data[["movie_id"]].drop_duplicates("movie_id")),
    batch_size=256,
    num_workers=0,
)

trainer.fit(train_dl)
```

## 5. Evaluation and Result Analysis

```python
# Generate embeddings
user_embedding = trainer.inference_embedding(model=trainer.model, mode="user", data_loader=test_dl)
item_embedding = trainer.inference_embedding(model=trainer.model, mode="item", data_loader=item_dl)

# MIND returns multiple user interest vectors instead of one.
# user_embedding shape: [n_users, interest_num, embed_dim]
print(user_embedding.shape)
```

### Vector Retrieval

```python
# Retrieve for each interest vector and merge the results
# For each user, search with every interest vector separately
```

## 6. Tuning Suggestions

- Start with `interest_num=4` and increase only if users truly have diverse interests
- Control sequence length carefully because dynamic routing adds cost

## 7. FAQ and Troubleshooting

### Q1: What is the online deployment difference between MIND and DSSM?

MIND produces multiple user vectors, so online retrieval usually needs multi-vector search and result merging instead of one user vector per request.

### Q2: How large should `interest_num` be?

Start from `4` or `6`. Too many interests can make retrieval noisier and more expensive.

## 8. Model Visualization

MIND uses dynamic routing internally, so automatic graph tracing is limited.

## 9. ONNX Export

```python
trainer.export_onnx("./saved/mind/mind.onnx", data_loader=test_dl, dynamic_batch=True)
```

## Full Example

The code blocks above form a complete runnable example. For a full MovieLens-based script, see [examples/matching/run_ml_mind.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/matching/run_ml_mind.py).
