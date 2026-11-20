---
title: MIND Tutorial
description: Learn several interest vectors per user with MIND and merge the movies retrieved by each vector
---

# MIND Tutorial

## 1. Model Overview and Use Cases

Someone who likes science fiction may also watch comedies. MIND extracts several interest vectors from the same viewing history, and each vector can retrieve movies from the catalog. The vectors are learned from interactions. There is no need to name or label the interests.

**Paper**: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030v1)

### Model Architecture

> **Note**: Because MIND uses a dynamically routed capsule network internally, torchview cannot currently trace its computation graph automatically, so no architecture visualization is provided.

- **Embedding Layer**: encodes user attributes and historical behavior sequences
- **Capsule Network (Dynamic Routing)**: extracts multiple interest vectors from the behavior sequence
- **User Representation**: multiple interest vectors rather than one, with shape `[batch_size, interest_num, embed_dim]`
- **Training**: list-wise (Softmax), similar to YoutubeDNN

### List-Wise Forward Output

During list-wise training with `mode=2`, `neg_item_feature` provides the set of negative samples for each example, and `item_tower` returns the candidate-item vectors:

```text
item_embedding: [batch_size, 1 + n_neg_items, embed_dim]
```

MIND first uses the positive item to select the most relevant `best_interest_emb` from the user's multiple interest vectors:

```text
best_interest_emb: [batch_size, 1, embed_dim]
```

It then computes the inner product with every candidate item and outputs the logits required by sampled softmax:

```text
y = (best_interest_emb * item_embedding).sum(dim=-1)
y: [batch_size, 1 + n_neg_items]
```

The reduction must be performed over the embedding dimension, `dim=-1`, rather than the candidate-item dimension. `MatchTrainer(mode=2)` uses `CrossEntropyLoss`, so `y_train = 0` means that candidate item 0 is the positive item.

During training, the positive movie selects an interest vector for scoring. At recommendation time the next movie is unknown, so query the index with each interest vector. Section 5 shows how to merge the results.

---

## 2. Data Preparation and Preprocessing

This tutorial uses the **MovieLens-1M** dataset. The data-processing flow is the same as for DSSM/YoutubeDNN and uses `mode=2` (list-wise) to build training data.

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

data = pd.read_csv("examples/matching/data/ml-1m/ml-1m_sample.csv")
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])

sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id']
user_col, item_col = "user_id", "movie_id"

feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates("user_id")
item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")

# mode=2: list-wise training
df_train, df_test = generate_seq_feature_match(
    data, user_col, item_col, time_col="timestamp",
    item_attribute_cols=[], sample_method=1, mode=2, neg_ratio=3, min_item=0
)

x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50, padding="post", truncating="post")
y_train = np.array([0] * df_train.shape[0])
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50, padding="post", truncating="post")
```

### Define Features

```python
user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']

user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]

# Historical behavior sequence
history_features = [
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"],
                    embed_dim=16, pooling="concat", shared_with="movie_id")
]

# Positive-item features
item_features = [
    SparseFeature("movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16)
]

# Negative-item features
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=feature_max_idx["movie_id"],
                    embed_dim=16, pooling="concat", shared_with="movie_id")
]

all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=2048, num_workers=0)
```

---

## 3. Model Configuration and Parameter Reference

### 3.1 Create the Model

```python
from torch_rechub.models.matching import MIND

model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,          # Maximum sequence length
    temperature=0.02,       # Only used for in-batch cross-entropy.
    interest_num=4          # Number of interest vectors
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Recommended Value |
|------|------|------|--------|
| `user_features` | `list[Feature]` | User-side features | User attributes |
| `history_features` | `list[Feature]` | User behavior sequence (`pooling="concat"`) | |
| `item_features` | `list[Feature]` | Positive-item features | |
| `neg_item_feature` | `list[Feature]` | Negative-item features | |
| `max_length` | `int` | Maximum sequence length | 50 |
| `temperature` | `float` | Temperature for in-batch cross-entropy. This example's `mode=2` forward pass uses dot products directly. | Used for in-batch sampling |
| `interest_num` | `int` | Number of extracted interest vectors | 4–8 |

Here `interest_num=4` gives each user four vectors, each the same width as a movie vector.

---

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import MatchTrainer

torch.manual_seed(2022)
save_dir = "./saved/mind/"
os.makedirs(save_dir, exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=2,                          # list-wise
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=10,
    device="cpu",
    model_path=save_dir
)

trainer.fit(train_dl)
```

---

## 5. Model Evaluation and Result Analysis

```python
# Generate embeddings
# MIND produces multiple user-interest vectors rather than a single user vector
user_embedding = trainer.inference_embedding(
    model=model, mode="user", data_loader=test_dl, model_path=save_dir
)
item_embedding = trainer.inference_embedding(
    model=model, mode="item", data_loader=item_dl, model_path=save_dir
)

# MIND user_embedding shape: [n_users, interest_num, embed_dim]
print(f"User Embedding shape: {user_embedding.shape}")
print(f"Item Embedding shape: {item_embedding.shape}")
```

> **Note**: MIND's User Embedding is a 3D tensor with shape `[n_users, interest_num, embed_dim]`. During vector retrieval, query separately with each interest vector, then merge and deduplicate the results.

### Vector Retrieval

Query the index with each active interest vector, then merge the candidates by ascending Annoy distance. When several interests retrieve the same movie, keep the closest match. The example checks Hit@10 against the first user's held-out movie.

```python
from torch_rechub.utils.match import Annoy

index = Annoy(n_trees=10)
index.fit(item_embedding)
candidates = []
for interest in user_embedding[0]:
    if interest.norm().item() > 0:
        rows, distances = index.query(interest, n=10)
        candidates.extend(zip(rows, distances))

candidates.sort(key=lambda pair: pair[1])
rows = list(dict.fromkeys(row for row, _ in candidates))[:10]
movie_ids = all_item["movie_id"][rows]
print("Hit@10", int(test_user["movie_id"][0] in movie_ids))
```

The example script evaluates all users and uses the matching vocabulary to recover the original item IDs.

## 6. Tuning Recommendations

Increasing `interest_num` gives the index more vectors to look up per user. Start with four and use the same held-out movies to compare Hit@10 and query time per user. If you change the history length, update both `seq_max_len` in preprocessing and `max_length` in the model.

## 7. FAQ and Troubleshooting

### Why can the same movie appear twice?

Two interest vectors can retrieve the same movie. Section 5 merges the results by distance and keeps the closest match for each movie, so it occupies only one recommendation slot.

---

## 8. Model Visualization Limitation

The project's visualization tool requires `pip install "torch-rechub[visualization]"` and a system installation of Graphviz. However, torchview cannot reliably trace MIND's current dynamic-routing loop. Do not call `visualize_model()` directly on MIND; this page instead documents its structure and tensor shapes above.

---

## 9. ONNX Export

First install the optional ONNX dependencies:

```bash
pip install "torch-rechub[onnx]"
```

The dynamic-routing computation is traced using the sequence length of the example input, so always validate the exported model with real inputs. The minimum export flow is:

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("mind_user_tower.onnx", mode="user")
exporter.export("mind_item_tower.onnx", mode="item")
```

---

## Run the example script

Run this from the repository root. The script saves the ID maps and weights, then evaluates the recommendations.

```bash
cd examples/matching
uv run --extra annoy python run_ml_mind.py --epoch 1 --device cpu
```
