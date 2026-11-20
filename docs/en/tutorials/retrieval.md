---
title: Retrieval Tutorial
description: Torch-RecHub retrieval tutorial covering DSSM, GRU4Rec, MIND, and vector search
---

# Retrieval Tutorial

Given a user's viewing history, how do you find ten more movies to recommend? We'll train DSSM on the MovieLens sample, compute user and movie embeddings, and check whether the recommendations include the user's last recorded interaction. Run the examples from the repository root.

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

### 2. Encode IDs and build the user and movie tables

Treat the sample's movies as a known catalog. `user_profile` has one row per user, and `item_profile` has one row per movie. Train on encoded IDs and keep `raw_movie_ids` to look up the original movie IDs when displaying recommendations. With production logs, use the vocabulary and catalog that were available at prediction time.

```python
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    # Adding 1 leaves index 0 available for padding.
    data[feature] = lbe.fit_transform(data[feature]) + 1
    if feature == item_col:
        raw_movie_ids = lbe.classes_
    feature_max_idx[feature] = data[feature].max() + 1

# Profile tables store static user/item attributes and will later be merged into the sequence samples.
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates("user_id")
item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")
```

### 3. Generate sequence samples

By default, each positive interaction gets three negative examples. Positives have label 1 and negatives have label 0. To try in-batch sampling, set `in_batch_neg=True` below and rerun from this point. The training set then contains only positive interactions, and the trainer takes negatives from the other movies in the batch.

```python
in_batch_neg = False
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,
    in_batch_neg=in_batch_neg,
    neg_ratio=3,
    min_item=0,
)

# gen_model_input packs user profile, target item, and history sequence into a model-ready dict.
x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train.pop("label")
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
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
    # DSSM uses mean pooling to turn each user's history into a fixed-length vector.
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="mean", shared_with="movie_id", padding_idx=0)
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16, padding_idx=0)
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

# Both towers output 64-dimensional vectors.
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# Create the directory before training so the trainer can save the weights.
save_dir = "./saved/dssm_inbatch" if in_batch_neg else "./saved/dssm"
os.makedirs(save_dir, exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=0,
    in_batch_neg=in_batch_neg,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # Change to "cuda:0" for GPU.
    model_path=save_dir,
)

trainer.fit(train_dl)
```

For a batch of three positive pairs, scoring each of the three users against all three movies gives a 3 × 3 matrix. The first user's positive is movie 1, the second user's is movie 2, and the third user's is movie 3. All positives therefore lie on the diagonal. Each batch needs at least two positive pairs.

### 3. Export embeddings

`test_dl` builds the user inputs from each user's held-out example. `item_dl` iterates over the full movie catalog. In the two shapes printed below, the first dimension counts user records or movies respectively. Both second dimensions should be 64, the output width of each tower.

```python
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path=save_dir,
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path=save_dir,
)

print(user_embedding.shape, item_embedding.shape)
```

## 3. Retrieve ten movies and check for a hit

Install the `annoy` extra first. Annoy returns row numbers in the movie embedding table. Use those rows to find the encoded `movie_id` values, then map them back to the original IDs. The example prints `Hit@10` for one user: 1 if the held-out movie appears in the top ten, 0 otherwise.

```python
from torch_rechub.utils.match import Annoy

index = Annoy(n_trees=10)
index.fit(item_embedding)
rows, distances = index.query(user_embedding[0], n=10)
recommended = raw_movie_ids[all_item[item_col][rows] - 1]
held_out = raw_movie_ids[test_user[item_col][0] - 1]
print("Recommended movies", recommended.tolist())
print("Held-out movie", held_out, "Hit@10", int(held_out in recommended))
```

## 4. Train with explicit candidate lists

The list-wise YoutubeDNN, GRU4Rec and MIND examples put one positive item and a set of negatives in each row. Scores have shape `[B, 1 + K]`. The positive is in column 0, so every cross-entropy target is 0. Choose a command below and run it from the repository root to build the samples, train and evaluate.

```bash
cd examples/matching
uv run --extra annoy python run_ml_youtube_dnn.py --epoch 1 --device cpu
uv run --extra annoy python run_ml_gru4rec.py --epoch 1 --device cpu
uv run --extra annoy python run_ml_mind.py --epoch 1 --device cpu
```

## 5. FAQ

### Q1: Why does DSSM use `pooling="mean"` while GRU4Rec / MIND use `pooling="concat"`?

- DSSM compresses the history into one fixed vector, so `mean` pooling is enough
- GRU4Rec / MIND need the full sequence tensor, so `concat` is required

### Q2: Why create `./saved/...` before training?

Create the model directory with `os.makedirs(path, exist_ok=True)` before training. `MatchTrainer` expects `model_path` to exist.

### Q3: Why set `num_workers=0` on Windows?

Retrieval pipelines create three DataLoaders at once: training, user inference, and item inference. In Windows and notebook environments, multiple workers are more likely to cause permission or handle issues, so `num_workers=0` is the safer default.

### Training and retrieval examples by model

- [DSSM tutorial](/tutorials/models/matching/dssm)
- [YoutubeDNN tutorial](/tutorials/models/matching/youtube_dnn)
- [MIND tutorial](/tutorials/models/matching/mind)
