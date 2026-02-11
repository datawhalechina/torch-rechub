---
title: Quick Start
description: Torch-RecHub Quick Start Guide - Run your first recommendation model in 5 minutes
---

# Quick Start

This tutorial will help you run a complete recommendation model training pipeline in **5 minutes**.

## Installation

```bash
pip install torch-rechub
```

> 💡 We recommend installing optional dependencies for full functionality:
> ```bash
> pip install torch-rechub[all]  # Includes ONNX export, experiment tracking, etc.
> ```

---

## Example 1: CTR Prediction (Ranking Model)

This is a complete, ready-to-run DeepFM model training example:

```python
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# ========== 1. Load Data ==========
# Using built-in Criteo sample data (100 records for demo)
# Full dataset: https://www.kaggle.com/c/criteo-display-ad-challenge
data_url = "https://raw.githubusercontent.com/datawhalechina/torch-rechub/main/examples/ranking/data/criteo/criteo_sample.csv"
data = pd.read_csv(data_url)
print(f"Dataset size: {len(data)} records")

# ========== 2. Feature Engineering ==========
# Criteo dataset contains 13 dense features (I1-I13) and 26 sparse features (C1-C26)
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# Fill missing values
data[sparse_features] = data[sparse_features].fillna("-996")
data[dense_features] = data[dense_features].fillna(0)

# Normalize dense features
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# Encode sparse features
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat].astype(str))

# ========== 3. Define Feature Types ==========
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

# ========== 4. Create DataLoader ==========
x = data.drop(columns=["label"])
y = data["label"]

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],
    batch_size=256
)

# ========== 5. Define Model ==========
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
)

# ========== 6. Train Model ==========
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=5,
    device="cpu",  # Use GPU: "cuda:0"
)

trainer.fit(train_dl, val_dl)

# ========== 7. Evaluate Model ==========
auc = trainer.evaluate(model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

**Expected Output:**

```
train: 100%|██████████| 1/1 [00:00<00:00,  4.47it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 333.15it/s]
epoch: 0 validation: auc: 0.3666666666666667
epoch: 1
train: 100%|██████████| 1/1 [00:00<00:00, 111.11it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 399.08it/s]
epoch: 1 validation: auc: 0.3666666666666667
epoch: 2
train: 100%|██████████| 1/1 [00:00<00:00, 95.60it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 492.75it/s]
epoch: 2 validation: auc: 0.33333333333333337
epoch: 3
train: 100%|██████████| 1/1 [00:00<00:00, 90.90it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 499.20it/s]
epoch: 3 validation: auc: 0.3
epoch: 4
train: 100%|██████████| 1/1 [00:00<00:00, 79.91it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 500.27it/s]
epoch: 4 validation: auc: 0.3333333333333333
validation: 100%|██████████| 1/1 [00:00<00:00, 249.71it/s]Test AUC: 0.9545
```

---

## Example 2: Retrieval Model (Two-Tower DSSM)

This is a complete, ready-to-run DSSM two-tower model training example:

```python
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import df_to_dict, MatchDataGenerator
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input

torch.manual_seed(2022)

# ========== 1. Load Data ==========
# Using built-in MovieLens sample data
data_url = "https://raw.githubusercontent.com/datawhalechina/torch-rechub/main/examples/matching/data/ml-1m/ml-1m_sample.csv"
data = pd.read_csv(data_url)
print(f"Dataset size: {len(data)} records")

# Process genres feature
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])

# ========== 2. Feature Encoding ==========
user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]

feature_max_idx = {}
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat]) + 1  # +1 to reserve 0 for padding
    feature_max_idx[feat] = data[feat].max() + 1

# ========== 3. Define User Tower and Item Tower Features ==========
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_profile = data[user_cols].drop_duplicates("user_id")
item_profile = data[item_cols].drop_duplicates("movie_id")

# ========== 4. Generate Sequence Features and Training Data ==========
df_train, df_test = generate_seq_feature_match(
    data,
    user_col,
    item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,  # point-wise
    neg_ratio=3,
    min_item=0
)

x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train["label"]
x_train = {k: v for k, v in x_train.items() if k != "label"}
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)

# ========== 5. Define Feature Types ==========
user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
user_features += [
    SequenceFeature(
        "hist_movie_id",
        vocab_size=feature_max_idx["movie_id"],
        embed_dim=16,
        pooling="mean",
        shared_with="movie_id"
    )
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in item_cols
]

# ========== 6. Create DataLoader ==========
all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

# ========== 7. Define Model ==========
model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [128, 64], "activation": "prelu"},
    item_params={"dims": [128, 64], "activation": "prelu"},
)

# ========== 8. Train Model ==========
trainer = MatchTrainer(
    model,
    mode=0,  # point-wise
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",
    model_path="./",
)

trainer.fit(train_dl)

# ========== 9. Export Embeddings ==========
user_embedding = trainer.inference_embedding(model, mode="user", data_loader=test_dl, model_path="./")
item_embedding = trainer.inference_embedding(model, mode="item", data_loader=item_dl, model_path="./")

print(f"User embedding shape: {user_embedding.shape}")
print(f"Item embedding shape: {item_embedding.shape}")
```

**Expected Output:**

```
n_train: 384, n_test: 2
0 cold start user dropped
epoch: 0
train: 100%|██████████| 2/2 [00:19<00:00,  9.81s/it]
epoch: 1
train: 100%|██████████| 2/2 [00:19<00:00,  9.65s/it]
user inference: 100%|██████████| 1/1 [00:04<00:00,  4.90s/it]
item inference: 100%|██████████| 1/1 [00:05<00:00,  5.23s/it]User embedding shape: torch.Size([2, 64])
Item embedding shape: torch.Size([93, 64])
```

---

## Next Steps

🎉 Congratulations! You've successfully run your first recommendation model. Here's what you can explore next:

- 📚 Check [Model Documentation](../models/intro.md) for more models (DCN, MMOE, YoutubeDNN, etc.)
- 🔧 Check [Full Examples](https://github.com/datawhalechina/torch-rechub/tree/main/examples) for more datasets and training tips
- 🚀 Check [Model Serving](../serving/intro.md) to learn ONNX export and vector indexing
- 📊 Check [Experiment Tracking](../tools/tracking.md) to use MLflow/TensorBoard for logging

---

## FAQ

### Q: How to use GPU for training?

Change `device="cpu"` to `device="cuda:0"`:

```python
trainer = CTRTrainer(model, device="cuda:0", ...)
```

### Q: How to save and load models?

```python
# Save
torch.save(model.state_dict(), "model.pth")

# Load
model.load_state_dict(torch.load("model.pth"))
```

### Q: How to export ONNX models?

```python
trainer.export_onnx("model.onnx")

# Two-tower models can export separately
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```
