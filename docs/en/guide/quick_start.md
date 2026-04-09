---
title: Quick Start
description: "Torch-RecHub quick start guide: run your first recommendation model in 5 minutes"
---

# Quick Start

This page provides two minimal end-to-end pipelines that you can run directly:

- CTR ranking model: `DeepFM`
- Two-tower retrieval model: `DSSM`

All commands and code snippets below assume you are running from the **repository root** and using the sample datasets already included in the repo, so no external download is required.

## Installation

```bash
pip install torch-rechub
```

If you need ONNX export, visualization, experiment tracking, or other optional features, we recommend installing the extra dependencies as well:

```bash
pip install "torch-rechub[all]"
```

---

## Example 1: CTR Prediction (DeepFM)

This is a complete ranking training pipeline using the built-in `Criteo` sample dataset.

```python
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# ========== 1. Load sample data ==========
data_path = "examples/ranking/data/criteo/criteo_sample.csv"
data = pd.read_csv(data_path)
print(f"Dataset size: {len(data)}")

# ========== 2. Feature preprocessing ==========
# In Criteo, I1-I13 are dense features and C1-C26 are sparse features.
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# Keep the preprocessing consistent with the official examples.
data[sparse_features] = data[sparse_features].fillna("-996")
data[dense_features] = data[dense_features].fillna(0)

# Normalize dense features so the MLP trains more stably.
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# Encode sparse features as integer ids for embedding lookup.
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat].astype(str))

# DenseFeature / SparseFeature are the unified feature abstractions in Torch-RecHub.
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

# ========== 3. Build DataLoader ==========
# DataGenerator wraps pandas-style inputs into ready-to-train DataLoaders.
x = data.drop(columns=["label"])
y = data["label"]

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],
    batch_size=256,
)

# ========== 4. Define model ==========
# DeepFM combines low-order feature interactions (FM) and higher-order nonlinearity (MLP).
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

# ========== 5. Train and evaluate ==========
# CTRTrainer handles training, validation, early stopping, and evaluation.
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=2,
    device="cpu",  # Change to "cuda:0" for GPU training.
    model_path="./saved/quick_start_deepfm",
)

# CTRTrainer does not create model_path automatically, so create it first.
os.makedirs("./saved/quick_start_deepfm", exist_ok=True)
trainer.fit(train_dl, val_dl)

# It is safer to evaluate trainer.model so you use the best validated checkpoint.
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

If you prefer running the full example script directly, you can also execute:

```bash
cd examples/ranking
python run_criteo.py --model_name deepfm --epoch 2 --device cuda:0
```

---

## Example 2: Retrieval Model (DSSM)

This is a complete two-tower retrieval pipeline using the built-in `MovieLens-1M` sample dataset.

```python
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

torch.manual_seed(2022)

# ========== 1. Load sample data ==========
data_path = "examples/matching/data/ml-1m/ml-1m_sample.csv"
data = pd.read_csv(data_path)
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
print(f"Dataset size: {len(data)}")

# ========== 2. Encode sparse features ==========
# In this minimal MovieLens example, all fields are handled as sparse categorical features.
user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]

# Reserve 0 for padding / OOV.
feature_max_idx = {}
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat]) + 1
    feature_max_idx[feat] = data[feat].max() + 1

# user_profile / item_profile will later be merged into the generated sequence samples.
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]
user_profile = data[user_cols].drop_duplicates("user_id")
item_profile = data[item_cols].drop_duplicates("movie_id")

# ========== 3. Build sequence samples ==========
# point-wise retrieval training: each user-item pair gets a 0/1 label.
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

# gen_model_input turns user profile, target item, and history sequence into a model-ready dict.
x_train = gen_model_input(
    df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50
)
y_train = x_train.pop("label")
x_test = gen_model_input(
    df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50
)

# ========== 4. Define features ==========
# The DSSM user tower is usually built from profile features plus an aggregated history vector.
user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
user_features += [
    # Use mean pooling here because DSSM compresses the history sequence into one vector.
    SequenceFeature(
        "hist_movie_id",
        vocab_size=feature_max_idx["movie_id"],
        embed_dim=16,
        pooling="mean",
        shared_with="movie_id",
    )
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in item_cols
]

# ========== 5. DataLoader ==========
# Retrieval returns three DataLoaders: train, user inference, and item inference.
all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,  # More stable for Windows / notebook environments.
)

# ========== 6. Define model ==========
# DSSM is the standard two-tower retrieval baseline.
model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# ========== 7. Train and export embeddings ==========
# MatchTrainer handles retrieval training and embedding export.
trainer = MatchTrainer(
    model,
    mode=0,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # Change to "cuda:0" for GPU training.
    model_path="./saved/quick_start_dssm",
)

# Create the model directory first; inference_embedding will reuse it.
os.makedirs("./saved/quick_start_dssm", exist_ok=True)
trainer.fit(train_dl)

# Retrieval evaluation usually starts by exporting user / item embeddings.
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path="./saved/quick_start_dssm",
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path="./saved/quick_start_dssm",
)

print(f"User embedding shape: {user_embedding.shape}")
print(f"Item embedding shape: {item_embedding.shape}")
```

If you prefer running the example script directly, you can also execute:

```bash
cd examples/matching
python run_ml_dssm.py --epoch 2 --device cuda:0
```

If you hit DataLoader multiprocessing issues in Windows or notebook environments, it is better to run the Python code block above directly, because the script does not currently expose a `num_workers` argument while retrieval examples often need `num_workers=0`.

---

## Runtime Tips

- GPU training: change `device="cpu"` to `device="cuda:0"`
- Working directory: all paths in this page are relative to the **repository root**
- Save directory: trainers do not create `model_path` automatically, so run `os.makedirs(path, exist_ok=True)` first
- Windows environments: retrieval pipelines are more stable with `num_workers=0`

---

## Next Steps

- Read the [ranking tutorial](../tutorials/ctr.md) for `WideDeep / DeepFM / DCN / DIN`
- Read the [retrieval tutorial](../tutorials/retrieval.md) for `DSSM / YoutubeDNN / MIND`
- Read the [multi-task tutorial](../tutorials/pipeline.md) for `MMOE / PLE / ESMM`
- Read the [serving guide](../serving/intro.md) for ONNX export and vector indexing

---

## FAQ

### Q: How do I save and load a model?

```python
import torch

torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

### Q: Can I export ONNX models?

```python
# Ranking model
trainer.export_onnx("model.onnx")

# Matching model
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

### Q: Is there a minimal command just to verify my environment?

```bash
cd examples/ranking
python run_criteo.py --model_name deepfm --epoch 1 --device cuda:0
```
