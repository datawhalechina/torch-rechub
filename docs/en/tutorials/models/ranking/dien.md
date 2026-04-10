---
title: DIEN Tutorial
description: "Complete Deep Interest Evolution Network tutorial: modeling dynamic interest evolution"
---

# DIEN Tutorial

## 1. Model Overview and Use Cases

DIEN (Deep Interest Evolution Network), proposed by Alibaba at AAAI 2019, is an evolution of DIN. DIEN introduces an **Interest Extractor** (GRU) and an **Interest Evolution** module (AUGRU) to model how user interests **change over time**, instead of only doing a weighted sum over historical behaviors.

**Paper**: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672)

### Model Architecture

> **Note**: DIEN contains dynamic GRU-based computation, so `torchview` cannot fully trace its internal graph at the moment.

- **Interest Extractor Layer**: uses GRU to model user behavior sequences
- **Auxiliary Loss**: uses the next-step behavior as an auxiliary supervision signal
- **Interest Evolution Layer**: uses AUGRU with target-aware attention to model interest evolution relevant to the current target item

### Suitable Scenarios

- CTR prediction
- News / e-commerce scenarios where user interests change clearly over time
- Sequential recommendation tasks with temporally ordered behavior data

---

## 2. Data Preparation and Preprocessing

DIEN uses almost the same data pipeline as DIN and also relies on the sampled **Amazon Electronics** dataset. The additional input is `history_labels`, which marks whether each historical behavior is a positive signal.

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

# Load data
data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# Automatically generate sequence features ordered by timestamp.
train, val, test = generate_seq_feature(
    data=data,
    user_col="user_id",
    item_col="item_id",
    time_col="timestamp",
    item_attribute_cols=["cate_id"],
    min_item=0,
    shuffle=True,
)

user_num = data["user_id"].max() + 1
item_num = data["item_id"].max() + 1
cate_num = data["cate_id"].max() + 1
```

### Define Features

```python
# Base features (target item + user profile features)
features = [
    SparseFeature("user_id", vocab_size=user_num, embed_dim=16),
    SparseFeature("gender", vocab_size=data["gender"].max() + 1, embed_dim=8),
    SparseFeature("age", vocab_size=data["age"].max() + 1, embed_dim=8),
]

# History sequence features
history_features = [
    SequenceFeature("hist_item_id", vocab_size=item_num, embed_dim=16, pooling="concat", shared_with="item_id"),
    SequenceFeature("hist_cate_id", vocab_size=cate_num, embed_dim=16, pooling="concat", shared_with="cate_id"),
]

target_features = [
    SparseFeature("item_id", vocab_size=item_num, embed_dim=16),
    SparseFeature("cate_id", vocab_size=cate_num, embed_dim=16),
]
```

### Build DataLoaders

```python
x_train, y_train = df_to_dict(train), train["label"].values
x_val, y_val = df_to_dict(val), val["label"].values
x_test, y_test = df_to_dict(test), test["label"].values

dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=1024,
)
```

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import DIEN

# `history_labels` marks whether each historical behavior is a positive signal.
# In this simplified tutorial we use all ones. In real applications this should
# come from business labels such as click / purchase / watch / favorite.
history_labels = torch.ones(len(history_features))

model = DIEN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "dice"},
    history_labels=history_labels,
    alpha=0.2,
)
```

### 3.2 Parameter Details

- `history_labels`: fixed model-level configuration used by the auxiliary loss
- `alpha`: weight of the auxiliary loss term
- `pooling="concat"` is still required because DIEN needs the raw sequence

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("./saved/dien", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=3,
    earlystop_patience=2,
    device="cpu",
    model_path="./saved/dien",
)

trainer.fit(train_dl, val_dl)
```

## 5. Evaluation and Result Analysis

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"DIEN test AUC: {auc:.4f}")
```

- DIEN is often more suitable than DIN when long-term sequential evolution matters.
- It is also heavier than DIN, so the extra gain depends on sequence quality and data scale.

## 6. Tuning Suggestions

- Tune `alpha` carefully. If the auxiliary loss is too strong, the main objective can suffer.
- DIEN is more computationally expensive than DIN, so start from smaller hidden sizes and shorter sequences.
- If the dataset is small, DIN may already be a better trade-off.

## 7. FAQ and Troubleshooting

### Q1: What is the core difference between DIEN and DIN?

DIN uses target attention over historical behaviors, while DIEN further models how interests evolve with GRU / AUGRU.

### Q2: How should `history_labels` be obtained?

They should be derived from the business definition of positive and negative feedback for each step in the historical sequence.

## 8. Model Visualization

DIEN contains GRU-based dynamic computation, so automatic graph visualization is limited compared with plain MLP models.

## 9. ONNX Export

```python
trainer.export_onnx(
    "./saved/dien/dien.onnx",
    data_loader=test_dl,
    dynamic_batch=True,
)
```

## Full Example

The snippets above form a complete runnable example. Use them together with the same Amazon Electronics preprocessing flow shown in [examples/ranking/run_amazon_electronics.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/run_amazon_electronics.py).
