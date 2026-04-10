---
title: BST Tutorial
description: "Complete Behavior Sequence Transformer tutorial: using Transformer for behavior modeling"
---

# BST Tutorial

## 1. Model Overview and Use Cases

BST (Behavior Sequence Transformer) was proposed by Alibaba in 2019. It introduces **Transformer self-attention** into recommendation systems and models **dependencies between any two items in the behavior sequence**, rather than only the relationship between the target item and history items as DIN does.

**Paper**: [Behavior Sequence Transformer for E-commerce Recommendation in Alibaba](https://arxiv.org/pdf/1905.06874)

### Model Architecture

> **Note**: BST contains Transformer-based dynamic sequence computation, so `torchview` cannot always render a complete architecture graph automatically.

- **Embedding Layer**: encodes user features, target item features, and behavior sequences
- **Transformer Encoder**: performs self-attention over behavior sequence + target item
- **MLP Layer**: combines Transformer output with other features and predicts the final score

### Suitable Scenarios

- CTR prediction
- Scenarios with long behavior sequences and complex dependencies between items
- Cases where stronger sequential modeling is needed than DIN-like attention pooling

---

## 2. Data Preparation and Preprocessing

BST uses the same **Amazon Electronics** data preparation flow as DIN / DIEN.

```python
import pandas as pd
import torch

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

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

```python
# Feature definition (same pattern as DIN)
features = [
    SparseFeature("user_id", vocab_size=user_num, embed_dim=16),
    SparseFeature("gender", vocab_size=data["gender"].max() + 1, embed_dim=8),
]

history_features = [
    SequenceFeature("hist_item_id", vocab_size=item_num, embed_dim=16, pooling="concat", shared_with="item_id"),
    SequenceFeature("hist_cate_id", vocab_size=cate_num, embed_dim=16, pooling="concat", shared_with="cate_id"),
]

target_features = [
    SparseFeature("item_id", vocab_size=item_num, embed_dim=16),
    SparseFeature("cate_id", vocab_size=cate_num, embed_dim=16),
]

# DataLoader
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
from torch_rechub.models.ranking import BST

model = BST(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    nhead=4,
    dropout=0.2,
    num_layers=1,
)
```

### 3.2 Parameter Details

- `nhead`: number of attention heads; the embedding dimension must be divisible by it
- `num_layers`: number of Transformer encoder layers
- `dropout`: regularization inside Transformer and the final MLP

## 4. Training Process and Code Example

```python
import os
from torch_rechub.trainers import CTRTrainer

os.makedirs("./saved/bst", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=3,
    earlystop_patience=2,
    device="cpu",
    model_path="./saved/bst",
)

trainer.fit(train_dl, val_dl)
```

## 5. Evaluation and Result Analysis

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"BST test AUC: {auc:.4f}")
```

- BST can outperform simpler sequence pooling methods when sequence relations are complex.
- It is also more expensive than DIN and may need more careful tuning.

## 6. Tuning Suggestions

- Check `embed_dim % nhead == 0` first.
- Try `nhead=2` or `4` before increasing `num_layers`.
- If training is slow or unstable, reduce sequence length, hidden size, or attention heads.

## 7. FAQ and Troubleshooting

### Q1: What is the core difference between BST and DIN?

DIN uses target-aware attention over the behavior sequence, while BST uses Transformer self-attention to model richer dependencies within the sequence.

### Q2: Why do I get an error about `embed_dim` and `nhead`?

The Transformer layer requires the embedding dimension to be divisible by `nhead`.

### Q3: How is BST for online inference?

BST is usually heavier than DIN / DIEN. For production use, sequence length and model size need to be controlled carefully.

## 8. Model Visualization

BST contains Transformer-based dynamic computation, so automatic visualization is more limited than for plain feed-forward models.

## 9. ONNX Export

```python
trainer.export_onnx(
    "./saved/bst/bst.onnx",
    data_loader=test_dl,
    dynamic_batch=True,
)
```

## Full Example

The snippets above form a complete runnable example. Use them together with the same Amazon Electronics preprocessing pipeline shown in [examples/ranking/run_amazon_electronics.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/run_amazon_electronics.py).
