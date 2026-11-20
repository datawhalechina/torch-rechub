---
title: DIEN Tutorial
description: Train DIEN on product interaction histories and construct negative histories for its auxiliary loss
---

# DIEN Tutorial

## 1. Model Overview

DIEN reads a user's actions in order, updating an interest representation at each step. During training it also compares the item the user interacted with next against a sampled negative. This additional training objective is the auxiliary loss.

**Paper**: [Deep Interest Evolution Network for Click-Through Rate Prediction](https://arxiv.org/pdf/1809.03672)

- **Interest Extractor Layer**: GRU over behaviour sequences; auxiliary loss supervises each hidden state with positive/negative next-step samples (paper Eq.7)
- **Interest Evolution Layer**: AUGRU embeds attention into the update gate; attention is softmax-normalised over the full valid sequence (paper Eq.14-16)
- **Auxiliary loss**: $L_{aux} = -\frac{1}{N}\sum[\log\sigma(h_t \cdot e^+_{t+1}) + \log(1-\sigma(h_t \cdot e^-_{t+1}))]$
- **Padding**: index 0 is the padding token; excluded from GRU, AUGRU attention, and auxiliary loss; all-padding samples keep zero hidden state

---

## 2. Key Conventions

| Convention | Detail |
|------|------|
| padding index | Sequences are zero-padded (`generate_seq_feature` default). All sequence and target features must set `padding_idx=0`. |
| shared_with | Set `shared_with` for both `history_features` and `neg_history_features` to the **name of the target feature**. Here `target_item_id` owns the item embedding table. Both histories look up IDs in that table. `hist_item_id` already points to `target_item_id`, so the negative history also sets `shared_with="target_item_id"`. |

---

## 3. Data Preparation

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch_rechub.utils.data import generate_seq_feature, df_to_dict

raw = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# Replicate generate_seq_feature's encoding to build item2cate mapping
enc_data = raw.copy()
for feat in enc_data:
    le = LabelEncoder()
    enc_data[feat] = le.fit_transform(enc_data[feat]) + 1
enc_data = enc_data.astype('int32')
item2cate = enc_data[['item_id','cate_id']].drop_duplicates().set_index('item_id')['cate_id'].to_dict()
n_items, n_users, n_cates = enc_data['item_id'].max(), enc_data['user_id'].max(), enc_data['cate_id'].max()

train, val, test = generate_seq_feature(
    data=raw, user_col="user_id", item_col="item_id",
    time_col="time", item_attribute_cols=["cate_id"]
)
```

---

## 4. Negative Sequence Construction

The main `label` records whether the user clicked the current candidate. The auxiliary task uses a separate `neg_hist_item_id` for each history position. Look up the negative item's category in `item2cate` to keep the item and category paired.

```python
from examples.ranking.run_dien import build_neg_history

train, val, test = df_to_dict(train), df_to_dict(val), df_to_dict(test)
train_y, val_y, test_y = train.pop("label"), val.pop("label"), test.pop("label")

for split in [train, val, test]:
    neg_items, neg_cates = build_neg_history(split, "hist_item_id", item2cate, n_items)
    split["neg_hist_item_id"] = neg_items
    split["neg_hist_cate_id"] = neg_cates
```

---

## 5. Feature Definition

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature

features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]

# padding_idx=0 must be on target_features — they own the embedding tables
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0),
]

history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id", padding_idx=0),
]
neg_history_features = [
    SequenceFeature("neg_hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
    SequenceFeature("neg_hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id", padding_idx=0),
]
```

---

## 6. Model and Training

```python
import os
from torch_rechub.models.ranking import DIEN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

os.makedirs("./saved/dien", exist_ok=True)

dg = DataGenerator(train, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val, y_val=val_y, x_test=test, y_test=test_y, batch_size=4096
)

model = DIEN(
    features=features,
    history_features=history_features,
    neg_history_features=neg_history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    alpha=0.2,
)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=5,
    earlystop_patience=2,
    device="cpu",
    model_path="./saved/dien",
    loss_mode=False,  # forward returns (prediction, aux_loss)
)
trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

---

## 7. FAQ

### Q1: Core difference between DIEN and DIN?
DIN weights past actions according to the current candidate. DIEN also updates the user's state in time order and trains intermediate states using the actual next item and a sampled negative.

### Q2: Which feature should the negative history share an embedding with?
`EmbeddingLayer` registers a feature in `embed_dict` only when `shared_with=None`. The entries in `history_features` already share a `target_*` embedding, so there is no `hist_*` key in `embed_dict`. Set `shared_with` in `neg_history_features` to the same target feature, such as `target_item_id`.

### Q3: Why set `padding_idx=0` on `target_features`?
`history_features` and `neg_history_features` share the target feature's embedding table. Only setting `padding_idx=0` on the table owner (target feature) makes row 0 a true zero vector protected by `nn.Embedding`.

### Q4: Recommended sequence length?
The GRU reads one historical interaction at a time. Inspect history lengths in the training data, then compare validation AUC and training time for a few truncation lengths.

---

## Full Example

See [examples/ranking/run_dien.py](https://github.com/datawhalechina/torch-rechub/blob/main/examples/ranking/run_dien.py) for a complete runnable script.
