---
title: Matching Models
description: Torch-RecHub matching/retrieval models
---

# Matching Models

Matching (retrieval) models fetch a candidate set from large catalogs. Torch-RecHub offers two-tower and sequence-based retrieval models for diverse scenarios.

## Model Overview (what to pick)

| Model              | When to use | Highlights |
| ---                | ---         | --- |
| DSSM               | Classic text/ID matching | Two-tower semantic matching; cosine/dot similarity |
| YoutubeDNN         | Large-scale retrieval    | Sequence-based user encoder + item tower |
| MIND               | Multi-interest users     | Capsule routing to learn multiple user interests |
| GRU4Rec / SASRec   | Sequential retrieval     | RNN / Transformer for recent behavior |
| ComirecDR / ComirecSA | Controllable interests | Disentangled or self-attentive multi-interest |

## Quick Usage (example: DSSM)

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.basic.features import SparseFeature, DenseFeature

user_features = [
    SparseFeature("user_id", vocab_size=10000, embed_dim=32),
    DenseFeature("age", embed_dim=1),
]
item_features = [
    SparseFeature("item_id", vocab_size=100000, embed_dim=32),
    SparseFeature("category", vocab_size=1000, embed_dim=16),
]

model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"},
)

trainer = MatchTrainer(
    model=model,
    mode=0,  # 0: point-wise, 1: pair-wise (BPR), 2: list-wise
    optimizer_params={"lr": 1e-3},
    n_epoch=10,
)
trainer.fit(train_dl)
```

### Export towers (two-tower serving)

```python
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

## Parameter Notes (common fields)

- `mode`: training objective (point / pair / list).
- `user_features` / `item_features`: feature schemas for user/item tower.
- `temperature`: softmax temperature for contrastive losses.
- `user_params` / `item_params`: MLP configs for each tower (`dims`, `dropout`, `activation`).

## Tips

- Multi-interest heavy users → MIND or Comirec family.  
- Need strong sequence signals → GRU4Rec (RNN) or SASRec (Transformer).  
- Large-scale production two-tower → YoutubeDNN / DSSM.  

