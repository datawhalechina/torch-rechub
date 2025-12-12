---
title: Ranking Models
description: Torch-RecHub ranking models
---

# Ranking Models

Ranking models predict CTR or preference scores to re-rank retrieved candidates. Torch-RecHub includes classic and modern architectures covering feature crosses, attention, and sequence modeling.

## Model Overview (what to pick)

| Model        | When to use | Highlights |
| ---          | ---         | --- |
| WideDeep     | Baseline, memory + generalization | Linear (Wide) + DNN (Deep) joint training |
| DeepFM       | Feature interactions matter | FM for 2nd-order + DNN for higher-order, shared embeddings |
| DCN / DCNv2  | Explicit feature crosses | Efficient cross network; v2 adds feature selection & scaling |
| EDCN         | Stronger crosses            | Enhanced cross + deep with importance weights |
| AFM          | Need interaction importance | Attention-weighted FM interactions |
| FiBiNET      | Uneven feature importance   | SE-style importance + bilinear interactions |
| DeepFFM / FatDeepFFM | Field-aware interactions | FFM + deep network for higher-order field-aware crosses |
| BST          | Sequence-focused CTR        | Transformer over behavior sequences |
| DIN          | Dynamic interest            | Attention over user history for target-aware interest |
| DIEN         | Evolving interest           | GRU + attention with interest evolution |
| AutoInt      | Automatic interaction       | Multi-head self-attention for feature crosses |

## Quick Usage (example: DeepFM)

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.features import SparseFeature, DenseFeature

sparse_features = [
    SparseFeature("user_id", vocab_size=10000, embed_dim=32),
    SparseFeature("item_id", vocab_size=50000, embed_dim=32),
]
dense_features = [DenseFeature("age", embed_dim=1)]

model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=10,
)
trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
```

## Parameter Notes (common fields)

- `deep_features`: feature list for deep/MLP branches.
- `fm_features` / `cross_features`: feature list for FM / cross layers.
- `mlp_params`: dict with `dims`, `dropout`, `activation`.
- `cross_num_layers`: number of cross layers (DCN/DCNv2/EDCN).
- `attention_params`: attention dim/dropout (AFM).

## Tips

- Need explicit feature crosses → DCN/DCNv2/EDCN.  
- Need interpretable interaction weights → AFM / FiBiNET.  
- Strong field-aware interactions → DeepFFM / FatDeepFFM.  
- User behavior sequences → DIN (target-aware), DIEN (evolving), BST (Transformer).  
- Fast baseline with good coverage → WideDeep or DeepFM.  

