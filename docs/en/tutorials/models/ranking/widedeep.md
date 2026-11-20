---
title: Wide&Deep Tutorial
description: Use the linear and deep branches of WideDeep on Criteo’s continuous and categorical fields
---

# Wide&Deep Tutorial

## 1. Model Overview and Use Cases

WideDeep computes two scores for each ad impression. This Criteo example sends continuous fields to a linear layer and categorical embeddings to a multilayer network. It combines both scores to get a click probability.

**Paper**: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

### Model Architecture

`wide_features` selects the fields for the linear layer, and `deep_features` selects the fields for the multilayer network. You can choose these groups yourself, and a field can appear in both.

---

## 2. Data Preparation and Preprocessing

Run [Prepare the Criteo inputs](/guide/quick_start#criteo-input) in the same Python session to get `dense_feas`, `sparse_feas` and the three DataLoaders. If you just trained another model, keep its data split but recreate the feature objects before constructing the new model.

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import WideDeep

model = WideDeep(
    wide_features=dense_feas,      # the Wide branch is more about "memorization", so dense or manual cross features fit well here
    deep_features=sparse_feas,     # the Deep branch is more about "generalization", so sparse categorical combinations fit well here
    mlp_params={
        "dims": [256, 128],        # hidden dimensions of the MLP
        "dropout": 0.2,
        "activation": "relu"
    }
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Suggested Value |
|------|------|------|--------|
| `wide_features` | `list[Feature]` | feature list for the Wide branch | dense features / cross features |
| `deep_features` | `list[Feature]` | feature list for the Deep branch | sparse features |
| `mlp_params.dims` | `list[int]` | hidden dimensions of each MLP layer | `[256, 128]` |
| `mlp_params.dropout` | `float` | dropout ratio | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | activation function | `"relu"` |

You can add manually crossed fields to the Wide branch. For example, encode the combination of city and time slot as a category so the linear layer can score that combination directly.

---

## 4. Training Process and Code Example

```python
import os
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/widedeep", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=2,
    earlystop_patience=10,
    device="cpu",                   # GPU: "cuda:0"
    model_path="./saved/widedeep"
)

trainer.fit(train_dl, val_dl)
```

---

## 5. Evaluation and Result Analysis

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

The bundled Criteo sample contains just 115 rows. Use it to check that training and evaluation run. To compare click prediction quality, use a larger dataset and the same training, validation and test splits for every model. [Benchmarks](/tools/benchmark)

## 6. Tuning Suggestions

To test a new field, keep the training and validation data fixed. Add only that field to `wide_features` or `deep_features`, then compare validation AUC.

## 7. FAQ and Troubleshooting

### When should I try DeepFM?

The Wide branch scores the fields you supply. To learn pairwise interactions between categorical fields automatically, try DeepFM on the same data and compare validation AUC and training time. [DeepFM tutorial](/tutorials/models/ranking/deepfm)

---

## Run the example script

```bash
cd examples/ranking
uv run python run_criteo.py --model_name widedeep --epoch 1 --device cpu
```
