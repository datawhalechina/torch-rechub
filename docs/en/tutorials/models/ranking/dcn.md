---
title: DCN / DCNv2 Tutorial
description: Compare the cross layers of DCN and DCNv2 using the same Criteo features
---

# DCN / DCNv2 Tutorial

## 1. Model Overview and Use Cases

DCN learns combinations of input fields in its cross layers while a multilayer network processes the same fields. Two fields can each contribute to the click score individually and have a different effect when they occur together. Train DCN or DCNv2 on the same Criteo sample below.

**Papers**:
- DCN: [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
- DCNv2: [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)

### Key Differences

Each DCN cross layer uses a weight vector. Torch-RecHub's DCNv2 defaults to a mixture of low-rank experts. Set `use_low_rank_mixture=False` to use a full matrix in each cross layer. Both models accept the same input features.

---

## 2. Data Preparation and Preprocessing

Run [Prepare the Criteo inputs](/guide/quick_start#criteo-input) in the same Python session to get `dense_feas`, `sparse_feas` and the three DataLoaders. If you just trained another model, keep its data split but recreate the feature objects before constructing the new model.

## 3. Model Configuration and Parameter Notes

Choose one model to construct and train. Replace `DCN` with `DCNv2` to try the other cross layer. For an independent comparison, rerun the data preparation so each model starts with its own embedding weights.

```python
from torch_rechub.models.ranking import DCN, DCNv2

model = DCN(
    features=dense_feas + sparse_feas,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128]},
)
```

### Model parameters

| Parameter | Type | Description | Example setting |
|------|------|------|--------|
| `features` | `list[Feature]` | All dense and sparse input features | All features |
| `n_cross_layers` | `int` | Number of cross-network layers | 2–4 |
| `mlp_params.dims` | `list[int]` | Hidden widths in the deep network | `[256, 128]` |
| `mlp_params.dropout` | `float` | Dropout rate | 0.1–0.3 |

---

## 4. Training Process and Code Example

```python
import os
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/dcn", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=2,
    earlystop_patience=10,
    device="cpu",
    model_path="./saved/dcn"
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

The example starts with three cross layers. To see whether depth helps, keep the split and MLP widths fixed, change only `n_cross_layers`, and compare validation AUC. For DCNv2, also record whether you used low-rank experts, since that choice affects the parameter count.

## 7. FAQ and Troubleshooting

### Do I need to create cross features myself?

Pass the original Criteo fields to the model. The cross layers learn combinations during training. If you have a manually crossed field, compare validation results on the same split with and without that extra column.

---

## Run the example script

```bash
cd examples/ranking
uv run python run_criteo.py --model_name dcn --epoch 1 --device cpu
```
