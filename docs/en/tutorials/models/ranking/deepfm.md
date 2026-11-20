---
title: DeepFM Tutorial
description: Predict ad clicks with DeepFM and check FM embedding dimensions and validation AUC
---

# DeepFM Tutorial

## 1. Model Overview and Use Cases

DeepFM predicts a click probability for each ad impression. This example passes Criteo's categorical fields to both an FM and a multilayer network. The FM scores pairwise interactions, while the multilayer network can combine the categorical and continuous fields in more complex ways.

**Paper**: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

### Model Architecture

<div align="center">
  <img src="/img/models/deepfm_arch.png" alt="DeepFM Model Architecture" width="600"/>
</div>

When a categorical field enters both the FM and the deep network, they look up its values in the same embedding table. The model adds the linear, FM and deep scores, then applies a sigmoid to get a click probability.

---

## 2. Data Preparation and Preprocessing

Run [Prepare the Criteo inputs](/guide/quick_start#criteo-input) in the same Python session to get `dense_feas`, `sparse_feas` and the three DataLoaders. If you just trained another model, keep its data split but recreate the feature objects before constructing the new model.

## 3. Model Configuration and Parameter Notes

### 3.1 Create the Model

```python
from torch_rechub.models.ranking import DeepFM

model = DeepFM(
    deep_features=dense_feas + sparse_feas,  # the Deep branch learns higher-order nonlinear relationships
    fm_features=sparse_feas,                 # the FM branch performs second-order interactions on sparse features
    mlp_params={
        "dims": [256, 128],                  # hidden layer dimensions of the MLP
        "dropout": 0.2,                      # dropout ratio
        "activation": "relu"                 # activation function
    }
)
```

### 3.2 Parameter Details

| Parameter | Type | Description | Suggested Value |
|------|------|------|--------|
| `deep_features` | `list[Feature]` | feature list for the Deep branch, usually all features | dense + sparse features |
| `fm_features` | `list[Feature]` | Features used by the FM branch, usually `SparseFeature` instances | All categorical features |
| `mlp_params.dims` | `list[int]` | hidden dimensions of each MLP layer | `[256, 128]` or `[256, 128, 64]` |
| `mlp_params.dropout` | `float` | dropout ratio to reduce overfitting | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | activation function (`relu`, `prelu`, `sigmoid`) | `"relu"` |

---

## 4. Training Process and Code Example

### 4.1 Create the Trainer and Train

```python
import os
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/deepfm", exist_ok=True)

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={
        "lr": 1e-3,                # learning rate
        "weight_decay": 1e-3       # L2 regularization
    },
    n_epoch=2,                    # Maximum number of epochs.
    earlystop_patience=10,         # early stopping patience
    device="cpu",                  # use "cuda:0" for GPU training
    model_path="./saved/deepfm"    # model save path
)

# Start training
ctr_trainer.fit(train_dl, val_dl)
```

### 4.2 Training Log Explanation

`fit` receives a validation set and computes validation AUC after each epoch. This example trains for at most two epochs and tests the weights from the epoch with the highest validation AUC. The bundled sample is small, so the score can change with the random seed.

---

## 5. Evaluation and Result Analysis

### 5.1 Test Set Evaluation

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

The bundled Criteo sample contains just 115 rows. Use it to check that training and evaluation run. To compare click prediction quality, use a larger dataset and the same training, validation and test splits for every model. [Benchmarks](/tools/benchmark)

## 6. Tuning Suggestions

### 6.1 Hyperparameter Tuning Priorities

Start with the example learning rate of `1e-3` and change one setting at a time. When changing `embed_dim`, give all categorical fields in the FM the same width so their embeddings can be stacked to calculate pairwise interactions.

### 6.2 Practical Tuning Tips

This optional setup uses CUDA and multiplies the learning rate by 0.8 every two epochs. To try it, replace the trainer configuration in section 4.

```python
import os

# Use a learning-rate scheduler
os.makedirs("./saved/deepfm", exist_ok=True)

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=100,
    earlystop_patience=10,
    device="cuda:0",
    model_path="./saved/deepfm",
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 2, "gamma": 0.8}  # decay every 2 epochs
)
```

---

## 7. FAQ and Troubleshooting

### Q1: The training loss does not go down
- check whether the learning rate is too large or too small
- make sure data preprocessing is correct, especially missing value handling and normalization
- check whether the label distribution is extremely imbalanced

### Q2: Overfitting (high training AUC, low validation AUC)
- increase dropout (for example, `0.2 -> 0.5`)
- increase `weight_decay` (for example, `1e-3 -> 1e-2`)
- reduce the MLP depth or width
- use more training data

### Q3: How should I choose `deep_features` and `fm_features`?
This example passes continuous and categorical fields to `deep_features`, and the categorical fields whose pairwise interactions matter to `fm_features`. Both lists may reference the same categorical feature object.

### Q4: GPU memory is not enough
- reduce `batch_size` (for example, `2048 -> 512`)
- reduce `embed_dim` (for example, `16 -> 8`)
- reduce MLP dimensions

---

## 8. Model Visualization

Draw the model to see which inputs feed the FM and which feed the deep network.

### Install dependencies

```bash
pip install torch-rechub[visualization]
# System-level dependency:
# Ubuntu: sudo apt-get install graphviz
# macOS: brew install graphviz
# Windows: choco install graphviz
```

### Visualize a DeepFM model

```python
from torch_rechub.utils.visualization import visualize_model

# Automatically generate dummy inputs and display inline in Jupyter
graph = visualize_model(model, depth=4)

# Save as a high-resolution PNG
visualize_model(model, save_path="deepfm_architecture.png", dpi=300)

# Save as a PDF
visualize_model(model, save_path="deepfm_architecture.pdf")
```

### DeepFM Architecture

![DeepFM Model Architecture](/img/models/deepfm_arch.png)

`visualize_model` builds a batch from the feature definitions. If the graph looks crowded, lower `depth` to see how the main modules connect.

---

## 9. ONNX Export

Use ONNX Runtime to load the export and check the output shape. The all-zero inputs below only check the interface. To compare predictions for real samples, encode the inputs exactly as you did during training. [Serving example](/serving/demo)

### Export the model

```python
from torch_rechub.utils.onnx_export import ONNXExporter

exporter = ONNXExporter(model, device="cpu")

# Export the DeepFM model
exporter.export("deepfm.onnx", verbose=True)

# Inspect model input info
info = exporter.get_input_info()
print(info)
```

### Run inference with ONNX Runtime

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("deepfm.onnx")

# Inspect model inputs
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

# Build inputs and run inference
input_feed = {}
for inp in session.get_inputs():
    if "int" in inp.type.lower():
        input_feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.int64)
    else:
        input_feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)

output = session.run(None, input_feed)
print(f"Output shape: {output[0].shape}")
```

> DeepFM, WideDeep, and DCN export to ONNX as complete ranking models. For two-tower models such as DSSM and YoutubeDNN, use `mode="user"` and `mode="item"` to export each tower separately.

---

## Run the example script

```bash
cd examples/ranking
uv run python run_criteo.py --model_name deepfm --epoch 1 --device cpu
```
