---
title: DeepFM Tutorial
description: "Complete DeepFM usage tutorial: from data preparation to training and evaluation"
---

# DeepFM Tutorial

## 1. Model Overview and Use Cases

DeepFM (Deep Factorization Machine), proposed by Huawei Noah's Ark Lab at IJCAI 2017, combines factorization machines (FM) with deep neural networks. It can **capture both low-order and high-order feature interactions at the same time** without requiring manual feature engineering.

**Paper**: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

### Model Architecture

<div align="center">
  <img src="/img/models/deepfm_arch.png" alt="DeepFM Model Architecture" width="600"/>
</div>

- **FM part**: captures pairwise feature interaction patterns
- **Deep part**: captures higher-order nonlinear feature interactions through multi-layer fully connected networks
- **Shared embeddings**: FM and Deep share the same bottom embedding layer, which reduces parameters

### Suitable Scenarios

- click-through rate (CTR) prediction
- ad recommendation ranking
- scenarios that need both low-order and high-order feature interactions
- a strong industry baseline and a good starting model

---

## 2. Data Preparation and Preprocessing

This example uses the **Criteo** ad click dataset. The raw data contains 13 dense features (`I1-I13`) and 26 categorical features (`C1-C26`).

### 2.1 Load Data

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature

# Load the sampled Criteo dataset
data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")
print(f"Dataset size: {data.shape[0]} rows, feature count: {data.shape[1] - 1}")
```

### 2.2 Feature Processing

```python
# Split dense and sparse features
dense_features = [f for f in data.columns if f.startswith("I")]
sparse_features = [f for f in data.columns if f.startswith("C")]

# Fill missing values
data[sparse_features] = data[sparse_features].fillna("0")
data[dense_features] = data[dense_features].fillna(0)

# Normalize dense features to [0, 1]
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# Encode categorical features with LabelEncoder
for feat in tqdm(sparse_features):
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
```

### 2.3 Define Features

```python
# Define DenseFeature and SparseFeature
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

# Extract labels
y = data["label"]
del data["label"]
x = data
```

### 2.4 Create DataLoaders

```python
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],  # train:val:test = 7:1:2
    batch_size=2048
)
```

---

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
| `fm_features` | `list[Feature]` | feature list for the FM branch, usually sparse features | all categorical features |
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
    n_epoch=50,                    # maximum number of epochs
    earlystop_patience=10,         # early stopping patience
    device="cpu",                  # use "cuda:0" for GPU training
    model_path="./saved/deepfm"    # model save path
)

# Start training
ctr_trainer.fit(train_dl, val_dl)
```

### 4.2 Training Log Explanation

During training, the trainer prints the training loss and validation AUC for each epoch:

```
epoch: 0, train loss: 0.5234
epoch: 0, val auc: 0.7156
epoch: 1, train loss: 0.4987
epoch: 1, val auc: 0.7321
...
```

---

## 5. Evaluation and Result Analysis

### 5.1 Test Set Evaluation

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 5.2 Expected Performance Reference

Typical AUC ranges on the Criteo dataset:

| Data Scale | Expected AUC |
|----------|----------|
| Sample (10k rows) | 0.70 ~ 0.75 |
| Full (45M rows) | 0.79 ~ 0.81 |

> **Note**: actual performance depends on dataset size, feature engineering, and hyperparameter settings.

---

## 6. Tuning Suggestions

### 6.1 Hyperparameter Tuning Priorities

1. **learning rate** (`lr`): the most important hyperparameter; start from `1e-3` and search over `[1e-4, 5e-4, 1e-3, 5e-3]`
2. **embedding dimension** (`embed_dim`): controls model capacity; usually `8 ~ 32`
3. **number of MLP layers and dimensions**: `[256, 128]` is a strong starting point; you can also try `[512, 256, 128]`
4. **dropout**: use `0.1 ~ 0.3`; increase it for smaller datasets

### 6.2 Practical Tuning Tips

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
- the usual practice is: **all features go to the Deep branch, sparse features go to the FM branch**
- you can also send only dense features to the Deep branch, but performance is usually worse than using all features

### Q4: GPU memory is not enough
- reduce `batch_size` (for example, `2048 -> 512`)
- reduce `embed_dim` (for example, `16 -> 8`)
- reduce MLP dimensions

---

## 8. Model Visualization

Torch-RecHub includes a model visualization tool based on `torchview`, which can generate computation graphs for the model.

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

> `visualize_model` automatically extracts feature information from the model and generates dummy inputs, so you do not need to build them by hand. It also supports custom `depth` and `batch_size`.

---

## 9. ONNX Export

Export the trained model to ONNX format for deployment with frameworks such as ONNX Runtime, TensorRT, or OpenVINO.

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

> Ranking models such as DeepFM / WideDeep / DCN are exported as full models. Two-tower models such as DSSM / YoutubeDNN also support separate export with `mode="user"` and `mode="item"`.

---

## Full Example

```python
import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tqdm import tqdm

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/deepfm", exist_ok=True)

    # 1. Load data
    data = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

    # 2. Feature preprocessing
    dense_features = [f for f in data.columns if f.startswith("I")]
    sparse_features = [f for f in data.columns if f.startswith("C")]

    data[sparse_features] = data[sparse_features].fillna("0")
    data[dense_features] = data[dense_features].fillna(0)

    scaler = MinMaxScaler()
    data[dense_features] = scaler.fit_transform(data[dense_features])

    for feat in tqdm(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    # 3. Define features
    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [
        SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
        for name in sparse_features
    ]

    y = data["label"]
    del data["label"]
    x = data

    # 4. Create DataLoaders
    dg = DataGenerator(x, y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        split_ratio=[0.7, 0.1], batch_size=2048
    )

    # 5. Create model
    model = DeepFM(
        deep_features=dense_feas + sparse_feas,
        fm_features=sparse_feas,
        mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
    )

    # 6. Train
    trainer = CTRTrainer(
        model,
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
        n_epoch=50,
        earlystop_patience=10,
        device="cpu",
        model_path="./saved/deepfm"
    )
    trainer.fit(train_dl, val_dl)

    # 7. Evaluate
    auc = trainer.evaluate(trainer.model, test_dl)
    print(f"Test AUC: {auc:.4f}")

    # 8. Visualization (optional)
    # from torch_rechub.utils.visualization import visualize_model
    # visualize_model(model, save_path="deepfm_arch.png", dpi=300)

    # 9. ONNX export (optional)
    # from torch_rechub.utils.onnx_export import ONNXExporter
    # exporter = ONNXExporter(model)
    # exporter.export("deepfm.onnx", verbose=True)


if __name__ == "__main__":
    main()
```
