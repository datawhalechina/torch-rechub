---
title: Visualization
description: Torch-RecHub model architecture visualization
---

# Visualization

Torch-RecHub provides model architecture visualization to help developers intuitively understand model structure and computation flow.

## Why torchview

Torch-RecHub uses [torchview](https://github.com/mert-kurttutan/torchview) as the visualization backend instead of other common solutions (e.g., torchviz, netron). **The core reason is: torchview is the only visualization tool that supports complex dictionary inputs**.

Recommendation model inputs are typically dictionaries containing multiple feature types:

```python
# Typical input format for recommendation models
x = {
    "user_id": tensor([1, 2, 3]),           # Sparse feature
    "age": tensor([0.5, 0.3, 0.8]),         # Dense feature
    "hist_items": tensor([[1,2,3], ...]),   # Sequence feature
}
model(x)  # Dictionary as input
```

Other visualization tools (torchviz, netron, etc.) only support simple Tensor inputs and cannot handle this dictionary-based complex input structure.

> **Tip**: If you have exported your model to ONNX format, you can also use [Netron](https://netron.app/) to view the model structure online. See [ONNX Export Documentation](/serving/onnx).

| Feature | torchview | torchviz | netron |
|---------|-----------|----------|--------|
| **Dictionary input support** | ✅ | ❌ | ❌ (requires ONNX export) |
| Forward pass tracing | ✅ | ❌ (autograd-based) | ❌ (static parsing) |
| Dynamic control flow | ✅ | ❌ | ❌ |
| Show tensor shapes | ✅ | ❌ | ✅ |
| Adjustable depth | ✅ | ❌ | ❌ |
| Nested module expansion | ✅ | ❌ | Partial |

**Other advantages**:
- **Forward tracing**: Traces forward pass via hooks, accurately capturing dynamic structures like attention mechanisms and multi-tower architectures
- **Depth control**: Flexibly control display granularity via `depth` parameter
- **Shape visualization**: Intuitively display tensor shapes at each layer

## Installation

Visualization requires additional dependencies:

```bash
pip install torch-rechub[visualization]
```

Also install system-level graphviz:

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

## Quick Start

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model

# Create model
model = DeepFM(
    deep_features=deep_features,
    fm_features=fm_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2}
)

# Visualize (auto-display in Jupyter)
graph = visualize_model(model, depth=4)

# Save as PDF
visualize_model(model, save_path="model_arch.pdf", dpi=300)
```

## Core Functions

### visualize_model

Generate computation graph visualization.

```python
from torch_rechub.utils.visualization import visualize_model

graph = visualize_model(
    model,                    # PyTorch model
    input_data=None,          # Input data (optional, auto-generated)
    batch_size=2,             # Batch size for auto-generated input
    seq_length=10,            # Sequence feature length
    depth=3,                  # Visualization depth
    show_shapes=True,         # Show tensor shapes
    expand_nested=True,       # Expand nested modules
    save_path=None,           # Save path
    graph_name="model",       # Graph name
    device="cpu",             # Device
    dpi=300,                  # Output resolution
)
```

**Parameter Description:**

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | nn.Module | PyTorch model to visualize | Required |
| `input_data` | dict | Input data dictionary, auto-generated if None | None |
| `batch_size` | int | Batch size for auto-generated input | 2 |
| `seq_length` | int | Sequence feature length | 10 |
| `depth` | int | Visualization depth, -1 shows all layers | 3 |
| `show_shapes` | bool | Show tensor shapes on edges | True |
| `expand_nested` | bool | Expand nested nn.Module | True |
| `save_path` | str | Save path, supports .pdf/.svg/.png | None |
| `dpi` | int | Output image resolution | 300 |

### display_graph

Display computation graph in Jupyter.

```python
from torch_rechub.utils.visualization import display_graph

graph = visualize_model(model, depth=4)
display_graph(graph, format='png')
```

## Usage via Trainer

Trainers also provide visualization methods:

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(model)
trainer.fit(train_dl, val_dl)

# Visualize model
trainer.visualization(save_path="model.pdf", depth=4)
```

## Usage Examples

### Ranking Model Visualization

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model
from torch_rechub.basic.features import DenseFeature, SparseFeature

# Define features
dense_features = [DenseFeature("age"), DenseFeature("income")]
sparse_features = [
    SparseFeature("city", vocab_size=100, embed_dim=16),
    SparseFeature("gender", vocab_size=3, embed_dim=8)
]

# Create model
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2}
)

# Visualize
visualize_model(model, depth=4, save_path="deepfm_arch.pdf")
```

### Retrieval Model Visualization

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.utils.visualization import visualize_model

model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64]},
    item_params={"dims": [256, 128, 64]}
)

visualize_model(model, depth=3, save_path="dssm_arch.png", dpi=300)
```

## Output Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| PDF | .pdf | Papers, reports (vector, scalable) |
| SVG | .svg | Web display (vector) |
| PNG | .png | General image format |

## FAQ

### Q: graphviz not installed error?

Make sure both Python package and system package are installed:

```bash
pip install graphviz
# Plus system-level installation (see above)
```

### Q: Image not displaying in VSCode?

Try setting output format to PNG:

```python
import graphviz
graphviz.set_jupyter_format('png')
```

### Q: How to adjust image size?

Use the `dpi` parameter to control resolution, or use the returned graph object:

```python
graph = visualize_model(model, depth=4)
graph.resize_graph(scale=1.5)
```

