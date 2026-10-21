---
title: Visualization
description: Torch-RecHub model architecture visualization
---

# Visualization

Torch-RecHub provides model architecture visualization to help developers intuitively understand model structure and computation flow.

## Why torchview

Torch-RecHub uses [torchview](https://github.com/mert-kurttutan/torchview) as its visualization backend. The project's wrapper passes the feature dictionary to the model as one positional argument, so it can trace the dictionary inputs commonly used by recommendation models.

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

torchviz is based on the autograd graph, while Netron primarily inspects an already exported static model. These tools serve different purposes; the distinction cannot be reduced to whether they "support dictionaries."

> **Tip**: If you have exported your model to ONNX format, you can also use [Netron](https://netron.app/) to view the model structure online. See [ONNX Export Documentation](/serving/onnx).

| Feature | torchview | torchviz | netron |
|---------|-----------|----------|--------|
| **This project's dictionary-input wrapper** | ✅ | Requires a custom wrapper | Requires model export first |
| Forward pass tracing | ✅ | ❌ (autograd-based) | ❌ (static parsing) |
| Shows one actually executed path | ✅ | ✅ | ❌ (static parsing) |
| Show tensor shapes | ✅ | ❌ | ✅ |
| Adjustable depth | ✅ | ❌ | ❌ |
| Nested module expansion | ✅ | ❌ | Partial |

**Other advantages**:
- **Forward tracing**: Captures the modules traversed by this input during one forward pass; control-flow branches that are not executed do not appear in the graph
- **Depth control**: Flexibly control display granularity via `depth` parameter
- **Shape visualization**: Intuitively display tensor shapes at each layer

## Installation

Visualization requires additional dependencies:

```bash
pip install "torch-rechub[visualization]"
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

Generate computation graph visualization. CTR, Match, MTL and Seq trainers share this drawing function; their training calls are unchanged.

Drawing preserves each submodule's training/evaluation mode, including when tracing fails. CPU tracing remains the default. A copy is used when tracing on another device, leaving the trainer's model untouched. For large models, pass their current device (for example, `device="cuda:0"`) to avoid a model copy.

The diagram shows the forward path taken by these inputs, not every possible branch. Drawing does not train the model or run backpropagation.

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
| `input_data` | dict / Tensor / tuple / list | Dict as one argument; tuple/list as positional arguments; None generates feature inputs | None |
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

# Get the computation graph
graph = visualize_model(model, depth=4)

# Display it in Jupyter
display_graph(graph, format='png');
```

The trailing semicolon prevents Jupyter from displaying the returned Graphviz object a second time. Usually, `graph = trainer.visualization()` already displays the diagram; no extra `display_graph()` call is needed.

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
    # Sparse features used in FM interactions must share the same embedding dimension
    SparseFeature("gender", vocab_size=3, embed_dim=16)
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

### Visualization via Trainer

Trainers also provide a visualization method:

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(model)
trainer.fit(train_dl, val_dl)

# Visualize model
trainer.visualization(save_path="model.pdf", depth=4)
```

The sequence trainer keeps its original argument order and generates non-padding tokens with zero time differences, for drawing only:

```python
seq_trainer.visualization(seq_length=20, save_path="sequence.svg")
```

For a custom model without recognizable feature definitions, provide its actual inputs:

```python
visualize_model(model, input_data=(tokens, time_diffs), save_path="custom.png")
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

