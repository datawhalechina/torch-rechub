---
title: Model diagrams
description: Inspect the shapes of model inputs and outputs in Jupyter or save the graph to a file
---

# Model diagrams

To see where DeepFM concatenates features or whether the two DSSM towers produce vectors of the same width, draw the model before training. By default, `visualize_model` creates a small batch from the model's feature definitions and traces one forward pass with torchview. Follow the edges to see the input and output shapes of each layer.

## Install the dependencies

From a source checkout, install the visualization extra.

```bash
uv sync --extra visualization
```

For a PyPI installation, run `pip install "torch-rechub[visualization]"`.

Rendering also requires Graphviz's `dot` executable. Use the installation command for your operating system.

```bash
# Ubuntu/Debian
sudo apt-get install graphviz

# macOS
brew install graphviz

# Windows
choco install graphviz
```

## Quick start

This DeepFM example defines two continuous and two categorical features. Its FM branch computes interactions between categorical embeddings, so both embeddings have width 16.

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model
from torch_rechub.basic.features import DenseFeature, SparseFeature

# Define the features.
dense_features = [DenseFeature("age"), DenseFeature("income")]
sparse_features = [
    SparseFeature("city", vocab_size=100, embed_dim=16),
    # Use the same embedding width for the features in the FM interactions.
    SparseFeature("gender", vocab_size=3, embed_dim=16)
]

# Create the model.
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2}
)
```

In Jupyter, run this line to display the model graph in the current cell.

```python
graph = visualize_model(model, depth=4)
```

In a terminal, set `save_path` to save the graph and then open the PDF.

```python
graph = visualize_model(model, depth=4, save_path="deepfm_arch.pdf")
```

## Drawing a graph

### visualize_model

The graph shows the layers the current input passes through. To trace another branch, draw it again with a different `input_data`. Increase `depth` to expand more submodules.

By default, tracing runs on the CPU. If the model is on a GPU, the function uses a CPU copy and leaves the original on its GPU. For a large model, pass its current device, such as `device="cuda:0"`, to avoid the copy. Each layer returns to its original training or evaluation mode after tracing.

```python
from torch_rechub.utils.visualization import visualize_model

graph = visualize_model(
    model,                    # PyTorch model
    input_data=None,          # Optional inputs, inferred by default
    batch_size=2,             # Batch size for generated inputs
    seq_length=10,            # Length of generated sequence features
    depth=3,                  # Number of module levels to display
    show_shapes=True,         # Show tensor shapes
    expand_nested=True,       # Expand nested modules
    save_path=None,           # Output path
    graph_name="model",       # Graph name
    device="cpu",             # Device for tracing
    dpi=300,                  # Output resolution
)
```

**Parameters**

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `model` | nn.Module | PyTorch model to draw | Required |
| `input_data` | dict / Tensor / tuple / list | A dictionary is passed as one argument. Tuples and lists supply positional arguments in order. `None` generates inputs. | None |
| `batch_size` | int | Batch size for generated inputs | 2 |
| `seq_length` | int | Length of generated sequence features | 10 |
| `depth` | int | Module depth to show. Use -1 to show all layers. | 3 |
| `show_shapes` | bool | Show tensor shapes on edges | True |
| `expand_nested` | bool | Expand nested nn.Module objects | True |
| `save_path` | str | Output path ending in .pdf, .svg or .png | None |
| `dpi` | int | Output resolution | 300 |

### Change the Jupyter display format

The call above displays PNG by default. If you already have `graph`, you can display the same graph as SVG in another cell.

```python
from torch_rechub.utils.visualization import display_graph

display_graph(graph, format="svg");
```

## Draw from a trainer

In the DSSM tutorial, you can run the following line as soon as you have created `trainer = MatchTrainer(...)`, even before training. `CTRTrainer` and `MTLTrainer` accept the same call. [DSSM tutorial](/tutorials/models/matching/dssm)

```python
trainer.visualization(save_path="dssm_arch.png", depth=3)
```

For `SeqTrainer`, set `seq_length` to choose the length of the sequence used for the diagram.

```python
seq_trainer.visualization(seq_length=20, save_path="sequence.svg")
```

If a custom model's `forward` takes `tokens` and `time_diffs` as separate arguments, pass the two tensors in a tuple.

```python
visualize_model(model, input_data=(tokens, time_diffs), save_path="custom.png")
```

## Output formats

| Format | Extension | When to use it |
| --- | --- | --- |
| PDF | .pdf | Papers and reports where the diagram needs to scale |
| SVG | .svg | Scalable diagrams on the web |
| PNG | .png | General-purpose raster images |

## Common questions

### Why does rendering fail even though the Python package is installed?

The Python package describes the graph. Graphviz's `dot` executable renders it. Run `dot -V` in the terminal where you launch Jupyter. If the command is missing, install Graphviz, add its `bin` directory to `PATH`, then restart the terminal and Jupyter.

### How can I change the image size?

Use `dpi` to set the resolution when saving a PNG. For a figure that needs to scale in a report, use an `.svg` or `.pdf` extension in `save_path`.
