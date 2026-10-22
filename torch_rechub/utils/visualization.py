"""Model architecture diagrams backed by the optional torchview dependency."""

from copy import deepcopy
from itertools import chain
from pathlib import Path

import torch

from .model_utils import extract_feature_info, generate_dummy_input_dict

try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
    TORCHVIEW_SKIP_REASON = ""
except ImportError as e:
    TORCHVIEW_AVAILABLE = False
    TORCHVIEW_SKIP_REASON = str(e)


def visualize_model(model, input_data=None, batch_size=2, seq_length=10, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device="cpu", dpi=300, **kwargs):
    """Draw one forward pass and return a torchview ComputationGraph.

    Args:
        model: Model to draw. DataParallel wrappers are unwrapped automatically.
        input_data: Feature dict, Tensor, or tuple/list of positional inputs.
            If omitted, inputs are generated from the model's feature definitions.
        batch_size: Batch size for generated inputs.
        seq_length: Sequence length for generated inputs.
        depth: Module depth to show; -1 shows all layers.
        show_shapes: Show tensor shapes.
        expand_nested: Expand nested modules.
        save_path: PNG, SVG or PDF path. No suffix defaults to PDF.
            If omitted, display in Jupyter or open the system viewer.
        graph_name: Name of the graph.
        device: Tracing device, default CPU. Tracing on a different device uses
            a model copy; use the model's device to avoid copying large models.
        dpi: Output image resolution.
        **kwargs: Extra arguments for torchview.draw_graph.

    Example:
        >>> visualize_model(model, save_path="model.png", depth=3)
    """
    if not TORCHVIEW_AVAILABLE:
        raise ImportError("Visualization requires torchview: pip install 'torch-rechub[visualization]'. "
                          "Also install Graphviz and add its dot executable to PATH. " + TORCHVIEW_SKIP_REASON)

    model = model.module if isinstance(model, torch.nn.DataParallel) else model
    device = torch.device(device or "cpu")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    # torchview uses a nonnegative depth; keep -1 as RecHub's show-all shortcut.
    depth = float("inf") if depth == -1 else depth

    if input_data is None:
        features = extract_feature_info(model)["features"]
        if not features:
            raise ValueError("Cannot infer model features. Please provide input_data.")
        input_data = generate_dummy_input_dict(features, batch_size, seq_length, device)

    # RecHub models expect model(feature_dict), not model(**feature_dict).
    if not isinstance(input_data, (tuple, list)):
        input_data = (input_data,)

    # torchview moves the model in place. Use a copy for cross-device tracing
    # so the trainer's parameters and optimizer remain on their original device.
    devices = {tensor.device for tensor in chain(model.parameters(), model.buffers())}
    if devices and devices != {device}:
        model = deepcopy(model)

    training_modes = [(module, module.training) for module in model.modules()]
    try:
        graph = draw_graph(model, input_data=input_data, graph_name=graph_name, depth=depth, device=device, expand_nested=expand_nested, show_shapes=show_shapes, save_graph=False, **kwargs)
    finally:
        # torchview only restores the top-level mode; frozen submodules may differ.
        for module, training in training_modes:
            module.training = training

    graph.visual_graph.graph_attr["dpi"] = str(dpi)
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        graph.visual_graph.render(filename=path.stem, directory=str(path.parent), format=path.suffix.lstrip(".") or "pdf", cleanup=True)
    elif _is_jupyter_environment():
        display_graph(graph)
    else:
        graph.visual_graph.view(cleanup=True)
    return graph


def display_graph(graph, format="png"):
    """Display a ComputationGraph in Jupyter and return its graphviz graph."""
    import graphviz

    try:
        from IPython.display import display
    except ImportError:
        return graph.visual_graph

    previous_format = graphviz.set_jupyter_format(format)
    try:
        display(graph.visual_graph)
    finally:
        graphviz.set_jupyter_format(previous_format)
    return graph.visual_graph


def _is_jupyter_environment():
    """Whether an IPython kernel is available for inline display."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    shell = get_ipython()
    return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
