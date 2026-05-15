"""
Model Visualization Utilities for Torch-RecHub.

This module provides model structure visualization using torchview library.
Requires optional dependencies: pip install torch-rechub[visualization]

Example:
    >>> from torch_rechub.utils.visualization import visualize_model, display_graph
    >>> graph = visualize_model(model, depth=4)
    >>> display_graph(graph)  # Display in Jupyter Notebook

    >>> # Save to file
    >>> visualize_model(model, save_path="model_arch.pdf")
"""

from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

# Check for optional dependencies
TORCHVIEW_AVAILABLE = False
TORCHVIEW_SKIP_REASON = "torchview not installed"

try:
    from torchview import draw_graph
    TORCHVIEW_AVAILABLE = True
except ImportError as e:
    TORCHVIEW_SKIP_REASON = f"torchview not available: {e}"


def _is_jupyter_environment() -> bool:
    """Check if running in Jupyter/IPython environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        # Check for Jupyter notebook or qtconsole
        shell_class = shell.__class__.__name__
        return shell_class in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except (ImportError, NameError):
        return False


def display_graph(graph: Any, format: str = 'png') -> Any:
    """Display a torchview ComputationGraph in Jupyter.

    Parameters
    ----------
    graph : ComputationGraph
        Returned by :func:`visualize_model`.
    format : str, default='png'
        Output format; 'png' recommended for VSCode.

    Returns
    -------
    graphviz.Digraph or None
        Displayed graph object, or None if display fails.
    """
    if not TORCHVIEW_AVAILABLE:
        raise ImportError(f"Visualization requires torchview. {TORCHVIEW_SKIP_REASON}\n"
                          "Install with: pip install torch-rechub[visualization]")

    try:
        import graphviz

        # Set format for VSCode compatibility
        graphviz.set_jupyter_format(format)
    except ImportError:
        pass

    # Get the visual_graph (graphviz.Digraph object)
    visual = graph.visual_graph

    # Try to use IPython display for explicit rendering
    try:
        from IPython.display import display
        display(visual)
        return visual
    except ImportError:
        # Not in IPython/Jupyter environment, return the graph
        return visual


def visualize_model(
    model: nn.Module,
    input_data: Optional[Dict[str,
                              torch.Tensor]] = None,
    batch_size: int = 2,
    seq_length: int = 10,
    depth: int = 3,
    show_shapes: bool = True,
    expand_nested: bool = True,
    save_path: Optional[str] = None,
    graph_name: str = "model",
    device: str = "cpu",
    dpi: int = 300,
    **kwargs
) -> Any:
    """Visualize a Torch-RecHub model's computation graph.

    This function generates a visual representation of the model architecture,
    showing layer connections, tensor shapes, and nested module structures.
    It automatically extracts feature information from the model to generate
    appropriate dummy inputs.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to visualize. Should be a Torch-RecHub model
        with feature attributes (e.g., DeepFM, DSSM, MMOE).
    input_data : dict, optional
        Dict of example inputs {feature_name: tensor}.
        If None, inputs are auto-generated based on model features.
    batch_size : int, default=2
        Batch size for auto-generated inputs.
    seq_length : int, default=10
        Sequence length for SequenceFeature inputs.
    depth : int, default=3
        Visualization depth - higher values show more detail.
        Set to -1 to show all layers.
    show_shapes : bool, default=True
        Whether to display tensor shapes on edges.
    expand_nested : bool, default=True
        Whether to expand nested nn.Module with dashed borders.
    save_path : str, optional
        Path to save the graph image. Supports .pdf, .svg, .png formats.
        If None, displays in Jupyter or opens system viewer.
    graph_name : str, default="model"
        Name for the computation graph.
    device : str, default="cpu"
        Device for model execution during tracing.
    dpi : int, default=300
        Resolution in dots per inch for output image.
        Higher values produce sharper images suitable for papers.
    **kwargs : dict
        Additional arguments passed to torchview.draw_graph().

    Returns
    -------
    ComputationGraph
        A torchview ComputationGraph object.
        - Use `.visual_graph` property to get the graphviz.Digraph
        - Use `.resize_graph(scale=1.5)` to adjust graph size

    Raises
    ------
    ImportError
        If torchview or graphviz is not installed.
    ValueError
        If model has no recognizable feature attributes.

    Notes
    -----
    Default Display Behavior:
        When `save_path` is None (default):
        - In Jupyter/IPython: automatically displays the graph inline
        - In Python script: opens the graph with system default viewer

    Requires graphviz system package: apt/brew/choco install graphviz.
    For Jupyter display issues, try: graphviz.set_jupyter_format('png').

    Examples
    --------
    >>> from torch_rechub.models.ranking import DeepFM
    >>> from torch_rechub.utils.visualization import visualize_model
    >>>
    >>> # Auto-display in Jupyter or open in viewer
    >>> visualize_model(model, depth=4)  # No save_path needed
    >>>
    >>> # Save to high-DPI PNG for paper
    >>> visualize_model(model, save_path="model.png", dpi=300)
    """
    if not TORCHVIEW_AVAILABLE:
        raise ImportError(
            f"Visualization requires torchview. {TORCHVIEW_SKIP_REASON}\n"
            "Install with: pip install torch-rechub[visualization]\n"
            "Also ensure graphviz is installed on your system:\n"
            "  - Ubuntu/Debian: sudo apt-get install graphviz\n"
            "  - macOS: brew install graphviz\n"
            "  - Windows: choco install graphviz"
        )

    # Import feature extraction utilities from model_utils
    from .model_utils import extract_feature_info, generate_dummy_input_dict

    model.eval()
    model.to(device)

    # Auto-generate input data if not provided
    if input_data is None:
        feature_info = extract_feature_info(model)
        features = feature_info['features']

        if not features:
            raise ValueError("Could not extract feature information from model. "
                             "Please provide input_data parameter manually.")

        # Generate dummy input dict
        input_data = generate_dummy_input_dict(features, batch_size=batch_size, seq_length=seq_length, device=device)
    else:
        # Ensure input tensors are on correct device
        input_data = {k: v.to(device) for k, v in input_data.items()}

    # IMPORTANT: Wrap input_data dict in a tuple to work around torchview's behavior
    #
    # torchview's forward_prop function checks the type of input_data:
    #   - If isinstance(x, (list, tuple)): model(*x)
    #   - If isinstance(x, Mapping): model(**x)  <- This unpacks dict as kwargs!
    #   - Else: model(x)
    #
    # torch-rechub models expect forward(self, x) where x is a complete dict.
    # By wrapping the dict in a tuple, torchview will call:
    #   model(*(input_dict,)) = model(input_dict)
    # which is exactly what our models expect.
    input_data_wrapped = (input_data,)

    # Call torchview.draw_graph without saving (we'll save manually with DPI)
    graph = draw_graph(
        model,
        input_data=input_data_wrapped,
        graph_name=graph_name,
        depth=depth,
        device=device,
        expand_nested=expand_nested,
        show_shapes=show_shapes,
        save_graph=False,  # Don't save here, we'll save manually with DPI
        **kwargs
    )

    # Set DPI for high-quality output (must be set BEFORE rendering/saving)
    graph.visual_graph.graph_attr['dpi'] = str(dpi)

    # Handle save_path: manually save with DPI applied
    if save_path:
        import os
        directory = os.path.dirname(save_path) or "."
        filename = os.path.splitext(os.path.basename(save_path))[0]
        ext = os.path.splitext(save_path)[1].lstrip('.')
        # Default to pdf if no extension
        output_format = ext if ext else 'pdf'
        # Create directory if it doesn't exist
        if directory != "." and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        # Render and save with DPI applied
        graph.visual_graph.render(
            filename=filename,
            directory=directory,
            format=output_format,
            cleanup=True  # Remove intermediate .gv file
        )

    # Handle default display behavior when save_path is None
    if save_path is None:
        if _is_jupyter_environment():
            # In Jupyter: display inline
            display_graph(graph)
        else:
            # In script: open with system viewer
            graph.visual_graph.view(cleanup=True)

    return graph
