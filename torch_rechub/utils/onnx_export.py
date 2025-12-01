"""
ONNX Export Utilities for Torch-RecHub models.

This module provides non-invasive ONNX export functionality for recommendation models.
It uses reflection to extract feature information from models and wraps dict-input models
to be compatible with ONNX's positional argument requirements.

Date: 2024
References:
    - PyTorch ONNX Export: https://pytorch.org/docs/stable/onnx.html
    - ONNX Runtime: https://onnxruntime.ai/docs/
Authors: Torch-RecHub Contributors
"""

import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..basic.features import DenseFeature, SequenceFeature, SparseFeature


class ONNXWrapper(nn.Module):
    """Wraps a dict-input model to accept positional arguments for ONNX compatibility.

    ONNX does not support dict as input, so this wrapper converts positional arguments
    back to dict format before passing to the original model.

    Args:
        model: The original PyTorch model that accepts dict input.
        input_names: Ordered list of feature names corresponding to input positions.
        mode: Optional mode for dual-tower models ("user" or "item").

    Example:
        >>> wrapper = ONNXWrapper(dssm_model, ["user_id", "movie_id", "hist_movie_id"])
        >>> # Now can call: wrapper(user_id_tensor, movie_id_tensor, hist_tensor)
    """

    def __init__(self, model: nn.Module, input_names: List[str], mode: Optional[str] = None):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self._original_mode = getattr(model, 'mode', None)

        # Set mode for dual-tower models
        if mode is not None and hasattr(model, 'mode'):
            model.mode = mode

    def forward(self, *args) -> torch.Tensor:
        """Convert positional args to dict and call original model."""
        if len(args) != len(self.input_names):
            raise ValueError(f"Expected {len(self.input_names)} inputs, got {len(args)}. "
                             f"Expected names: {self.input_names}")
        x_dict = {name: arg for name, arg in zip(self.input_names, args)}
        return self.model(x_dict)

    def restore_mode(self):
        """Restore the original mode of the model."""
        if hasattr(self.model, 'mode'):
            self.model.mode = self._original_mode


def extract_feature_info(model: nn.Module) -> Dict[str, Any]:
    """Extract feature information from a model using reflection.

    This function inspects model attributes to find feature lists without
    modifying the model code. Supports various model architectures.

    Args:
        model: The recommendation model to inspect.

    Returns:
        Dict containing:
            - 'features': List of unique Feature objects
            - 'input_names': List of feature names in order
            - 'input_types': Dict mapping feature name to feature type
            - 'user_features': List of user-side features (for dual-tower models)
            - 'item_features': List of item-side features (for dual-tower models)
    """
    # Common feature attribute names across different model types
    feature_attrs = [
        'features',           # MMOE, DCN, etc.
        'deep_features',      # DeepFM, WideDeep
        'fm_features',        # DeepFM
        'wide_features',      # WideDeep
        'linear_features',    # DeepFFM
        'cross_features',     # DeepFFM
        'user_features',      # DSSM, YoutubeDNN, MIND
        'item_features',      # DSSM, YoutubeDNN, MIND
        'history_features',   # DIN, MIND
        'target_features',    # DIN
        'neg_item_feature',   # YoutubeDNN, MIND
    ]

    all_features = []
    user_features = []
    item_features = []

    for attr in feature_attrs:
        if hasattr(model, attr):
            feat_list = getattr(model, attr)
            if isinstance(feat_list, list) and len(feat_list) > 0:
                all_features.extend(feat_list)
                # Track user/item features for dual-tower models
                if 'user' in attr or 'history' in attr:
                    user_features.extend(feat_list)
                elif 'item' in attr:
                    item_features.extend(feat_list)

    # Deduplicate features by name while preserving order
    seen = set()
    unique_features = []
    for f in all_features:
        if hasattr(f, 'name') and f.name not in seen:
            seen.add(f.name)
            unique_features.append(f)

    # Deduplicate user/item features
    seen_user = set()
    unique_user = [f for f in user_features if hasattr(f, 'name') and f.name not in seen_user and not seen_user.add(f.name)]
    seen_item = set()
    unique_item = [f for f in item_features if hasattr(f, 'name') and f.name not in seen_item and not seen_item.add(f.name)]

    return {
        'features': unique_features,
        'input_names': [f.name for f in unique_features],
        'input_types': {
            f.name: type(f).__name__ for f in unique_features
        },
        'user_features': unique_user,
        'item_features': unique_item,
    }


def generate_dummy_input(features: List[Any], batch_size: int = 2, seq_length: int = 10, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
    """Generate dummy input tensors for ONNX export based on feature definitions.

    Args:
        features: List of Feature objects (SparseFeature, DenseFeature, SequenceFeature).
        batch_size: Batch size for dummy input (default: 2).
        seq_length: Sequence length for SequenceFeature (default: 10).
        device: Device to create tensors on (default: 'cpu').

    Returns:
        Tuple of tensors in the order of input features.

    Example:
        >>> features = [SparseFeature("user_id", 1000), SequenceFeature("hist", 500)]
        >>> dummy = generate_dummy_input(features, batch_size=4)
        >>> # Returns (user_id_tensor[4], hist_tensor[4, 10])
    """
    inputs = []
    for feat in features:
        if isinstance(feat, SequenceFeature):
            # Sequence features have shape [batch_size, seq_length]
            tensor = torch.randint(0, feat.vocab_size, (batch_size, seq_length), device=device)
        elif isinstance(feat, SparseFeature):
            # Sparse features have shape [batch_size]
            tensor = torch.randint(0, feat.vocab_size, (batch_size,), device=device)
        elif isinstance(feat, DenseFeature):
            # Dense features have shape [batch_size, embed_dim]
            tensor = torch.randn(batch_size, feat.embed_dim, device=device)
        else:
            raise TypeError(f"Unsupported feature type: {type(feat)}")
        inputs.append(tensor)
    return tuple(inputs)


def generate_dynamic_axes(input_names: List[str], output_names: List[str] = None, batch_dim: int = 0, include_seq_dim: bool = True, seq_features: List[str] = None) -> Dict[str, Dict[int, str]]:
    """Generate dynamic axes configuration for ONNX export.

    Args:
        input_names: List of input tensor names.
        output_names: List of output tensor names (default: ["output"]).
        batch_dim: Dimension index for batch size (default: 0).
        include_seq_dim: Whether to include sequence dimension as dynamic (default: True).
        seq_features: List of feature names that are sequences (default: auto-detect).

    Returns:
        Dynamic axes dict for torch.onnx.export.
    """
    if output_names is None:
        output_names = ["output"]

    dynamic_axes = {}

    # Input axes
    for name in input_names:
        dynamic_axes[name] = {batch_dim: "batch_size"}
        # Add sequence dimension for sequence features
        if include_seq_dim and seq_features and name in seq_features:
            dynamic_axes[name][1] = "seq_length"

    # Output axes
    for name in output_names:
        dynamic_axes[name] = {batch_dim: "batch_size"}

    return dynamic_axes


class ONNXExporter:
    """Main class for exporting Torch-RecHub models to ONNX format.

    This exporter handles the complexity of converting dict-input models to ONNX
    by automatically extracting feature information and wrapping the model.

    Args:
        model: The PyTorch recommendation model to export.
        device: Device for export operations (default: 'cpu').

    Example:
        >>> exporter = ONNXExporter(deepfm_model)
        >>> exporter.export("model.onnx")

        >>> # For dual-tower models
        >>> exporter = ONNXExporter(dssm_model)
        >>> exporter.export("user_tower.onnx", mode="user")
        >>> exporter.export("item_tower.onnx", mode="item")
    """

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.feature_info = extract_feature_info(model)

    def export(
        self,
        output_path: str,
        mode: Optional[str] = None,
        dummy_input: Optional[Dict[str,
                                   torch.Tensor]] = None,
        batch_size: int = 2,
        seq_length: int = 10,
        opset_version: int = 14,
        dynamic_batch: bool = True,
        verbose: bool = False
    ) -> bool:
        """Export the model to ONNX format.

        Args:
            output_path: Path to save the ONNX model.
            mode: For dual-tower models, specify "user" or "item" to export
                  only that tower. None exports the full model.
            dummy_input: Optional dict of example inputs. If not provided,
                         dummy inputs will be generated automatically.
            batch_size: Batch size for generated dummy input (default: 2).
            seq_length: Sequence length for SequenceFeature (default: 10).
            opset_version: ONNX opset version (default: 14).
            dynamic_batch: Whether to enable dynamic batch size (default: True).
            verbose: Whether to print export details (default: False).

        Returns:
            True if export succeeded, False otherwise.

        Raises:
            RuntimeError: If ONNX export fails.
        """
        self.model.eval()
        self.model.to(self.device)

        # Determine which features to use based on mode
        if mode == "user":
            features = self.feature_info['user_features']
            if not features:
                raise ValueError("No user features found in model for mode='user'")
        elif mode == "item":
            features = self.feature_info['item_features']
            if not features:
                raise ValueError("No item features found in model for mode='item'")
        else:
            features = self.feature_info['features']

        input_names = [f.name for f in features]

        # Create wrapped model
        wrapper = ONNXWrapper(self.model, input_names, mode=mode)
        wrapper.eval()

        # Generate or use provided dummy input
        if dummy_input is not None:
            dummy_tuple = tuple(dummy_input[name].to(self.device) for name in input_names)
        else:
            dummy_tuple = generate_dummy_input(features, batch_size=batch_size, seq_length=seq_length, device=self.device)

        # Configure dynamic axes
        dynamic_axes = None
        if dynamic_batch:
            seq_feature_names = [f.name for f in features if isinstance(f, SequenceFeature)]
            dynamic_axes = generate_dynamic_axes(input_names=input_names, output_names=["output"], seq_features=seq_feature_names)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        try:
            with torch.no_grad():
                torch.onnx.export(
                    wrapper,
                    dummy_tuple,
                    output_path,
                    input_names=input_names,
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=opset_version,
                    do_constant_folding=True,
                    verbose=verbose,
                    dynamo=False  # Use legacy exporter for dynamic_axes support
                )

            if verbose:
                print(f"Successfully exported ONNX model to: {output_path}")
                print(f"  Input names: {input_names}")
                print(f"  Opset version: {opset_version}")
                print(f"  Dynamic batch: {dynamic_batch}")

            return True

        except Exception as e:
            warnings.warn(f"ONNX export failed: {str(e)}")
            raise RuntimeError(f"Failed to export ONNX model: {str(e)}") from e
        finally:
            # Restore original mode
            wrapper.restore_mode()

    def get_input_info(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Get information about model inputs.

        Args:
            mode: For dual-tower models, "user" or "item".

        Returns:
            Dict with input names, types, and shapes.
        """
        if mode == "user":
            features = self.feature_info['user_features']
        elif mode == "item":
            features = self.feature_info['item_features']
        else:
            features = self.feature_info['features']

        info = []
        for f in features:
            feat_info = {'name': f.name, 'type': type(f).__name__, 'embed_dim': f.embed_dim}
            if hasattr(f, 'vocab_size'):
                feat_info['vocab_size'] = f.vocab_size
            if hasattr(f, 'pooling'):
                feat_info['pooling'] = f.pooling
            info.append(feat_info)

        return {'mode': mode, 'inputs': info, 'input_names': [f.name for f in features]}
