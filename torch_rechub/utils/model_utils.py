"""Common model utility functions for Torch-RecHub.

This module provides shared utilities for model introspection and input generation,
used by both ONNX export and visualization features.

Examples
--------
>>> from torch_rechub.utils.model_utils import extract_feature_info, generate_dummy_input
>>> feature_info = extract_feature_info(model)
>>> dummy_input = generate_dummy_input(feature_info['features'], batch_size=2)
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Import feature types for type checking
try:
    from ..basic.features import DenseFeature, SequenceFeature, SparseFeature
except ImportError:
    # Fallback for standalone usage
    SparseFeature = None
    DenseFeature = None
    SequenceFeature = None


def extract_feature_info(model: nn.Module) -> Dict[str, Any]:
    """Extract feature information from a torch-rechub model via reflection.

    Parameters
    ----------
    model : nn.Module
        Model to inspect.

    Returns
    -------
    dict
        {
            'features': list of unique Feature objects,
            'input_names': ordered feature names,
            'input_types': map name -> feature type,
            'user_features': user-side features (dual-tower),
            'item_features': item-side features (dual-tower),
        }

    Examples
    --------
    >>> from torch_rechub.models.ranking import DeepFM
    >>> model = DeepFM(deep_features, fm_features, mlp_params)
    >>> info = extract_feature_info(model)
    >>> info['input_names']  # ['user_id', 'item_id', ...]
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

    # Build input names and types
    input_names = [f.name for f in unique_features if hasattr(f, 'name')]
    input_types = {f.name: type(f).__name__ for f in unique_features if hasattr(f, 'name')}

    return {
        'features': unique_features,
        'input_names': input_names,
        'input_types': input_types,
        'user_features': unique_user,
        'item_features': unique_item,
    }


def generate_dummy_input(features: List[Any], batch_size: int = 2, seq_length: int = 10, device: str = 'cpu') -> Tuple[torch.Tensor, ...]:
    """Generate dummy input tensors based on feature definitions.

    Parameters
    ----------
    features : list
        List of Feature objects (SparseFeature, DenseFeature, SequenceFeature).
    batch_size : int, default=2
        Batch size for dummy input.
    seq_length : int, default=10
        Sequence length for SequenceFeature.
    device : str, default='cpu'
        Device to create tensors on.

    Returns
    -------
    tuple of Tensor
        Tuple of tensors in the order of input features.

    Examples
    --------
    >>> features = [SparseFeature("user_id", 1000), SequenceFeature("hist", 500)]
    >>> dummy = generate_dummy_input(features, batch_size=4)
    >>> # Returns (user_id_tensor[4], hist_tensor[4, 10])
    """
    # Dynamic import to handle feature types
    from ..basic.features import DenseFeature, SequenceFeature, SparseFeature

    inputs = []
    for feat in features:
        if isinstance(feat, SequenceFeature):
            # Sequence features have shape [batch_size, seq_length]
            tensor = torch.randint(0, feat.vocab_size, (batch_size, seq_length), device=device)
        elif isinstance(feat, SparseFeature):
            # Sparse features have shape [batch_size]
            tensor = torch.randint(0, feat.vocab_size, (batch_size,), device=device)
        elif isinstance(feat, DenseFeature):
            # Dense features always have shape [batch_size, embed_dim]
            tensor = torch.randn(batch_size, feat.embed_dim, device=device)
        else:
            raise TypeError(f"Unsupported feature type: {type(feat)}")
        inputs.append(tensor)
    return tuple(inputs)


def generate_dummy_input_dict(features: List[Any], batch_size: int = 2, seq_length: int = 10, device: str = 'cpu') -> Dict[str, torch.Tensor]:
    """Generate dummy input dict based on feature definitions.

    Similar to generate_dummy_input but returns a dict mapping feature names
    to tensors. This is the expected input format for torch-rechub models.

    Parameters
    ----------
    features : list
        List of Feature objects (SparseFeature, DenseFeature, SequenceFeature).
    batch_size : int, default=2
        Batch size for dummy input.
    seq_length : int, default=10
        Sequence length for SequenceFeature.
    device : str, default='cpu'
        Device to create tensors on.

    Returns
    -------
    dict
        Dict mapping feature names to tensors.

    Examples
    --------
    >>> features = [SparseFeature("user_id", 1000)]
    >>> dummy = generate_dummy_input_dict(features, batch_size=4)
    >>> # Returns {"user_id": tensor[4]}
    """
    dummy_tuple = generate_dummy_input(features, batch_size, seq_length, device)
    input_names = [f.name for f in features if hasattr(f, 'name')]
    return {name: tensor for name, tensor in zip(input_names, dummy_tuple)}


def generate_dynamic_axes(input_names: List[str], output_names: Optional[List[str]] = None, batch_dim: int = 0, include_seq_dim: bool = True, seq_features: Optional[List[str]] = None) -> Dict[str, Dict[int, str]]:
    """Generate dynamic axes configuration for ONNX export.

    Parameters
    ----------
    input_names : list of str
        List of input tensor names.
    output_names : list of str, optional
        List of output tensor names. Default is ["output"].
    batch_dim : int, default=0
        Dimension index for batch size.
    include_seq_dim : bool, default=True
        Whether to include sequence dimension as dynamic.
    seq_features : list of str, optional
        List of feature names that are sequences.

    Returns
    -------
    dict
        Dynamic axes dict for torch.onnx.export.

    Examples
    --------
    >>> axes = generate_dynamic_axes(["user_id", "item_id"], seq_features=["hist"])
    >>> # Returns {"user_id": {0: "batch_size"}, "item_id": {0: "batch_size"}, ...}
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
