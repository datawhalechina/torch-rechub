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


# Re-export from model_utils for backward compatibility
# The actual implementations are now in model_utils.py
from .model_utils import extract_feature_info, generate_dummy_input, generate_dummy_input_dict, generate_dynamic_axes


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
