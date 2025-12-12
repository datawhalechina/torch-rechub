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

import inspect
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..basic.features import DenseFeature, SequenceFeature, SparseFeature


class ONNXWrapper(nn.Module):
    """Wrap a dict-input model to accept positional args for ONNX.

    ONNX disallows dict inputs; this wrapper maps positional args back to dict
    before calling the original model.

    Parameters
    ----------
    model : nn.Module
        Original dict-input model.
    input_names : list[str]
        Ordered feature names matching positional inputs.
    mode : {'user', 'item'}, optional
        For dual-tower models, set tower mode.

    Examples
    --------
    >>> wrapper = ONNXWrapper(dssm_model, ["user_id", "movie_id", "hist_movie_id"])
    >>> wrapper(user_id_tensor, movie_id_tensor, hist_tensor)
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
        verbose: bool = False,
        onnx_export_kwargs: Optional[Dict[str,
                                          Any]] = None,
    ) -> bool:
        """Export model to ONNX format.

        Parameters
        ----------
        output_path : str
            Destination path.
        mode : {'user', 'item'}, optional
            For dual-tower, export specific tower; None exports full model.
        dummy_input : dict[str, Tensor], optional
            Example inputs; auto-generated if None.
        batch_size : int, default=2
            Batch size for dummy input generation.
        seq_length : int, default=10
            Sequence length for SequenceFeature.
        opset_version : int, default=14
            ONNX opset.
        dynamic_batch : bool, default=True
            Enable dynamic batch axes.
        verbose : bool, default=False
            Print export details.
        onnx_export_kwargs : dict, optional
            Extra keyword args forwarded to ``torch.onnx.export`` (e.g. ``operator_export_type``,
            ``keep_initializers_as_inputs``, ``do_constant_folding``).
            Notes:
              - If you pass keys that overlap with the explicit parameters above
                (like ``opset_version`` / ``dynamic_axes`` / ``input_names``), this function
                will raise a ``ValueError`` to avoid ambiguous behavior.
              - Some kwargs (like ``dynamo``) are only available in newer PyTorch; unsupported
                keys will be ignored for compatibility.

        Returns
        -------
        bool
            True if export succeeds.

        Raises
        ------
        RuntimeError
            If ONNX export fails.
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
                export_kwargs: Dict[str,
                                    Any] = {
                                        "f": output_path,
                                        "input_names": input_names,
                                        "output_names": ["output"],
                                        "dynamic_axes": dynamic_axes,
                                        "opset_version": opset_version,
                                        "do_constant_folding": True,
                                        "verbose": verbose,
                                    }

                if onnx_export_kwargs:
                    # Prevent silent conflicts with explicit arguments
                    overlap = set(export_kwargs.keys()) & set(onnx_export_kwargs.keys())
                    # allow user to set 'dynamo' even if we inject it later
                    overlap.discard("dynamo")
                    if overlap:
                        raise ValueError("onnx_export_kwargs contains keys that overlap with explicit args: "
                                         f"{sorted(overlap)}. Please set them via export() parameters instead.")
                    export_kwargs.update(onnx_export_kwargs)

                # Auto-pick exporter:
                # - When dynamic axes are requested, prefer legacy exporter (dynamo=False),
                #   because the dynamo exporter may not honor `dynamic_axes` consistently
                #   across torch versions.
                # - When no dynamic axes are requested, prefer dynamo exporter (dynamo=True)
                #   for better operator coverage in newer torch.
                #
                # In older torch versions, 'dynamo' kwarg does not exist.
                sig = inspect.signature(torch.onnx.export)
                if "dynamo" in sig.parameters:
                    if "dynamo" not in export_kwargs:
                        export_kwargs["dynamo"] = False if dynamic_axes is not None else True
                else:
                    export_kwargs.pop("dynamo", None)

                torch.onnx.export(wrapper, dummy_tuple, **export_kwargs)

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
