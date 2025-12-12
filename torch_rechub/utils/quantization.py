"""
ONNX Quantization Utilities.

This module provides a lightweight API to quantize exported ONNX models:
- INT8 dynamic quantization (recommended for MLP-heavy rec models on CPU)
- FP16 conversion (recommended for GPU inference)

The functions are optional-dependency friendly:
- INT8 quantization requires: onnxruntime
- FP16 conversion requires: onnx + onnxconverter-common
"""

from __future__ import annotations

import inspect
import os
from typing import Any, Dict, Optional


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def quantize_model(
    input_path: str,
    output_path: str,
    mode: str = "int8",
    *,
    # INT8(dynamic) params
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: str = "qint8",
    optimize_model: bool = False,
    op_types_to_quantize: Optional[list[str]] = None,
    nodes_to_quantize: Optional[list[str]] = None,
    nodes_to_exclude: Optional[list[str]] = None,
    extra_options: Optional[Dict[str,
                                 Any]] = None,
    # FP16 params
    keep_io_types: bool = True,
) -> str:
    """Quantize an ONNX model.

    Parameters
    ----------
    input_path : str
        Input ONNX model path (FP32).
    output_path : str
        Output ONNX model path.
    mode : str, default="int8"
        Quantization mode:
        - "int8" / "dynamic_int8": ONNX Runtime dynamic quantization (weights INT8).
        - "fp16": convert float tensors to float16.
    per_channel : bool, default=False
        Enable per-channel quantization for weights (INT8).
    reduce_range : bool, default=False
        Use reduced quantization range (INT8), sometimes helpful on certain CPUs.
    weight_type : {"qint8", "quint8"}, default="qint8"
        Weight quant type for dynamic quantization.
    optimize_model : bool, default=False
        Run ORT graph optimization before quantization.
    op_types_to_quantize / nodes_to_quantize / nodes_to_exclude / extra_options
        Advanced options forwarded to ``onnxruntime.quantization.quantize_dynamic``.
    keep_io_types : bool, default=True
        For FP16 conversion, keep model input/output types as float32 for compatibility.

    Returns
    -------
    str
        The output_path.
    """
    mode_norm = (mode or "").strip().lower()
    _ensure_parent_dir(output_path)

    if mode_norm in ("int8", "dynamic_int8", "dynamic"):
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except Exception as e:  # pragma: no cover
            raise ImportError("INT8 quantization requires onnxruntime. Install with: pip install -U \"torch-rechub[onnx]\"") from e

        wt = (weight_type or "").strip().lower()
        if wt in ("qint8", "int8", "signed"):
            qt = QuantType.QInt8
        elif wt in ("quint8", "uint8", "unsigned"):
            qt = QuantType.QUInt8
        else:
            raise ValueError("weight_type must be one of {'qint8','quint8'}")

        q_kwargs: Dict[str,
                       Any] = {
                           "model_input": input_path,
                           "model_output": output_path,
                           "per_channel": per_channel,
                           "reduce_range": reduce_range,
                           "weight_type": qt,
                           "optimize_model": optimize_model,
                           "op_types_to_quantize": op_types_to_quantize,
                           "nodes_to_quantize": nodes_to_quantize,
                           "nodes_to_exclude": nodes_to_exclude,
                           "extra_options": extra_options,
                       }

        # Compatibility: different onnxruntime versions expose different kwargs.
        sig = inspect.signature(quantize_dynamic)
        q_kwargs = {k: v for k, v in q_kwargs.items() if k in sig.parameters and v is not None}

        quantize_dynamic(**q_kwargs)
        return output_path

    if mode_norm in ("fp16", "float16"):
        try:
            import onnx
        except Exception as e:  # pragma: no cover
            raise ImportError("FP16 conversion requires onnx. Install with: pip install -U \"torch-rechub[onnx]\"") from e

        try:
            from onnxconverter_common import float16
        except Exception as e:  # pragma: no cover
            raise ImportError("FP16 conversion requires onnxconverter-common. Install with: pip install -U onnxconverter-common") from e

        model = onnx.load(input_path)
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
        onnx.save(model_fp16, output_path)
        return output_path

    raise ValueError("mode must be one of {'int8','dynamic_int8','fp16'}")
