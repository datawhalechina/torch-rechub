"""Convert an exported ONNX model to dynamic INT8 (CPU) or FP16 (GPU)."""

from pathlib import Path


def quantize_model(
    input_path,
    output_path,
    mode="int8",
    *,
    per_channel=False,
    reduce_range=False,
    weight_type="qint8",
    optimize_model=False,
    op_types_to_quantize=None,
    nodes_to_quantize=None,
    nodes_to_exclude=None,
    extra_options=None,
    keep_io_types=True,
):
    """Save a quantized model and return ``output_path``.

    ``int8`` uses ONNX Runtime dynamic quantization; its optional weight/node
    arguments are passed directly to ORT. ``fp16`` uses onnxconverter-common
    and keeps floating-point inputs/outputs in FP32 by default.

    ``optimize_model`` is supported only by ORT versions older than 1.16.
    This utility is intended for small/medium models, not >2 GB external data.
    """
    mode = (mode or "").strip().lower()
    mode = {"dynamic_int8": "int8", "dynamic": "int8", "float16": "fp16"}.get(mode, mode)
    if mode not in ("int8", "fp16"):
        raise ValueError("mode must be 'int8' (CPU) or 'fp16' (GPU)")
    weight_type = (weight_type or "").strip().lower()
    if mode == "int8" and weight_type not in ("qint8", "int8", "signed", "quint8", "uint8", "unsigned"):
        raise ValueError("weight_type must be 'qint8' or 'quint8'")
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("input_path and output_path must be different files")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if mode == "int8":
        try:
            import onnx
            import onnxruntime as ort
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError as error:
            raise ImportError('INT8 requires: pip install "torch-rechub[onnx]"') from error

        if weight_type in ("qint8", "int8", "signed"):
            quant_type = QuantType.QInt8
        else:
            quant_type = QuantType.QUInt8

        # ORT 1.16 removed this argument; older ORT defaults to True, not False.
        ort_version = tuple(int(part) for part in ort.__version__.split(".")[:2])
        optimizer_options = {}
        if ort_version < (1, 16):
            optimizer_options["optimize_model"] = optimize_model
        elif optimize_model:
            raise ValueError("ORT >= 1.16 removed optimize_model; use ORT quant_pre_process() before quantizing")

        model_input = str(input_path)
        if ort_version >= (1, 18):
            # ORT transposes Gemm weights; their duplicated shape metadata becomes stale.
            model_input = onnx.load(input_path)
            initializer_names = {value.name for value in model_input.graph.initializer}
            value_info = [value for value in model_input.graph.value_info if value.name not in initializer_names]
            del model_input.graph.value_info[:]
            model_input.graph.value_info.extend(value_info)

        quantize_dynamic(
            model_input=model_input,
            model_output=str(output_path),
            per_channel=per_channel,
            reduce_range=reduce_range,
            weight_type=quant_type,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_options,
            **optimizer_options,
        )
    else:
        try:
            import onnx
            from onnxconverter_common import float16
        except ImportError as error:
            raise ImportError('FP16 requires: pip install "torch-rechub[onnx]"') from error

        model = onnx.load(input_path)
        model = float16.convert_float_to_float16(model, keep_io_types=keep_io_types)
        onnx.save(model, output_path)

    return output_path
