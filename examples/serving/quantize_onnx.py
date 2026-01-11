"""
Quantize an exported ONNX model (INT8 dynamic / FP16).

Examples:
  python examples/serving/quantize_onnx.py --input deepfm.onnx --output deepfm.int8.onnx --mode int8
  python examples/serving/quantize_onnx.py --input deepfm.onnx --output deepfm.fp16.onnx --mode fp16
"""

from __future__ import annotations

import argparse

from torch_rechub.utils.quantization import quantize_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input ONNX model path (fp32).")
    parser.add_argument("--output", required=True, help="Output ONNX model path.")
    parser.add_argument("--mode", default="int8", choices=["int8", "dynamic_int8", "fp16"], help="Quantization mode.")
    parser.add_argument("--per-channel", action="store_true", help="Enable per-channel weight quantization (INT8).")
    parser.add_argument("--reduce-range", action="store_true", help="Use reduced quant range (INT8).")
    parser.add_argument("--weight-type", default="qint8", choices=["qint8", "quint8"], help="Weight quant type (INT8).")
    parser.add_argument("--optimize-model", action="store_true", help="Run ORT graph optimization before quantization.")
    parser.add_argument("--keep-io-types", action="store_true", help="Keep model I/O as fp32 when converting to fp16.")
    args = parser.parse_args()

    out = quantize_model(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
        weight_type=args.weight_type,
        optimize_model=args.optimize_model,
        keep_io_types=args.keep_io_types,
    )
    print("saved:", out)


if __name__ == "__main__":
    main()
