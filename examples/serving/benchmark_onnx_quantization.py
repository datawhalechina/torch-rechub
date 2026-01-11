"""
Benchmark ONNX inference latency + model size.

This script compares:
  - ONNX FP32
  - ONNX INT8 (dynamic quantized)
  - ONNX FP16 (converted)

It generates dummy inputs from the ONNX model input signatures.

Examples:
  python examples/serving/benchmark_onnx_quantization.py --fp32 deepfm.onnx --int8 deepfm.int8.onnx
  python examples/serving/benchmark_onnx_quantization.py --fp32 deepfm.onnx --fp16 deepfm.fp16.onnx --provider CUDAExecutionProvider
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Dict

import numpy as np


def _model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def _make_feeds(sess, batch_size: int, seq_len: int) -> Dict[str, np.ndarray]:
    feeds: Dict[str, np.ndarray] = {}
    for inp in sess.get_inputs():
        shape = []
        for i, d in enumerate(inp.shape):
            if d is None or isinstance(d, str):
                # heuristic: first dynamic dim -> batch, second -> seq_len
                if i == 0:
                    shape.append(batch_size)
                elif i == 1:
                    shape.append(seq_len)
                else:
                    shape.append(1)
            else:
                shape.append(int(d))

        t = inp.type
        if "int64" in t:
            arr = np.random.randint(0, 100, size=shape, dtype=np.int64)
        elif "int32" in t:
            arr = np.random.randint(0, 100, size=shape, dtype=np.int32)
        elif "float16" in t:
            arr = np.random.rand(*shape).astype(np.float16)
        else:
            # default float32
            arr = np.random.rand(*shape).astype(np.float32)

        feeds[inp.name] = arr
    return feeds


def _bench_one(path: str, provider: str, batch_size: int, seq_len: int, warmup: int, repeat: int) -> None:
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess = ort.InferenceSession(path, sess_options=sess_options, providers=[provider])
    feeds = _make_feeds(sess, batch_size=batch_size, seq_len=seq_len)

    # warmup
    for _ in range(warmup):
        _ = sess.run(None, feeds)

    start = time.perf_counter()
    for _ in range(repeat):
        _ = sess.run(None, feeds)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / repeat
    print(f"- {os.path.basename(path)}")
    print(f"  size: { _model_size_mb(path):.2f} MB")
    print(f"  provider: {provider}")
    print(f"  avg latency: {avg_ms:.3f} ms  (batch={batch_size}, seq_len={seq_len}, repeat={repeat})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True, help="FP32 ONNX model path.")
    parser.add_argument("--int8", default=None, help="INT8 ONNX model path (optional).")
    parser.add_argument("--fp16", default=None, help="FP16 ONNX model path (optional).")
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for dynamic inputs.")
    parser.add_argument("--seq-len", type=int, default=10, help="Dummy seq length for dynamic sequence inputs.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs.")
    parser.add_argument("--repeat", type=int, default=200, help="Benchmark runs.")
    args = parser.parse_args()

    print("## ONNX Benchmark")
    _bench_one(args.fp32, args.provider, args.batch_size, args.seq_len, args.warmup, args.repeat)
    if args.int8:
        _bench_one(args.int8, args.provider, args.batch_size, args.seq_len, args.warmup, args.repeat)
    if args.fp16:
        _bench_one(args.fp16, args.provider, args.batch_size, args.seq_len, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
