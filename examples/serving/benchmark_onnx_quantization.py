"""Compare ONNX latency on the same inputs; use --inputs batch.npz for real data."""

import argparse
import os
import time

import numpy as np

FLOAT_DTYPES = {"tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64}


def _make_feeds(sess, batch_size, seq_len):
    """Generate floating-point demo data; IDs/masks/lengths require real inputs."""
    feeds = {}
    rng = np.random.default_rng(2022)
    for inp in sess.get_inputs():
        if inp.type not in FLOAT_DTYPES:
            raise ValueError(f"Input {inp.name!r} has type {inp.type}; supply valid IDs/masks/lengths with --inputs batch.npz")
        shape = []
        for i, d in enumerate(inp.shape):
            if d is None or isinstance(d, str):
                # Demo convention only; real inputs need no shape guessing.
                shape.append(batch_size if i == 0 else seq_len if i == 1 else 1)
            else:
                shape.append(int(d))

        feeds[inp.name] = rng.random(shape).astype(FLOAT_DTYPES[inp.type])
    return feeds


def _bench_one(sess, path, feeds, warmup, repeat):
    if {inp.name for inp in sess.get_inputs()} != set(feeds):
        raise ValueError(f"Input names differ for {path}; all compared models must use the same input names")
    feeds = dict(feeds)
    for inp in sess.get_inputs():
        # A model with converted FP16 I/O needs the same values in float16.
        if inp.type in FLOAT_DTYPES and np.issubdtype(feeds[inp.name].dtype, np.floating):
            feeds[inp.name] = feeds[inp.name].astype(FLOAT_DTYPES[inp.type], copy=False)
    for _ in range(warmup):
        _ = sess.run(None, feeds)

    start = time.perf_counter()
    for _ in range(repeat):
        _ = sess.run(None, feeds)
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / repeat
    print(f"- {os.path.basename(path)}")
    print(f"  ONNX file: {os.path.getsize(path) / (1024 * 1024):.2f} MB (external weight files excluded)")
    print(f"  registered providers: {sess.get_providers()} (some operators may run on CPU)")
    print(f"  input shapes: { {name: list(value.shape) for name, value in feeds.items()} }")
    print(f"  avg latency: {avg_ms:.3f} ms  (repeat={repeat}, CPU threads=1, includes host/device copies)")


def main():
    import onnxruntime as ort

    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True, help="FP32 ONNX model path.")
    parser.add_argument("--int8", default=None, help="INT8 ONNX model path (optional).")
    parser.add_argument("--fp16", default=None, help="FP16 ONNX model path (optional).")
    parser.add_argument("--provider", default="CPUExecutionProvider", help="ONNX Runtime execution provider.")
    parser.add_argument("--inputs", help="A .npz batch keyed by ONNX input name; required for IDs/masks/lengths.")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for dynamic inputs.")
    parser.add_argument("--seq-len", type=int, default=10, help="Dummy seq length for dynamic sequence inputs.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs.")
    parser.add_argument("--repeat", type=int, default=200, help="Benchmark runs.")
    args = parser.parse_args()
    if min(args.batch_size, args.seq_len, args.repeat) < 1 or args.warmup < 0:
        parser.error("batch-size, seq-len and repeat must be positive; warmup must be nonnegative")
    if args.provider not in ort.get_available_providers():
        parser.error(f"{args.provider} is unavailable; installed providers: {ort.get_available_providers()}")

    paths = [path for path in (args.fp32, args.int8, args.fp16) if path]
    feeds = None
    if args.inputs:
        with np.load(args.inputs, allow_pickle=False) as batch:
            feeds = {name: batch[name] for name in batch.files}

    print("## ONNX Benchmark")
    for path in paths:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        sess = ort.InferenceSession(path, sess_options=options, providers=[args.provider])
        if args.provider not in sess.get_providers():
            raise RuntimeError(f"{args.provider} failed to initialize; actual providers: {sess.get_providers()}")
        if feeds is None:
            feeds = _make_feeds(sess, args.batch_size, args.seq_len)
        _bench_one(sess, path, feeds, args.warmup, args.repeat)


if __name__ == "__main__":
    main()
