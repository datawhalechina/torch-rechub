---
title: ONNX Export & Quantization
description: Torch-RecHub model export to ONNX and quantization
---

# ONNX Export & Quantization

Torch-RecHub supports exporting trained models to ONNX for cross-platform inference deployment. This page covers the complete usage and recommended practices for **ONNX export** and **quantization (INT8/FP16)** in production inference scenarios that require low latency and low memory usage.

![ONNX export and quantization pipeline](/img/diagrams/onnx_quantization_flow.png)

## Install Dependencies

ONNX-related dependencies are optional and can be installed as needed:

```bash
pip install "torch-rechub[onnx]"
```

Notes:
- `torch-rechub[onnx]` installs `ml-dtypes`, `onnx`, `onnxscript`, `onnxruntime`, and `onnxconverter-common` for FP16 conversion.
- For GPU inference, use the `onnxruntime-gpu` package that matches your local CUDA version and driver, and verify that your environment actually imports the intended Runtime variant.

## Export to ONNX (Trainer export_onnx)

Every Torch-RecHub trainer provides an `export_onnx()` method (CTR/Matching/MTL/Seq). The export process automatically constructs dummy inputs and supports a **dynamic batch size**.

### Export CTR Models (Ranking/Re-ranking)

```python
from torch_rechub.trainers import CTRTrainer

# ... trainer.fit(train_dl, val_dl)
trainer.export_onnx("deepfm.onnx")
```

### Export Matching Models (Retrieval): Full Model / Separate Towers

For two-tower models, it is generally best to export the user tower and item tower separately so their embeddings can be computed independently online:

```python
from torch_rechub.trainers import MatchTrainer

# Export the user tower (for user embeddings)
trainer.export_onnx("user_tower.onnx", mode="user")

# Export the item tower (for item embeddings)
trainer.export_onnx("item_tower.onnx", mode="item")
```

### Export Multi-Task Models (MTL)

```python
from torch_rechub.trainers import MTLTrainer

trainer.export_onnx("mmoe.onnx")
```

### Export Arguments and Advanced Control (onnx_export_kwargs)

To select an exporter or pass arguments to `torch.onnx.export()` that are not already used by the wrapper, provide them through `onnx_export_kwargs`:

```python
trainer.export_onnx(
    "model.onnx",
    dynamic_batch=True,  # Dynamic batch size (recommended)
    onnx_export_kwargs={
        "dynamo": False,  # Force the legacy exporter in this example
    },
)
```

The wrapper already sets `f`, `input_names`, `output_names`, `opset_version`, `do_constant_folding`, `verbose`, and the automatically generated `dynamic_axes`. Do not repeat them in `onnx_export_kwargs`, or a `ValueError` will be raised.

Exporter selection recommendations:
- **CTR / Matching / MTL**: By default, first try `dynamo=True`, expressing the dynamic batch through `dynamic_shapes`; on failure, the exporter automatically falls back to the legacy exporter and `dynamic_axes`.
- **SeqTrainer**: Dynamic batch and sequence lengths use the legacy exporter by default; for fixed shapes, you can explicitly pass `dynamo=True` when needed.
- **Older PyTorch versions**: If `dynamo` is unsupported, the exporter ignores the argument and uses a compatible path. Operator coverage may still differ across PyTorch versions.

### Inspect the ONNX Model Structure

After exporting to ONNX, you can inspect the model structure online with [Netron](https://netron.app/):

1. Open https://netron.app/
2. Drag or upload the exported `.onnx` file
3. Inspect the model's network structure, layer parameters, and tensor shapes

> **Tip**: Netron supports multiple model formats (ONNX, TensorFlow, PyTorch, and others) and is a convenient tool for debugging and validating exported models.

## ONNX Quantization

FP32 is often not the best choice for production inference. Two common compression methods are:
- **INT8 dynamic quantization**: Intended for **CPU** inference, primarily quantizing operations such as Linear/MatMul. Measure performance and accuracy on your own data.
- **FP16 conversion**: Intended for **GPUs** supporting FP16. It reduces floating-point weight storage; throughput depends on hardware and operator support.

These are not two GPU presets selected by VRAM size. The INT8 entry point does not implement TensorRT GPU INT8 calibration, and this utility does not handle models with more than 2 GB of external weights.

Torch-RecHub provides a unified API in its quantization module:

```python
from torch_rechub.utils.quantization import quantize_model
```

### 1) INT8 Dynamic Quantization (Recommended for CPU)

```python
from torch_rechub.utils.quantization import quantize_model

quantize_model("model_fp32.onnx", "model_int8.onnx", mode="int8")
```

Optional arguments:
- `per_channel=True`: Enable per-channel quantization for weights
- `reduce_range=True`: Reduce the quantization range, which may be more stable on some CPUs
- `weight_type="qint8"|"quint8"`: Weight quantization type

Existing `dynamic_int8`/`dynamic` aliases and node-selection arguments remain available; unsupported arguments are no longer silently discarded. `optimize_model` is supported only by ORT < 1.16 and retains the original `False` default. ORT >= 1.16 removed that argument, so explicitly requesting `True` raises a clear error. Use ORT's separate `quant_pre_process()` step if graph optimization is needed.

Compatibility: some Dynamo-exported Gemm models contain duplicated weight-shape metadata that becomes stale when ORT transposes the weights. ORT >= 1.18 accepts a ModelProto directly, allowing this utility to remove redundant `value_info` entries matching initializer names in an in-memory copy while preserving other intermediate metadata and the original file. The exporter is unchanged. Older ORT versions retain the original file-path interface; if they encounter this Dynamo/Gemm issue, explicitly export with `trainer.export_onnx(..., onnx_export_kwargs={"dynamo": False})` or pass `dynamo=False` to `torch.onnx.export()`.

### 2) FP16 Conversion (Recommended for GPU)

```python
from torch_rechub.utils.quantization import quantize_model

quantize_model("model_fp32.onnx", "model_fp16.onnx", mode="fp16")
```

Both the Python API and CLI default to `keep_io_types=True`, retaining FP32 floating-point inputs/outputs. Integer category IDs are unchanged. Set `keep_io_types=False`, or pass `--no-keep-io-types` to the CLI, to convert floating-point I/O too. The existing `float16` alias is preserved.

## Example Scripts and Benchmarks

The repository provides scripts for quick validation:

### Quantization Scripts

```bash
python examples/serving/quantize_onnx.py --input model_fp32.onnx --output model_int8.onnx --mode int8
python examples/serving/quantize_onnx.py --input model_fp32.onnx --output model_fp16.onnx --mode fp16
```

### Performance Comparison Scripts

Compare **model size** and **inference latency** across FP32, INT8, and FP16:

```bash
python examples/serving/benchmark_onnx_quantization.py --fp32 model_fp32.onnx --int8 model_int8.onnx --inputs batch.npz
python examples/serving/benchmark_onnx_quantization.py --fp32 model_fp32.onnx --fp16 model_fp16.onnx --inputs batch.npz --provider CUDAExecutionProvider
```

Save one batch of preprocessed inputs in `batch.npz`, keyed by the ONNX input names, for example `np.savez("batch.npz", user_id=user_ids, item_id=item_ids)`. Integer arrays must use the model's required dtype. Every compared model receives this same batch; only floating-point dtype conversion is applied when a model uses FP16 I/O.

Without `--inputs`, fixed-seed demo data is generated only for floating-point inputs. IDs, masks and lengths require real input data rather than guessed vocabulary sizes or valid values. The script checks that the requested provider is available and successfully registered instead of reporting CPU fallback as GPU success; registering CUDA still does not mean that every operator runs on GPU.

Reported size covers the `.onnx` file only, excluding external weights. The benchmark fixes ORT CPU operator threads to one. Latency includes input/output copies rather than measuring GPU kernel time alone.

> A successful export only confirms that the model file was generated. Before production use, run both PyTorch and ONNX Runtime on real samples to compare output shapes, numerical errors, and dynamic batching, then reevaluate business metrics after quantization. The benchmark script does not replace these consistency checks.
