import inspect
import os

import pytest


@pytest.mark.parametrize("dynamo", [False, True])
def test_quantize_dynamic_int8_creates_model(tmp_path, dynamo):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    import numpy as np
    import torch

    from torch_rechub.utils.quantization import quantize_model

    has_dynamo = "dynamo" in inspect.signature(torch.onnx.export).parameters
    if dynamo and (not has_dynamo or tuple(int(part) for part in ort.__version__.split(".")[:2]) < (1, 18)):
        pytest.skip("Dynamo fixture requires a supporting PyTorch exporter and ORT >= 1.18")
    if dynamo:
        pytest.importorskip("onnxscript")
    torch.manual_seed(2026)

    class TinyMLP(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))

        def forward(self, x):
            return self.net(x)

    model = TinyMLP().eval()
    x = torch.randn(2, 8)

    fp32_path = str(tmp_path / "tiny_fp32.onnx")
    int8_path = str(tmp_path / "tiny_int8.onnx")

    with torch.no_grad():
        export_options = {"dynamo": dynamo} if has_dynamo else {}
        torch.onnx.export(
            model,
            (x,
             ),
            fp32_path,
            input_names=["x"],
            output_names=["output"],
            opset_version=18 if dynamo else 14,
            do_constant_folding=True,
            **export_options,
        )

    quantize_model(fp32_path, int8_path, mode="int8")
    assert os.path.exists(int8_path)
    assert os.path.getsize(int8_path) > 0

    # sanity: onnxruntime can load + run
    sess = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"x": x.numpy().astype(np.float32)})
    assert len(out) == 1
    assert out[0].shape == (2, 1)
    np.testing.assert_allclose(out[0], model(x).detach().numpy(), rtol=0.1, atol=0.02)


def test_convert_fp16_creates_model(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxconverter_common")

    import onnx
    import torch

    from torch_rechub.utils.quantization import quantize_model

    model = torch.nn.Linear(4, 2).eval()
    x = torch.randn(2, 4)

    fp32_path = str(tmp_path / "linear_fp32.onnx")
    fp16_path = str(tmp_path / "linear_fp16.onnx")

    with torch.no_grad():
        # Test the supported legacy-export path; older torch has no dynamo flag.
        export_options = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
        torch.onnx.export(
            model,
            (x,
             ),
            fp32_path,
            input_names=["x"],
            output_names=["output"],
            opset_version=14,
            do_constant_folding=True,
            **export_options,
        )

    quantize_model(fp32_path, fp16_path, mode="fp16", keep_io_types=True)
    assert os.path.exists(fp16_path)
    converted = onnx.load(fp16_path)
    onnx.checker.check_model(converted)
    assert converted.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert converted.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
