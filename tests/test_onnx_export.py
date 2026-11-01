import os

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.models.multi_task import MMOE
from torch_rechub.models.ranking import DIN, WideDeep
from torch_rechub.trainers import CTRTrainer, MatchTrainer, MTLTrainer
from torch_rechub.utils.quantization import quantize_model


@pytest.mark.parametrize("kind", ["ranking", "sequence", "user", "item", "multitask"])
def test_trainer_export(tmp_path, kind):
    torch.manual_seed(2026)
    features = [DenseFeature("score"), SparseFeature("item", 32, 4)]
    x = {"score": torch.randn(3, 1), "item": torch.tensor([1, 2, 3])}
    options = {}
    if kind == "ranking":
        model = WideDeep(features[:1], features[1:], mlp_params={"dims": [8]})
        trainer = CTRTrainer(model, device="cpu")
    elif kind == "sequence":
        history = [SequenceFeature("history", 32, 4, pooling="concat", shared_with="item")]
        model = DIN(features[:1], history, features[1:], mlp_params={"dims": [8]}, attention_mlp_params={"dims": [8]})
        x["history"] = torch.tensor([[1, 2, 0], [3, 0, 0], [2, 3, 4]])
        options["seq_length"] = 3
        trainer = CTRTrainer(model, device="cpu")
    elif kind in ("user", "item"):
        model = DSSM([SparseFeature("user", 16, 4)], features[1:], user_params={"dims": [8]}, item_params={"dims": [8]})
        x = {"user": torch.tensor([1, 2, 3]), "item": x["item"]}
        trainer = MatchTrainer(model, device="cpu")
        options["mode"] = kind
    else:
        model = MMOE(features, ["classification"] * 2, n_expert=2, expert_params={"dims": [8]}, tower_params_list=[{"dims": [8]}] * 2)
        trainer = MTLTrainer(model, ["classification"] * 2, device="cpu")

    path = str(tmp_path / f"{kind}.onnx")
    assert trainer.export_onnx(path, verbose=False, **options)
    onnx.checker.check_model(onnx.load(path))
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    model.eval()
    # Exercise dynamic batch sizes through ONNX, not just file creation.
    for size in (1, 3):
        batch = {name: values[:size] for name, values in x.items()}
        with torch.no_grad():
            expected = getattr(model, f"{kind}_tower")(batch) if kind in ("user", "item") else model(batch)
        inputs = {value.name: batch[value.name].numpy() for value in session.get_inputs()}
        actual = session.run(None, inputs)[0]
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-3, atol=1e-5)


@pytest.mark.parametrize("dynamo", [False, True])
def test_quantize_dynamic_int8_creates_model(tmp_path, dynamo):
    if dynamo:
        pytest.importorskip("onnxscript")
    torch.manual_seed(2026)

    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)).eval()
    x = torch.randn(2, 8)

    fp32_path = str(tmp_path / "tiny_fp32.onnx")
    int8_path = str(tmp_path / "tiny_int8.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,
             ),
            fp32_path,
            input_names=["x"],
            output_names=["output"],
            opset_version=18 if dynamo else 14,
            do_constant_folding=True,
            dynamo=dynamo,
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
    pytest.importorskip("onnxconverter_common")

    model = torch.nn.Linear(4, 2).eval()
    x = torch.randn(2, 4)

    fp32_path = str(tmp_path / "linear_fp32.onnx")
    fp16_path = str(tmp_path / "linear_fp16.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x,
             ),
            fp32_path,
            input_names=["x"],
            output_names=["output"],
            opset_version=14,
            do_constant_folding=True,
            dynamo=False,
        )

    quantize_model(fp32_path, fp16_path, mode="fp16", keep_io_types=True)
    assert os.path.exists(fp16_path)
    converted = onnx.load(fp16_path)
    onnx.checker.check_model(converted)
    assert converted.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert converted.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
