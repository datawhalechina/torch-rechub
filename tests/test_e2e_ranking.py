import copy

import numpy as np
import pytest
import torch

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models import ranking
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator


@pytest.mark.parametrize("model_name", ranking.__all__)
def test_ranking_e2e(model_name, tmp_path):
    """Every public ranking model trains, validates and reloads with a small CPU fixture."""
    torch.manual_seed(2022)
    rng = np.random.default_rng(2022)
    dense = [DenseFeature("d1"), DenseFeature("d2")]
    sparse = [SparseFeature("s1", 32, 4), SparseFeature("s2", 32, 4)]
    features = dense + sparse
    data = {f.name: rng.normal(size=64).astype("float32") for f in dense}
    data.update({f.name: rng.integers(1, 32, size=64) for f in sparse})
    labels = np.tile([0, 1], 32)
    params = {"mlp_params": {"dims": [8]}}
    if model_name == "WideDeep":
        params.update(wide_features=dense, deep_features=sparse)
    elif model_name == "DeepFM":
        params.update(deep_features=dense, fm_features=sparse)
    elif model_name in ("DCN", "DCNv2", "EDCN"):
        params.update(features=features, n_cross_layers=2)
    elif model_name == "AFM":
        params = dict(fm_features=sparse, embed_dim=4, t=4)
    elif model_name == "FiBiNet":
        params.update(features=features, reduction_ratio=2)
    elif model_name == "AutoInt":
        params.update(dense_features=dense, sparse_features=sparse, num_layers=1, num_heads=2)
    elif model_name in ("DeepFFM", "FatDeepFFM"):
        params.update(linear_features=[SparseFeature(f.name, 32, 1) for f in sparse], cross_features=[SparseFeature(f.name, 32 * len(sparse), 4) for f in sparse], embed_dim=4)
        if model_name == "FatDeepFFM":
            params["reduction_ratio"] = 1
    elif model_name in ("BST", "DIN", "DIEN"):
        target = [SparseFeature("item", 32, 4, padding_idx=0)]
        history = [SequenceFeature("history", 32, 4, pooling="concat", shared_with="item", padding_idx=0)]
        data["item"] = rng.integers(1, 32, size=64)
        data["history"] = rng.integers(1, 32, size=(64, 4))
        data["history"][:, -1] = 0
        params.update(features=features, history_features=history, target_features=target)
        if model_name == "BST":
            params.update(nhead=2, max_seq_len=5)
        elif model_name == "DIN":
            params["attention_mlp_params"] = {"dims": [8]}
        else:
            data["neg_history"] = rng.integers(1, 32, size=(64, 4))
            data["neg_history"][:, -1] = 0
            params["neg_history_features"] = [SequenceFeature("neg_history", 32, 4, pooling="concat", shared_with="item", padding_idx=0)]
    else:
        raise AssertionError(f"Add an explicit training fixture for new model {model_name}")

    def subset(start, end):
        return {name: values[start:end] for name, values in data.items()}

    train, val, test = DataGenerator(subset(0, 32), labels[:32]).generate_dataloader(x_val=subset(32, 48), y_val=labels[32:48], x_test=subset(48, 64), y_test=labels[48:], batch_size=8)
    model = getattr(ranking, model_name)(**params)
    initial = {name: value.clone() for name, value in model.named_parameters()}
    trainer = CTRTrainer(model, n_epoch=1, device="cpu", model_path=str(tmp_path), loss_mode=model_name != "DIEN")
    trainer.fit(train, val)
    auc = trainer.evaluate(model, test)
    assert np.isfinite(auc) and 0 <= auc <= 1
    assert any(not torch.equal(initial[name], value) for name, value in model.named_parameters())
    batch, _ = next(iter(test))
    model.eval()
    with torch.no_grad():
        prediction = model(batch)
        if isinstance(prediction, tuple):
            prediction = prediction[0]
    assert prediction.shape == (8,) and torch.isfinite(prediction).all()
    # Reconstruct explicitly, then verify the actual file emitted by the Trainer.
    restored = getattr(ranking, model_name)(**copy.deepcopy(params))
    restored.load_state_dict(torch.load(tmp_path / "model.pth", map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        actual = restored(batch)
        if isinstance(actual, tuple):
            actual = actual[0]
    torch.testing.assert_close(actual, prediction)


@pytest.mark.parametrize("context", ["sparse", "dense", "mixed", "empty"])
def test_din_context_features(context, tmp_path):
    """DIN keeps dense context values and still accepts the original sparse inputs."""
    torch.manual_seed(2026)
    sparse = [SparseFeature("user_id", vocab_size=10, embed_dim=4)]
    dense = [DenseFeature("price"), DenseFeature("profile", embed_dim=3)]
    features = {"sparse": sparse, "dense": dense, "mixed": sparse + dense, "empty": []}[context]
    target = [SparseFeature("item_id", vocab_size=20, embed_dim=4)]
    history = [SequenceFeature("history", vocab_size=20, embed_dim=4, pooling="concat", shared_with="item_id")]
    model = ranking.DIN(features, history, target, mlp_params={"dims": [8]}, attention_mlp_params={"dims": [8]})
    x = {
        "user_id": torch.arange(8),
        "price": torch.randn(8,
                             requires_grad=True),
        "profile": torch.randn(8,
                               3,
                               requires_grad=True),
        "item_id": torch.arange(1,
                                9),
        "history": torch.randint(1,
                                 20,
                                 (8,
                                  3)),
    }
    y = torch.tensor([0, 1] * 4, dtype=torch.float32)
    before = model.mlp.mlp[0].weight.detach().clone()
    trainer = CTRTrainer(model, n_epoch=1, device="cpu", model_path=str(tmp_path))
    trainer.fit([(x, y)])
    assert not torch.equal(before, model.mlp.mlp[0].weight)
    if context in {"dense", "mixed"}:
        assert x["price"].grad.abs().sum() > 0
        assert x["profile"].grad.abs().sum() > 0
    model.eval()
    with torch.no_grad():
        expected = model(x)
        assert expected.shape == (8,)
        assert torch.isfinite(expected).all()
        model.mlp.mlp[0].weight.add_(1)
        model.load_state_dict(torch.load(tmp_path / "model.pth", weights_only=True))
        torch.testing.assert_close(model(x), expected)
        assert model({name: values[:1] for name, values in x.items()}).shape == (1,)
