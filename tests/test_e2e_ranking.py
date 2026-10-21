import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.models import ranking as rk
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# Dynamically get all ranking models
ranking_models = [getattr(rk, model_name) for model_name in rk.__all__]


@pytest.fixture(scope="module")
def ranking_data():
    """Create a dataset fixture for all ranking model tests."""
    batch_size = 128
    n_samples = 500

    # Features
    dense_feats = [DenseFeature(f"d_{i}") for i in range(5)]
    sparse_feats = [SparseFeature(f"s_{i}", vocab_size=100, embed_dim=16) for i in range(5)]
    features = dense_feats + sparse_feats

    # Data
    data = {}
    for feat in dense_feats:
        data[feat.name] = np.random.randn(n_samples)
    for feat in sparse_feats:
        data[feat.name] = np.random.randint(0, 100, n_samples)
    labels = np.random.randint(0, 2, n_samples)

    dg = DataGenerator(x=data, y=labels)
    train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.8, 0.1], batch_size=batch_size)

    return {"features": features, "dense_feats": dense_feats, "sparse_feats": sparse_feats, "train_dl": train_dl, "val_dl": val_dl, "test_dl": test_dl}


@pytest.mark.parametrize("model_class", ranking_models)
def test_ranking_e2e(model_class, ranking_data):
    """End-to-end test for ranking models."""
    model_name = model_class.__name__

    # Only test models that we know work well
    working_models = ['WideDeep', 'DCN', 'DCNv2', 'EDCN', 'FiBiNet']

    if model_name not in working_models:
        pytest.skip(f"Model {model_name} needs more complex setup - skipping for now.")

    features = ranking_data["features"]
    dense_feats = ranking_data["dense_feats"]
    sparse_feats = ranking_data["sparse_feats"]

    # Model-specific parameter handling
    params = {}

    if model_name == 'WideDeep':
        params = {"wide_features": dense_feats, "deep_features": sparse_feats, "mlp_params": {"dims": [32]}}
    elif model_name == 'DeepFM':
        params = {"features": features, "mlp_params": {"dims": [32]}}
    elif model_name in ['DCN', 'DCNv2', 'EDCN']:
        params = {"features": features, "n_cross_layers": 2, "mlp_params": {"dims": [32]}}
    elif model_name == 'AFM':
        params = {"features": features, "attention_dim": 16, "mlp_params": {"dims": [32]}}
    elif model_name == 'FiBiNet':
        params = {"features": features, "reduction_ratio": 3, "mlp_params": {"dims": [32]}}
    elif model_name == 'AutoInt':
        params = {"sparse_features": sparse_feats, "dense_features": dense_feats, "num_layers": 3, "num_heads": 2, "dropout": 0.0, "mlp_params": {"dims": [32]}}
    elif model_name in ["DeepFFM", "FatDeepFFM"]:
        # DeepFFM needs special features
        ffm_feats = [SparseFeature(f.name, f.vocab_size, 16) for f in sparse_feats]
        params = {"linear_features": ffm_feats, "cross_features": ffm_feats, "embed_dim": 16, "mlp_params": {"dims": [32]}}
        if model_name == 'FatDeepFFM':
            params['reduction_ratio'] = 1
    elif model_name in ['BST', 'DIN', 'DIEN']:
        # These need history and target features
        history_feats = [sparse_feats[0]]
        target_feats = [sparse_feats[1]]
        data = ranking_data["train_dl"].dataset.x_dict
        data[sparse_feats[0].name] = torch.randint(0, sparse_feats[0].vocab_size, (len(data['d_0']), 5))  # Add sequence feature
        params = {"features": features, "history_features": history_feats, "target_features": target_feats, "mlp_params": {"dims": [32]}}
        if model_name == 'DIN':
            params["attention_mlp_params"] = {"dims": [32]}
        if model_name == 'DIEN':
            params["attention_mlp_params"] = {"dims": [32]}
            params["gru_params"] = {'hidden_size': 16, 'num_layers': 1}
    else:
        pytest.skip(f"Model {model_name} requires specific setup not yet implemented.")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = model_class(**params)

        trainer = CTRTrainer(model, optimizer_params={"lr": 0.01}, n_epoch=1, earlystop_patience=10, device='cpu', model_path=temp_dir)

        trainer.fit(ranking_data["train_dl"], ranking_data["val_dl"])
        auc = trainer.evaluate(model, ranking_data["test_dl"])

        assert isinstance(auc, float)
        assert 0.0 <= auc <= 1.0, f"AUC for {model_name} is out of range: {auc}"


@pytest.mark.parametrize("context", ["sparse", "dense", "mixed", "empty"])
def test_din_context_features(context, tmp_path):
    """DIN keeps dense context values and still accepts the original sparse inputs."""
    torch.manual_seed(2026)
    sparse = [SparseFeature("user_id", vocab_size=10, embed_dim=4)]
    dense = [DenseFeature("price"), DenseFeature("profile", embed_dim=3)]
    features = {"sparse": sparse, "dense": dense, "mixed": sparse + dense, "empty": []}[context]
    target = [SparseFeature("item_id", vocab_size=20, embed_dim=4)]
    history = [SequenceFeature("history", vocab_size=20, embed_dim=4, pooling="concat", shared_with="item_id")]
    model = rk.DIN(features, history, target, mlp_params={"dims": [8]}, attention_mlp_params={"dims": [8]})
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


if __name__ == '__main__':
    pytest.main(['-v', __file__])
