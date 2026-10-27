import copy

import numpy as np
import pytest
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models import multi_task
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


@pytest.mark.parametrize("model_name", multi_task.__all__)
def test_multitask_e2e(model_name, tmp_path):
    """All public MTL models complete real fit/evaluate/save; no unconditional skip."""
    torch.manual_seed(2022)
    rng = np.random.default_rng(2022)
    features = [DenseFeature("d1"), SparseFeature("s1", 32, 4), SparseFeature("s2", 32, 4)]
    x = {"d1": rng.normal(size=64).astype("float32"), "s1": rng.integers(1, 32, 64), "s2": rng.integers(1, 32, 64)}
    ctr = np.tile([0, 1], 32)
    cvr = np.tile([0, 0, 1, 1], 16)
    ys = np.stack([ctr, cvr], axis=1)
    task_types = ["classification"] * 2
    towers = [{"dims": [8]}, {"dims": [8]}]
    params = dict(features=features, task_types=task_types, tower_params_list=towers)
    if model_name == "SharedBottom":
        params["bottom_params"] = {"dims": [8]}
    elif model_name == "MMOE":
        params.update(n_expert=2, expert_params={"dims": [8]})
    elif model_name == "PLE":
        params.update(n_level=1, n_expert_specific=1, n_expert_shared=1, expert_params={"dims": [8]})
    elif model_name == "AITM":
        params = dict(features=features, n_task=2, bottom_params={"dims": [8]}, tower_params_list=towers)
    elif model_name == "ESMM":
        params = dict(user_features=[features[1]], item_features=[features[2]], cvr_params={"dims": [8]}, ctr_params={"dims": [8]})
        # ESMM outputs [CVR, CTR, CTCVR]; Trainer optimizes CTR and CTCVR.
        ys = np.stack([cvr, ctr, cvr * ctr], axis=1)
        task_types = ["classification"] * 3
    else:
        raise AssertionError(f"Add an explicit training fixture for new model {model_name}")
    train, val, _ = DataGenerator({k: v[:32] for k, v in x.items()}, ys[:32]).generate_dataloader(x_val={k: v[32:] for k, v in x.items()}, y_val=ys[32:], batch_size=8)
    model = getattr(multi_task, model_name)(**params)
    initial = {name: value.clone() for name, value in model.named_parameters()}
    trainer = MTLTrainer(model, task_types, n_epoch=1, device="cpu", model_path=str(tmp_path))
    trainer.fit(train, val)
    scores = trainer.evaluate(model, val)
    assert len(scores) == len(task_types) and all(np.isfinite(score) and 0 <= score <= 1 for score in scores)
    assert any(not torch.equal(initial[name], value) for name, value in model.named_parameters())
    batch, _ = next(iter(val))
    with torch.no_grad():
        expected = model(batch)
    restored = getattr(multi_task, model_name)(**copy.deepcopy(params))
    restored.load_state_dict(torch.load(tmp_path / "model_base_0.pth", map_location="cpu", weights_only=True))
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(batch), expected)
