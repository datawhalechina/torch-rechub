import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gather_inbatch_logits, gen_model_input, generate_seq_feature_match, inbatch_negative_sampling


def test_inbatch_negative_sampling_random_and_uniform():
    scores = torch.zeros((4, 4))
    neg_idx = inbatch_negative_sampling(scores, neg_ratio=2, generator=torch.Generator().manual_seed(0))
    logits = gather_inbatch_logits(scores, neg_idx)
    assert logits.shape == (4, 3)
    assert neg_idx.shape == (4, 2)
    for row, sampled in enumerate(neg_idx):
        assert row not in sampled.tolist()

    # Different seed should give different permutations to ensure randomness
    neg_idx_second = inbatch_negative_sampling(scores, neg_ratio=2, generator=torch.Generator().manual_seed(1))
    assert not torch.equal(neg_idx, neg_idx_second)


def test_inbatch_negative_sampling_hard_negative():
    scores = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 0.0]])
    neg_idx = inbatch_negative_sampling(scores, neg_ratio=1, hard_negative=True)
    # highest non-diagonal scores for each row
    assert torch.equal(neg_idx.squeeze(1), torch.tensor([2, 2, 1]))


def _build_small_match_dataloader():
    np.random.seed(42)
    n_users, n_items, n_samples = 12, 24, 80
    data = pd.DataFrame({
        "user_id": np.random.randint(0,
                                     n_users,
                                     n_samples),
        "item_id": np.random.randint(0,
                                     n_items,
                                     n_samples),
        "time": np.arange(n_samples),
    })
    user_profile = pd.DataFrame({"user_id": np.arange(n_users)})
    item_profile = pd.DataFrame({"item_id": np.arange(n_items)})

    df_train, _ = generate_seq_feature_match(data, "user_id", "item_id", "time", mode=0, neg_ratio=0)
    x_train = gen_model_input(df_train, user_profile, "user_id", item_profile, "item_id", seq_max_len=8)
    # labels are unused in in-batch mode; keep zero array for shape alignment
    y_train = np.zeros(len(df_train))

    user_features = [
        SparseFeature("user_id",
                      n_users,
                      embed_dim=8),
        SequenceFeature("hist_item_id",
                        n_items,
                        embed_dim=8,
                        pooling="mean",
                        shared_with="item_id"),
    ]
    item_features = [SparseFeature("item_id", n_items, embed_dim=8)]

    dg = MatchDataGenerator(x_train, y_train)
    train_dl, _, _ = dg.generate_dataloader(x_train, df_to_dict(item_profile), batch_size=8, num_workers=0)

    model = DSSM(user_features, item_features, user_params={"dims": [16]}, item_params={"dims": [16]})
    return train_dl, model


def test_match_trainer_inbatch_flow_runs_and_updates():
    train_dl, model = _build_small_match_dataloader()

    trainer = MatchTrainer(model, mode=0, in_batch_neg=True, in_batch_neg_ratio=3, sampler_seed=2, n_epoch=1, device="cpu")
    trainer.train_one_epoch(train_dl, log_interval=100)

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


@pytest.mark.parametrize("mode, temperature", [(0, 0.05), (2, 1.0)])
def test_full_inbatch_matches_reference_and_updates_both_towers(monkeypatch, mode, temperature):
    train_dl, model = _build_small_match_dataloader()
    model.temperature = temperature
    x, y = next(iter(train_dl))
    trainer = MatchTrainer(model, mode=mode, in_batch_neg=True, optimizer_fn=torch.optim.SGD, optimizer_params={"lr": 0.01})

    def unexpected_sampling(*args, **kwargs):
        raise AssertionError("Full-batch training must not sample or gather negative indices")

    monkeypatch.setattr("torch_rechub.trainers.match_trainer.inbatch_negative_sampling", unexpected_sampling)
    monkeypatch.setattr("torch_rechub.trainers.match_trainer.gather_inbatch_logits", unexpected_sampling)
    with torch.no_grad():
        user_emb = F.normalize(model.user_tower(x), dim=-1)
        item_emb = F.normalize(model.item_tower(x), dim=-1)
        logits = user_emb @ item_emb.T / temperature
        expected_loss = F.cross_entropy(logits, torch.arange(len(y))).item()
    user_before = [p.detach().clone() for p in model.user_mlp.parameters()]
    item_before = [p.detach().clone() for p in model.item_mlp.parameters()]
    singleton = ({key: value[:1] for key, value in x.items()}, y[:1])

    with pytest.warns(RuntimeWarning, match="Skipped 1"):
        actual_loss = trainer.train_one_epoch([(x, y), singleton])

    assert actual_loss == pytest.approx(expected_loss, rel=1e-5)
    assert any(not torch.equal(before, after) for before, after in zip(user_before, model.user_mlp.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(item_before, model.item_mlp.parameters()))
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_full_inbatch_rejects_an_epoch_without_negatives():
    train_dl, model = _build_small_match_dataloader()
    x, y = next(iter(train_dl))
    trainer = MatchTrainer(model, in_batch_neg=True)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    singleton = ({key: value[:1] for key, value in x.items()}, y[:1])

    with pytest.raises(ValueError, match="No usable in-batch"):
        trainer.train_one_epoch([singleton])
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())


def test_full_inbatch_validates_temperature_and_single_device():
    _, model = _build_small_match_dataloader()
    for temperature in (0.0, -0.1, float("nan"), float("inf")):
        model.temperature = temperature
        with pytest.raises(ValueError, match="temperature"):
            MatchTrainer(model, in_batch_neg=True)

    model.temperature = 1.0
    with pytest.raises(ValueError, match="single device"):
        MatchTrainer(model, in_batch_neg=True, gpus=[0, 1])

    # Point-wise DSSM still uses its original BCE path without temperature scaling.
    model.temperature = 0.0
    trainer = MatchTrainer(model)
    assert isinstance(trainer.criterion, torch.nn.BCELoss)
