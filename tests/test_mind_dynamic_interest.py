"""MIND dynamic interest count (issue #240).

The MIND paper (Li et al., 2019, https://arxiv.org/abs/1904.08030) sizes the
number of interest capsules per user by their history length::

    K'_u = max(1, min(K, floor(log2(|I_u|))))

i.e. a user with few interacted items gets fewer interest vectors. The default
``CapsuleNetwork`` instead emits a fixed ``interest_num`` capsules for everyone.
``dynamic_interest=True`` opts into the paper's per-user heuristic while leaving
the default (fixed) behaviour byte-identical.
"""

import math

import torch

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.basic.layers import CapsuleNetwork
from torch_rechub.models.matching import MIND


def _expected_k(hist_len, interest_num):
    return max(1, min(interest_num, int(math.floor(math.log2(max(hist_len, 1))))))


def _active_capsule_counts(interest_capsule, eps=1e-6):
    # An interest capsule is "active" when it carries a non-zero vector.
    return (interest_capsule.norm(dim=-1) > eps).sum(dim=1)


def test_capsule_dynamic_interest_scales_with_history_length():
    torch.manual_seed(0)
    seq_len, dim, interest_num = 16, 8, 4
    hist_lens = [1, 3, 8, 16]  # -> K'_u = [1, 1, 3, 4]

    capsule = CapsuleNetwork(dim, seq_len, bilinear_type=0, interest_num=interest_num, dynamic_interest=True)
    item_eb = torch.randn(len(hist_lens), seq_len, dim)
    mask = torch.zeros(len(hist_lens), seq_len, dtype=torch.long)
    for row, length in enumerate(hist_lens):
        mask[row, :length] = 1

    out = capsule(item_eb, mask)
    assert out.shape == (len(hist_lens), interest_num, dim)

    counts = _active_capsule_counts(out).tolist()
    expected = [_expected_k(length, interest_num) for length in hist_lens]
    assert counts == expected, f"active interests {counts} != paper K'_u {expected}"


def test_capsule_fixed_interest_is_default_and_unchanged():
    # Without the flag, every user keeps all ``interest_num`` capsules, even a
    # single-item history — the pre-#240 behaviour.
    torch.manual_seed(0)
    seq_len, dim, interest_num = 16, 8, 4

    capsule = CapsuleNetwork(dim, seq_len, bilinear_type=0, interest_num=interest_num)
    item_eb = torch.randn(2, seq_len, dim)
    mask = torch.zeros(2, seq_len, dtype=torch.long)
    mask[0, :1] = 1  # one-item user
    mask[1, :] = 1  # full-history user

    out = capsule(item_eb, mask)
    counts = _active_capsule_counts(out).tolist()
    assert counts == [interest_num, interest_num]


def _build_mind(max_length, interest_num, dynamic_interest):
    n_items = 50
    user_features = [SparseFeature("user_id", vocab_size=20, embed_dim=16)]
    history_features = [SequenceFeature("hist_movie_id", vocab_size=n_items, embed_dim=16, pooling="concat", shared_with="movie_id")]
    item_features = [SparseFeature("movie_id", vocab_size=n_items, embed_dim=16)]
    neg_item_feature = [SequenceFeature("neg_items", vocab_size=n_items, embed_dim=16, pooling="concat", shared_with="movie_id")]
    return MIND(user_features, history_features, item_features, neg_item_feature, max_length=max_length, interest_num=interest_num, dynamic_interest=dynamic_interest)


def test_mind_user_tower_emits_dynamic_interests():
    # End-to-end through MIND's user tower: the user embedding exposes exactly
    # K'_u non-zero interest vectors per user.
    torch.manual_seed(0)
    max_length, interest_num = 16, 4
    hist_lens = [1, 8, 16]  # -> K'_u = [1, 3, 4]

    model = _build_mind(max_length, interest_num, dynamic_interest=True)
    model.mode = "user"
    hist = torch.zeros(len(hist_lens), max_length, dtype=torch.long)
    for row, length in enumerate(hist_lens):
        hist[row, :length] = torch.arange(1, length + 1)
    x = {"user_id": torch.arange(len(hist_lens)), "hist_movie_id": hist}

    user_emb = model.user_tower(x)
    assert user_emb.shape == (len(hist_lens), interest_num, user_emb.shape[-1])

    counts = _active_capsule_counts(user_emb).tolist()
    expected = [_expected_k(length, interest_num) for length in hist_lens]
    assert counts == expected, f"user-tower interests {counts} != paper K'_u {expected}"
