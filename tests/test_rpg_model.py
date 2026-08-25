"""Tests for RPG: the parallel semantic-ID model, its dataset and its OPQ tokenizer.

``RPGSeqDataset`` is pure Python and always runs. The model needs
``transformers`` (GPT-2 backbone) and the tokenizer needs ``faiss``, so those
groups are skipped when the optional dependencies are missing.
"""

import importlib.util
import math

import numpy as np
import pytest
import torch

from torch_rechub.utils.data import RPGSeqDataset

HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
HAS_FAISS = importlib.util.find_spec("faiss") is not None

N_ITEMS = 41
N_DIGIT = 4
CODEBOOK_SIZE = 8


def build_item_tokens(seed=0):
    """A ``(N_ITEMS, N_DIGIT)`` token table with row 0 reserved for PAD."""
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, CODEBOOK_SIZE, (N_ITEMS, N_DIGIT), generator=generator)
    tokens = codes + torch.arange(N_DIGIT) * CODEBOOK_SIZE + 1
    tokens[0] = 0
    return tokens


def build_model(**kwargs):
    from torch_rechub.models.generative import RPGModel

    torch.manual_seed(0)
    params = dict(codebook_size=CODEBOOK_SIZE, n_embd=32, n_layer=2, n_head=4, n_inner=64, max_seq_len=20)
    params.update(kwargs)
    return RPGModel(build_item_tokens(), **params)


def build_batch(model=None):
    input_ids = torch.tensor([[3, 7, 11, 5], [9, 2, 0, 0]])
    attention_mask = (input_ids != 0).long()
    labels = torch.tensor([[7, 11, 5, 8], [2, 4, -100, -100]])
    return input_ids, attention_mask, labels


# =========================================================
# RPGSeqDataset
# =========================================================
def test_leave_one_out_targets():
    seqs = {"u0": [1, 2, 3, 4, 5, 6]}
    assert RPGSeqDataset(seqs, max_seq_len=50, mode="valid")[0] == {"input_ids": [1, 2, 3, 4], "labels": [-100, -100, -100, 5], "target": 5}
    assert RPGSeqDataset(seqs, max_seq_len=50, mode="test")[0] == {"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, -100, -100, 6], "target": 6}


def test_short_train_sequence_supervises_every_position():
    data = RPGSeqDataset({"u0": [1, 2, 3, 4, 5, 6]}, max_seq_len=50, mode="train")
    # The training prefix drops the two held-out items, leaving [1, 2, 3, 4].
    assert len(data) == 1
    assert data[0] == {"input_ids": [1, 2, 3], "labels": [2, 3, 4], "target": 4}


def test_long_train_sequence_slides_a_window():
    data = RPGSeqDataset({"u0": list(range(1, 10))}, max_seq_len=2, mode="train")
    # Prefix is [1..7]; the first window supervises both positions and each
    # later target gets its own window with a label only at the last position.
    assert [s["input_ids"] for s in data] == [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
    assert [s["labels"] for s in data] == [[2, 3], [-100, 4], [-100, 5], [-100, 6], [-100, 7]]
    # Every item after the first is predicted exactly once.
    assert sorted(s["target"] for s in data) == [3, 4, 5, 6, 7]


def test_sequences_too_short_to_train_are_dropped():
    # [7] is the only training prefix item, so there is nothing to predict.
    assert len(RPGSeqDataset({"u0": [7, 8, 9]}, max_seq_len=50, mode="train")) == 0
    assert len(RPGSeqDataset({"u0": [7, 8, 9]}, max_seq_len=50, mode="test")) == 1


def test_dataset_accepts_a_list_of_sequences():
    assert len(RPGSeqDataset([[1, 2, 3, 4], [5, 6, 7, 8]], max_seq_len=50, mode="test")) == 2


def test_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        RPGSeqDataset({"u0": [1, 2, 3]}, mode="predict")


def test_collate_right_pads_and_masks():
    batch = RPGSeqDataset.collate_fn([{"input_ids": [1, 2, 3], "labels": [2, 3, 4], "target": 4}, {"input_ids": [5], "labels": [6], "target": 6}])
    assert batch["input_ids"].tolist() == [[1, 2, 3], [5, 0, 0]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    # Padded label slots are ignored, not treated as item 0.
    assert batch["labels"].tolist() == [[2, 3, 4], [6, -100, -100]]
    assert batch["seq_lens"].tolist() == [3, 1]
    assert batch["target"].tolist() == [4, 6]


# =========================================================
# Model
# =========================================================
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_resblock_starts_as_identity():
    from torch_rechub.models.generative.rpg import ResBlock

    x = torch.randn(4, 16)
    assert torch.allclose(ResBlock(16)(x), x)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_forward_shapes_and_untrained_loss():
    # Heads are identity at init and the codebooks are random, so at
    # temperature 1 every digit is close to uniform over its own codebook.
    model = build_model(temperature=1.0).eval()
    states, loss = model(*build_batch())
    assert states.shape == (2, 4, N_DIGIT, 32)
    assert loss.item() == pytest.approx(math.log(CODEBOOK_SIZE), abs=0.15)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_loss_is_none_without_labels_and_backpropagates_with_them():
    model = build_model()
    input_ids, attention_mask, labels = build_batch()
    assert model(input_ids, attention_mask)[1] is None

    _, loss = model(input_ids, attention_mask, labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.pred_heads[0].linear.weight.grad).all()


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_ignored_labels_do_not_contribute():
    """Masking the extra positions of a row must give the same loss as dropping them."""
    model = build_model().eval()
    input_ids = torch.tensor([[3, 7, 11, 5]])
    attention_mask = torch.ones_like(input_ids)

    _, full = model(input_ids, attention_mask, torch.tensor([[7, 11, 5, 8]]))
    _, masked = model(input_ids, attention_mask, torch.tensor([[7, -100, -100, 8]]))
    assert masked.item() != pytest.approx(full.item())

    _, twice = model(input_ids.repeat(2, 1), attention_mask.repeat(2, 1), torch.tensor([[7, -100, -100, 8], [7, -100, -100, 8]]))
    assert twice.item() == pytest.approx(masked.item(), abs=1e-5)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_next_token_logits_are_per_digit_log_probs():
    model = build_model().eval()
    input_ids, attention_mask, _ = build_batch()
    seq_lens = attention_mask.sum(1)

    states, _ = model(input_ids, attention_mask)
    token_logits = model.next_token_logits(states, seq_lens)
    assert token_logits.shape == (2, N_DIGIT * CODEBOOK_SIZE)
    per_digit = token_logits.view(2, N_DIGIT, CODEBOOK_SIZE).logsumexp(-1)
    assert torch.allclose(per_digit, torch.zeros_like(per_digit), atol=1e-5)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_item_score_is_the_mean_log_prob_of_its_tokens():
    model = build_model().eval()
    input_ids, attention_mask, _ = build_batch()
    states, _ = model(input_ids, attention_mask)
    token_logits = model.next_token_logits(states, attention_mask.sum(1))

    scores = model.score_all_items(token_logits)
    assert scores.shape == (2, N_ITEMS - 1)
    expected = token_logits[0, model.item_tokens[7] - 1].mean()
    assert scores[0, 6].item() == pytest.approx(expected.item(), abs=1e-5)

    # The per-row candidate path must agree with the exhaustive one.
    candidates = torch.tensor([[7, 3], [7, 3]])
    assert torch.allclose(model.score_candidates(token_logits, candidates)[:, 0], scores[:, 6], atol=1e-5)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_exhaustive_generate_returns_distinct_valid_items():
    model = build_model().eval()
    input_ids, attention_mask, _ = build_batch()
    preds = model.generate(input_ids, attention_mask, attention_mask.sum(1), topk=5)
    assert preds.shape == (2, 5)
    assert preds.min() >= 1 and preds.max() < N_ITEMS
    assert all(len(set(row.tolist())) == 5 for row in preds)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_decoding_graph_holds_valid_neighbours():
    model = build_model().eval()
    model.build_decoding_graph(n_edges=6, chunk_size=8)
    assert model.adjacency.shape == (N_ITEMS, 6)
    # Row 0 is a placeholder so that adjacency can be indexed by item id.
    assert model.adjacency[1:].min() >= 1
    assert model.adjacency[1:].max() < N_ITEMS


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_graph_generation_deduplicates_candidates():
    model = build_model().eval()
    model.build_decoding_graph(n_edges=6, chunk_size=8)
    input_ids, attention_mask, _ = build_batch()
    preds = model.generate(input_ids, attention_mask, attention_mask.sum(1), topk=5, use_graph=True, num_beams=10, propagation_steps=2)
    assert preds.shape == (2, 5)
    assert all(len(set(row.tolist())) == 5 for row in preds)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_graph_generation_requires_a_graph():
    model = build_model().eval()
    input_ids, attention_mask, _ = build_batch()
    with pytest.raises(RuntimeError, match="build_decoding_graph"):
        model.generate(input_ids, attention_mask, attention_mask.sum(1), use_graph=True)


# =========================================================
# Trainer metrics
# =========================================================
@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers is not installed")
def test_rank_metrics_match_hand_computed_values():
    from torch_rechub.trainers import RPGTrainer

    trainer = RPGTrainer(build_model(), topk=(1, 3))
    # Row 0 hits at rank 0, row 1 at rank 2, row 2 never hits.
    preds = torch.tensor([[5, 1, 2], [1, 2, 5], [1, 2, 3]])
    results = trainer._rank_metrics(preds, torch.tensor([5, 5, 9]))
    assert results["recall@1"] == 1.0
    assert results["recall@3"] == 2.0
    assert results["ndcg@1"] == pytest.approx(1.0)
    assert results["ndcg@3"] == pytest.approx(1.0 + 1.0 / math.log2(4))


# =========================================================
# OPQ tokenizer
# =========================================================
@pytest.mark.skipif(not HAS_FAISS, reason="faiss is not installed")
def test_opq_produces_offset_tokens_with_a_pad_row():
    from torch_rechub.utils.opq import OPQTokenizer

    embeddings = np.random.default_rng(0).standard_normal((400, 32)).astype(np.float32)
    tokenizer = OPQTokenizer(n_codebook=4, codebook_size=16, pca_dim=16).fit(embeddings)

    assert tokenizer.codes.shape == (400, 4)
    assert tokenizer.codes.min() >= 0 and tokenizer.codes.max() < 16
    assert tokenizer.vocab_size == 4 * 16 + 2

    tokens = tokenizer.item_tokens()
    assert tokens.shape == (401, 4)
    assert tokens[0].tolist() == [0, 0, 0, 0]
    # Digit j owns ids [j * 16 + 1, (j + 1) * 16].
    for digit in range(4):
        column = tokens[1:, digit]
        assert column.min() >= digit * 16 + 1
        assert column.max() <= (digit + 1) * 16


@pytest.mark.skipif(not HAS_FAISS, reason="faiss is not installed")
def test_opq_train_mask_still_encodes_every_item():
    from torch_rechub.utils.opq import OPQTokenizer

    # FAISS trains the OPQ rotation with a 256-centroid quantizer whatever
    # codebook_size is, so the masked set has to stay above that.
    embeddings = np.random.default_rng(1).standard_normal((700, 32)).astype(np.float32)
    mask = np.zeros(700, dtype=bool)
    mask[:500] = True
    tokenizer = OPQTokenizer(n_codebook=4, codebook_size=16, pca_dim=16).fit(embeddings, mask)
    assert len(tokenizer.codes) == 700


@pytest.mark.skipif(not HAS_FAISS, reason="faiss is not installed")
def test_opq_save_load_roundtrip(tmp_path):
    from torch_rechub.utils.opq import OPQTokenizer

    embeddings = np.random.default_rng(2).standard_normal((400, 32)).astype(np.float32)
    tokenizer = OPQTokenizer(n_codebook=4, codebook_size=16, pca_dim=16).fit(embeddings)
    path = tmp_path / "semantic_ids.json"
    tokenizer.save(str(path))

    reloaded = OPQTokenizer.load(str(path))
    assert torch.equal(reloaded.item_tokens(), tokenizer.item_tokens())
    assert reloaded.vocab_size == tokenizer.vocab_size


def test_opq_rejects_non_power_of_two_codebook():
    from torch_rechub.utils.opq import OPQTokenizer

    with pytest.raises(ValueError, match="power of two"):
        OPQTokenizer(codebook_size=100)
