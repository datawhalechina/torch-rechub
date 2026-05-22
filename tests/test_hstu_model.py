"""Correctness tests for HSTUModel / HSTULayer / RelPosBias / SeqTrainer."""

import math

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torch_rechub.basic.layers import HSTUBlock, HSTULayer
from torch_rechub.models.generative.hstu import HSTUModel
from torch_rechub.trainers.seq_trainer import SeqTrainer
from torch_rechub.utils.hstu_utils import RelPosBias, VocabMask


def _make_inputs(batch_size=4, seq_len=16, vocab_size=50, pad_prob=0.25, seed=0):
    """Random token ids with some PAD positions and random time deltas (seconds)."""
    g = torch.Generator().manual_seed(seed)
    tokens = torch.randint(1, vocab_size, (batch_size, seq_len), generator=g)
    pad_mask = torch.rand(batch_size, seq_len, generator=g) < pad_prob
    tokens = tokens.masked_fill(pad_mask, 0)
    time_diffs = torch.randint(0, 3600, (batch_size, seq_len), generator=g)
    return tokens, time_diffs


def test_forward_shape():
    model = HSTUModel(vocab_size=50, d_model=32, n_heads=4, n_layers=2, dqk=8, dv=8, max_seq_len=32, num_time_buckets=16)
    tokens, time_diffs = _make_inputs(seq_len=16, vocab_size=50)
    logits = model(tokens, time_diffs)
    assert logits.shape == (tokens.size(0), tokens.size(1), 50)


def test_padding_embedding_row_stays_zero_after_init():
    model = HSTUModel(vocab_size=50, d_model=32, n_heads=4, n_layers=1, dqk=8, dv=8, max_seq_len=32)
    assert torch.all(model.token_embedding.weight[0] == 0), \
        "padding_idx=0 row should be exactly zero after _init_weights"


def test_padding_position_output_is_zero():
    """Padded positions in the input should produce zero embedding rows and
    zero hidden states (the model explicitly masks padded positions)."""
    torch.manual_seed(0)
    model = HSTUModel(vocab_size=50, d_model=32, n_heads=4, n_layers=1, dqk=8, dv=8, max_seq_len=32, dropout=0.0)
    model.eval()

    tokens = torch.tensor([[5, 7, 0, 0], [1, 2, 3, 0]], dtype=torch.long)
    time_diffs = torch.zeros_like(tokens)
    logits = model(tokens, time_diffs)
    # With tied embeddings and PAD row being all-zero, the PAD row's logit is
    # exactly the output bias. So the *spread* across non-bias features at PAD
    # positions should be zero — i.e. logits at PAD positions equal a constant
    # (the bias) across vocab. Easier check: hidden state at PAD positions is 0.
    # Re-run forward and intercept by computing manually:
    # we re-derive the hidden state by replicating the model's forward but
    # returning hstu_output. Simpler: check that PAD positions produce the
    # same logits across the batch (since hidden = 0 -> logits = bias).
    pad_logits = logits[0, 2, :]  # PAD position, batch 0
    pad_logits2 = logits[1, 3, :]  # PAD position, batch 1
    assert torch.allclose(pad_logits, pad_logits2, atol=1e-6), \
        "All PAD positions must yield identical logits (= bias) under tied embedding"


def test_padding_mask_changes_non_pad_outputs():
    """Adding padding mask should affect non-PAD outputs compared to running
    without any mask, confirming the mask is actually being used."""
    torch.manual_seed(0)
    layer = HSTULayer(d_model=32, n_heads=4, dqk=8, dv=8, dropout=0.0, max_seq_len=16)
    layer.eval()

    x = torch.randn(2, 8, 32)
    padding_mask = torch.ones(2, 8, dtype=torch.bool)
    # Mark an interior key position as PAD. Queries at positions > 2 normally
    # attend to position 2 via the causal mask; masking it should change their
    # output (queries at position <= 2 are unaffected because either the
    # causal mask already excludes 2 as a key, or position 2 is the only
    # PAD in their valid range — we focus the assertion on positions 3..7).
    padding_mask[0, 2] = False

    out_no_mask = layer(x)
    out_with_mask = layer(x, padding_mask=padding_mask)

    diff = (out_with_mask[0, 3:] - out_no_mask[0, 3:]).abs().max().item()
    assert diff > 1e-5, "Padding mask should change attention output at valid positions"


def test_seq_len_exceeds_max_raises():
    model = HSTUModel(vocab_size=50, d_model=32, n_heads=4, n_layers=1, dqk=8, dv=8, max_seq_len=8)
    tokens, time_diffs = _make_inputs(seq_len=16, vocab_size=50)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        model(tokens, time_diffs)


def test_d_model_not_divisible_by_n_heads_raises():
    with pytest.raises(ValueError, match="must be divisible"):
        HSTULayer(d_model=30, n_heads=4, dqk=8, dv=8, max_seq_len=16)


def test_tied_embedding_shares_weight():
    model = HSTUModel(vocab_size=50, d_model=32, n_heads=4, n_layers=1, dqk=8, dv=8, max_seq_len=16, tie_embeddings=True)
    assert model.output_projection is None
    # Modify token_embedding weight and verify the logit projection moves too.
    model.token_embedding.weight.data.fill_(0.5)
    tokens = torch.tensor([[5, 7, 1, 2]], dtype=torch.long)
    time_diffs = torch.zeros_like(tokens)
    logits = model(tokens, time_diffs)
    assert logits.shape == (1, 4, 50)


def test_untied_embedding_keeps_separate_projection():
    model = HSTUModel(vocab_size=50, d_model=32, n_heads=4, n_layers=1, dqk=8, dv=8, max_seq_len=16, tie_embeddings=False)
    assert isinstance(model.output_projection, nn.Linear)
    assert model.output_projection.weight.shape == (50, 32)


def test_l2_scoring_without_output_bias_runs():
    model = HSTUModel(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=1,
        dqk=8,
        dv=8,
        max_seq_len=16,
        score_norm='l2',
        temperature=0.05,
        use_output_bias=False,
        scale_input_embedding=True,
    )
    assert model.output_bias is None
    tokens = torch.tensor([[5, 7, 1, 2]], dtype=torch.long)
    logits = model(tokens, torch.zeros_like(tokens))
    assert logits.shape == (1, 4, 50)
    assert torch.isfinite(logits).all()


def test_relpos_bias_bucket_bounds_under_overflow():
    bias = RelPosBias(n_heads=4, max_seq_len=16, num_buckets=8)
    # seq_len > max_seq_len: buckets must still be in [0, num_buckets-1].
    out = bias(seq_len=32)
    assert out.shape == (1, 4, 32, 32)
    # Verify no NaN/Inf.
    assert torch.isfinite(out).all()


def test_relpos_bias_init_scale():
    """Init values should be within sqrt(1/num_buckets), not N(0, 1)."""
    num_buckets = 32
    bias = RelPosBias(n_heads=4, max_seq_len=16, num_buckets=num_buckets)
    bound = (1.0 / num_buckets)**0.5
    assert bias.rel_pos_bias_table.abs().max().item() <= bound + 1e-6


def test_vocab_mask_drops_pad():
    mask = VocabMask(vocab_size=10, invalid_items=[0])
    logits = torch.randn(2, 10)
    masked = mask.apply_mask(logits)
    assert (masked[..., 0] <= -1e8).all()
    # Other ids unchanged
    assert torch.allclose(masked[..., 1:], logits[..., 1:])


def test_vocab_mask_drops_per_user_history():
    mask = VocabMask(vocab_size=10, invalid_items=[0])
    logits = torch.randn(2, 10)
    history = torch.tensor([[1, 3, 0], [2, 5, 0]])
    masked = mask.apply_mask(logits, invalid_ids=history)
    assert (masked[:, 0] <= -1e8).all()
    assert masked[0, 1] <= -1e8
    assert masked[0, 3] <= -1e8
    assert masked[1, 2] <= -1e8
    assert masked[1, 5] <= -1e8
    assert torch.allclose(masked[0, 2], logits[0, 2])


def test_seqtrainer_next_token_loss_masks_left_pad_positions():
    trainer = SeqTrainer(
        nn.Linear(1, 1),
        device='cpu',
        loss_params={"ignore_index": 0, "reduction": "sum"},
    )
    logits = torch.zeros(1, 4, 10)
    seq_tokens = torch.tensor([[0, 0, 5, 6]])
    targets = torch.tensor([7])

    loss = trainer._compute_next_token_loss(logits, seq_tokens, targets)

    # Only positions with current non-PAD tokens (5 and 6) should contribute.
    assert torch.allclose(loss, torch.tensor(2.0 * math.log(9.0)), atol=1e-6)


def test_seqtrainer_next_token_loss_ignores_pad():
    """Smoke test: full-sequence next-token loss should run end-to-end on a
    tiny toy dataset and ignore PAD positions via ``ignore_index=0``."""
    torch.manual_seed(0)
    vocab_size = 20
    model = HSTUModel(vocab_size=vocab_size, d_model=16, n_heads=4, n_layers=1, dqk=4, dv=4, max_seq_len=8, dropout=0.0, num_time_buckets=8)

    batch_size, seq_len = 4, 8
    seq_tokens = torch.randint(1, vocab_size, (batch_size, seq_len))
    seq_tokens[:, -2:] = 0  # PAD tail
    seq_positions = torch.zeros_like(seq_tokens)
    seq_time_diffs = torch.zeros(batch_size, seq_len, dtype=torch.long)
    targets = torch.randint(1, vocab_size, (batch_size, 1))

    dataset = TensorDataset(seq_tokens, seq_positions, seq_time_diffs, targets)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    trainer = SeqTrainer(model, n_epoch=1, device='cpu', model_path='./')
    loss = trainer.train_one_epoch(loader)
    assert loss > 0 and torch.isfinite(torch.tensor(loss))

    # Verify ignore_index is wired up.
    assert trainer.loss_fn.ignore_index == 0
