import importlib.util

import pytest

HAS_TORCH = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed")

if HAS_TORCH:
    import torch

    from torch_rechub.models.generative.hllm import HLLMModel


def test_hllm_accepts_token_indexed_item_embeddings():
    item_embeddings = torch.randn(5, 4) * 10
    item_embeddings[0].zero_()

    model = HLLMModel(
        item_embeddings=item_embeddings,
        vocab_size=5,
        d_model=4,
        n_heads=2,
        n_layers=1,
        max_seq_len=3,
    )

    assert model.item_embeddings.shape == (5, 4)
    assert torch.equal(model.item_embeddings[0], torch.zeros(4))
    assert model.temperature == pytest.approx(0.07)
    assert torch.allclose(model.item_embeddings[1:].norm(dim=-1), torch.ones(4), atol=1e-6)


def test_hllm_forward_returns_bounded_cosine_logits():
    item_embeddings = torch.randn(5, 4) * 10
    item_embeddings[0].zero_()
    model = HLLMModel(
        item_embeddings=item_embeddings,
        vocab_size=5,
        d_model=4,
        n_heads=2,
        n_layers=0,
        max_seq_len=3,
        dropout=0.0,
        use_rel_pos_bias=False,
        use_time_embedding=False,
    )

    logits = model(torch.tensor([[1, 2, 0]]))
    bound = 1.0 / model.temperature

    assert logits.shape == (1, 3, 5)
    assert logits.max().item() <= bound + 1e-5
    assert logits.min().item() >= -bound - 1e-5


def test_hllm_rejects_wrong_vocab_size():
    item_embeddings = torch.randn(4, 4)

    with pytest.raises(ValueError, match=r"item_embeddings\.shape\[0\]=4 != vocab_size=5"):
        HLLMModel(
            item_embeddings=item_embeddings,
            vocab_size=5,
            d_model=4,
            n_heads=2,
            n_layers=1,
            max_seq_len=3,
        )


def test_hllm_rejects_wrong_embedding_dim():
    item_embeddings = torch.randn(5, 3)

    with pytest.raises(ValueError, match=r"item_embeddings\.shape\[1\]=3 != d_model=4"):
        HLLMModel(
            item_embeddings=item_embeddings,
            vocab_size=5,
            d_model=4,
            n_heads=2,
            n_layers=1,
            max_seq_len=3,
        )
