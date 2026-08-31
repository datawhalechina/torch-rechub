import pytest
import torch

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.basic.layers import EmbeddingLayer


def test_concat_sequence_rejects_mixed_features():
    features = [
        SparseFeature("user_id",
                      vocab_size=10,
                      embed_dim=4),
        SequenceFeature("history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="concat"),
    ]
    inputs = {
        "user_id": torch.tensor([1,
                                 2]),
        "history": torch.tensor([[1,
                                  2,
                                  0],
                                 [3,
                                  0,
                                  0]]),
    }

    with pytest.raises(ValueError, match='pooling="concat".*cannot be mixed'):
        EmbeddingLayer(features)(inputs, features)


def test_concat_sequence_rejects_reduced_sequence():
    features = [
        SequenceFeature("raw_history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="concat"),
        SequenceFeature("pooled_history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="mean"),
    ]
    inputs = {
        "raw_history": torch.tensor([[1,
                                      2,
                                      0],
                                     [3,
                                      0,
                                      0]]),
        "pooled_history": torch.tensor([[4,
                                         5,
                                         0],
                                        [6,
                                         7,
                                         8]]),
    }

    with pytest.raises(ValueError, match="incompatible features.*pooled_history"):
        EmbeddingLayer(features)(inputs, features)


def test_concat_sequence_preserves_sequence_dimension():
    features = [
        SequenceFeature("view_history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="concat"),
        SequenceFeature("buy_history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="concat"),
    ]
    inputs = {
        "view_history": torch.tensor([[1,
                                       2,
                                       0],
                                      [3,
                                       0,
                                       0]]),
        "buy_history": torch.tensor([[4,
                                      5,
                                      0],
                                     [6,
                                      7,
                                      8]]),
    }

    output = EmbeddingLayer(features)(inputs, features)

    assert output.shape == (2, 2, 3, 4)


def test_concat_sequence_can_flatten_with_dense_feature():
    features = [
        SequenceFeature("history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="concat"),
        DenseFeature("score"),
    ]
    inputs = {
        "history": torch.tensor([[1,
                                  2,
                                  0],
                                 [3,
                                  0,
                                  0]]),
        "score": torch.tensor([0.2,
                               0.8]),
    }

    output = EmbeddingLayer(features)(inputs, features, squeeze_dim=True)

    assert output.shape == (2, 13)


def test_mean_sequence_can_mix_with_sparse_feature():
    features = [
        SparseFeature("user_id",
                      vocab_size=10,
                      embed_dim=4),
        SequenceFeature("history",
                        vocab_size=10,
                        embed_dim=4,
                        pooling="mean"),
    ]
    inputs = {
        "user_id": torch.tensor([1,
                                 2]),
        "history": torch.tensor([[1,
                                  2,
                                  0],
                                 [3,
                                  0,
                                  0]]),
    }

    output = EmbeddingLayer(features)(inputs, features)

    assert output.shape == (2, 2, 4)
