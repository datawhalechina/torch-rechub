import os
import sys
from pathlib import Path

# Add the project root to Python path so we can import torch_rechub
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pytest
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.basic.layers import MLP, EmbeddingLayer


class TestMLP(object):

    def test_mlp(self):
        x = torch.randn(16, 10)
        mlp = MLP(input_dim=10, dims=[32, 16], output_layer=True)
        y = mlp(x)
        assert y.shape == (16, 1)


class TestEmbedding(object):

    def test_embedding_layer(self):
        features = [SparseFeature(name=f"f_{i}", vocab_size=10, embed_dim=4) for i in range(5)]
        embed = EmbeddingLayer(features)

        # Create input as a dictionary
        x = {f"f_{i}": torch.randint(0, 10, (16,)) for i in range(5)}
        y = embed(x, features)
        assert y.shape == (16, 5, 4)

    def test_embedding_with_dense(self):
        dense_features = [DenseFeature(name=f"dense_{i}") for i in range(3)]
        sparse_features = [SparseFeature(name=f"sparse_{i}", vocab_size=10, embed_dim=4) for i in range(2)]

        features = dense_features + sparse_features
        embed = EmbeddingLayer(features)

        # Create input as a dictionary with both dense and sparse features
        x = {}
        for i in range(3):
            x[f"dense_{i}"] = torch.randn(16)
        for i in range(2):
            x[f"sparse_{i}"] = torch.randint(0, 10, (16,))

        # Test with squeeze_dim=True to get flattened output
        y = embed(x, features, squeeze_dim=True)
        # Should have dense features (3) + sparse features flattened (2*4=8) = 11 total
        assert y.shape == (16, 11)


if __name__ == '__main__':
    pytest.main(['-v', 'tests/test_basic.py'])
