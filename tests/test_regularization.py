"""
Test cases for RegularizationLoss functionality
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch_rechub.basic.loss_func import RegularizationLoss


@pytest.fixture(scope="module")
def simple_model():
    """Create a simple model fixture for testing."""

    class SimpleModel(nn.Module):
        """A simple model for testing"""

        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 10)
            self.linear1 = nn.Linear(10, 5)
            self.linear2 = nn.Linear(5, 1)

        def forward(self, x):
            x = self.embedding(x)
            x = x.view(x.size(0), -1)
            x = self.linear1(x)
            x = self.linear2(x)
            return x

    return SimpleModel()


def test_regularization_loss_no_penalty(simple_model):
    """Test that zero regularization returns zero loss"""
    model = simple_model
    reg_loss_fn = RegularizationLoss(embedding_l1=0.0, embedding_l2=0.0, dense_l1=0.0, dense_l2=0.0)

    loss = reg_loss_fn(model)
    assert loss == 0.0, "Zero regularization should return zero loss"


def test_regularization_loss_embedding_l2(simple_model):
    """Test embedding L2 regularization"""
    model = simple_model
    reg_loss_fn = RegularizationLoss(embedding_l1=0.0, embedding_l2=1.0, dense_l1=0.0, dense_l2=0.0)

    loss = reg_loss_fn(model)

    # Calculate expected loss manually using instance-based identification
    expected_loss = 0.0
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            for param in module.parameters():
                expected_loss += torch.sum(param**2).item()

    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_regularization_loss_embedding_l1(simple_model):
    """Test embedding L1 regularization"""
    model = simple_model
    reg_loss_fn = RegularizationLoss(embedding_l1=1.0, embedding_l2=0.0, dense_l1=0.0, dense_l2=0.0)

    loss = reg_loss_fn(model)

    # Calculate expected loss manually using instance-based identification
    expected_loss = 0.0
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            for param in module.parameters():
                expected_loss += torch.sum(torch.abs(param)).item()

    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_regularization_loss_dense_l2(simple_model):
    """Test dense L2 regularization"""
    model = simple_model
    reg_loss_fn = RegularizationLoss(embedding_l1=0.0, embedding_l2=0.0, dense_l1=0.0, dense_l2=1.0)

    loss = reg_loss_fn(model)

    # Calculate expected loss manually using instance-based identification
    embedding_params = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            for param in module.parameters():
                embedding_params.add(id(param))

    expected_loss = 0.0
    for param in model.parameters():
        if id(param) not in embedding_params:
            expected_loss += torch.sum(param**2).item()

    assert abs(loss - expected_loss) < 1e-6, f"Expected {expected_loss}, got {loss}"


def test_regularization_loss_combined(simple_model):
    """Test combined regularization"""
    model = simple_model
    reg_loss_fn = RegularizationLoss(embedding_l1=0.1, embedding_l2=0.2, dense_l1=0.3, dense_l2=0.4)

    loss = reg_loss_fn(model)

    # Calculate expected loss manually using instance-based identification
    embedding_params = set()
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            for param in module.parameters():
                embedding_params.add(id(param))

    expected_loss = 0.0
    for param in model.parameters():
        if id(param) in embedding_params:
            expected_loss += 0.1 * torch.sum(torch.abs(param)).item()
            expected_loss += 0.2 * torch.sum(param**2).item()
        else:
            expected_loss += 0.3 * torch.sum(torch.abs(param)).item()
            expected_loss += 0.4 * torch.sum(param**2).item()

    assert abs(loss - expected_loss) < 1e-4, f"Expected {expected_loss}, got {loss}"


def test_regularization_loss_backward(simple_model):
    """Test that regularization loss can be backpropagated"""
    model = simple_model
    reg_loss_fn = RegularizationLoss(embedding_l2=0.01, dense_l2=0.01)

    # Create dummy input and target
    x = torch.randint(0, 100, (4,))
    y = torch.randn(4, 1)

    # Forward pass
    y_pred = model(x)
    task_loss = nn.MSELoss()(y_pred, y)
    reg_loss = reg_loss_fn(model)
    total_loss = task_loss + reg_loss

    # Backward pass - should not raise any errors
    total_loss.backward()

    # Check that gradients were computed
    for param in model.parameters():
        assert param.grad is not None, "Gradients should be computed"


def test_regularization_loss_parameter_identification():
    """Test that embedding and dense parameters are correctly identified"""

    class TestModel(nn.Module):

        def __init__(self):
            super().__init__()
            self.user_embedding = nn.Embedding(100, 10)
            self.item_embed = nn.Embedding(50, 10)  # Should be recognized as embedding by instance type
            self.dense_layer = nn.Linear(20, 1)
            self.other_weight = nn.Parameter(torch.randn(5, 5))

    model = TestModel()
    reg_loss_fn = RegularizationLoss(embedding_l2=1.0, dense_l2=0.0)

    loss = reg_loss_fn(model)

    # Only embedding parameters should contribute (identified by instance type)
    expected_loss = 0.0
    for module in model.modules():
        if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
            for param in module.parameters():
                expected_loss += torch.sum(param**2).item()

    assert abs(loss - expected_loss) < 1e-6


if __name__ == '__main__':
    pytest.main(['-v', __file__])
