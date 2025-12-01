"""
Tests for ONNX export functionality.

This module tests the ONNX export capabilities for various recommendation models.
It includes validation helpers to verify that ONNX model outputs match PyTorch outputs.

Note: These tests require the optional 'onnx' dependencies to be installed.
Tests will be skipped if the onnx package is not available or fails to load.
"""

import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# Module-level ONNX availability check
# Skip all tests if onnx package is not available or fails to load
# ============================================================================

ONNX_AVAILABLE = False
ONNX_SKIP_REASON = "onnx package not installed"

try:
    import onnx

    # Try to actually use onnx to verify it loads correctly
    _ = onnx.__version__
    ONNX_AVAILABLE = True
except ImportError as e:
    ONNX_SKIP_REASON = f"onnx package not installed: {e}"
except OSError as e:
    ONNX_SKIP_REASON = f"onnx package failed to load: {e}"
except Exception as e:
    ONNX_SKIP_REASON = f"onnx package failed to initialize: {e}"

# Also check onnxruntime availability
ONNXRUNTIME_AVAILABLE = False
ONNXRUNTIME_SKIP_REASON = "onnxruntime package not installed"

try:
    import onnxruntime as ort
    _ = ort.__version__
    ONNXRUNTIME_AVAILABLE = True
except ImportError as e:
    ONNXRUNTIME_SKIP_REASON = f"onnxruntime package not installed: {e}"
except OSError as e:
    ONNXRUNTIME_SKIP_REASON = f"onnxruntime package failed to load: {e}"
except Exception as e:
    ONNXRUNTIME_SKIP_REASON = f"onnxruntime package failed to initialize: {e}"

# Skip entire module if onnx is not available
pytestmark = pytest.mark.skipif(not ONNX_AVAILABLE, reason=ONNX_SKIP_REASON)

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.utils.onnx_export import ONNXExporter, ONNXWrapper, extract_feature_info, generate_dummy_input

# ============================================================================
# Validation Helpers (in-test utilities as per requirements)
# ============================================================================


def validate_onnx_output(onnx_path: str, pytorch_model: torch.nn.Module, input_dict: dict, input_names: list, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    """Validate ONNX model output matches PyTorch model output.

    Args:
        onnx_path: Path to the ONNX model file.
        pytorch_model: Original PyTorch model.
        input_dict: Input dict for the model.
        input_names: Ordered list of input names.
        rtol: Relative tolerance for comparison.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    if not ONNXRUNTIME_AVAILABLE:
        pytest.skip(ONNXRUNTIME_SKIP_REASON)
        return True

    import onnxruntime as ort

    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_dict).numpy()

    # Get ONNX output
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    onnx_inputs = {name: input_dict[name].numpy() for name in input_names}
    onnx_output = sess.run(None, onnx_inputs)[0]

    # Compare
    return np.allclose(pytorch_output, onnx_output, rtol=rtol, atol=atol)


def check_onnx_model_valid(onnx_path: str) -> bool:
    """Check if ONNX model file is valid.

    Args:
        onnx_path: Path to the ONNX model file.

    Returns:
        True if model is valid.
    """
    if not ONNX_AVAILABLE:
        pytest.skip(ONNX_SKIP_REASON)
        return True

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return True


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_features():
    """Create simple features for testing."""
    sparse_feats = [
        SparseFeature("user_id",
                      vocab_size=100,
                      embed_dim=8),
        SparseFeature("item_id",
                      vocab_size=200,
                      embed_dim=8),
    ]
    dense_feats = [
        DenseFeature("price"),
        DenseFeature("rating"),
    ]
    return {"sparse": sparse_feats, "dense": dense_feats, "all": sparse_feats + dense_feats}


@pytest.fixture
def dual_tower_features():
    """Create features for dual-tower models."""
    user_feats = [
        SparseFeature("user_id",
                      vocab_size=100,
                      embed_dim=16),
        SparseFeature("gender",
                      vocab_size=3,
                      embed_dim=4),
    ]
    item_feats = [
        SparseFeature("item_id",
                      vocab_size=200,
                      embed_dim=16),
    ]
    return {"user": user_feats, "item": item_feats}


@pytest.fixture
def sequence_features():
    """Create features including sequence features."""
    sparse_feats = [
        SparseFeature("user_id",
                      vocab_size=100,
                      embed_dim=8),
    ]
    seq_feats = [
        SequenceFeature("hist_items",
                        vocab_size=200,
                        embed_dim=8,
                        pooling="mean"),
    ]
    return {"sparse": sparse_feats, "sequence": seq_feats, "all": sparse_feats + seq_feats}


# ============================================================================
# Unit Tests for Core Components
# ============================================================================


class TestExtractFeatureInfo:
    """Tests for extract_feature_info function."""

    def test_extract_from_simple_model(self, simple_features):
        """Test feature extraction from a model with 'features' attribute."""

        class MockModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.features = simple_features["all"]

        model = MockModel()
        info = extract_feature_info(model)

        assert len(info["features"]) == 4
        assert len(info["input_names"]) == 4
        assert "user_id" in info["input_names"]
        assert "item_id" in info["input_names"]

    def test_extract_from_dual_tower_model(self, dual_tower_features):
        """Test feature extraction from dual-tower model."""

        class MockDualTower(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.user_features = dual_tower_features["user"]
                self.item_features = dual_tower_features["item"]

        model = MockDualTower()
        info = extract_feature_info(model)

        assert len(info["user_features"]) == 2
        assert len(info["item_features"]) == 1
        assert "user_id" in [f.name for f in info["user_features"]]
        assert "item_id" in [f.name for f in info["item_features"]]

    def test_extract_deduplicates_features(self):
        """Test that duplicate features are removed."""
        feat1 = SparseFeature("same_name", vocab_size=100, embed_dim=8)
        feat2 = SparseFeature("same_name", vocab_size=100, embed_dim=8)

        class MockModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.deep_features = [feat1]
                self.fm_features = [feat2]

        model = MockModel()
        info = extract_feature_info(model)

        # Should deduplicate by name
        assert len(info["features"]) == 1


class TestGenerateDummyInput:
    """Tests for generate_dummy_input function."""

    def test_sparse_feature_shape(self):
        """Test dummy input shape for sparse features."""
        features = [SparseFeature("user_id", vocab_size=100, embed_dim=8)]
        dummy = generate_dummy_input(features, batch_size=4)

        assert len(dummy) == 1
        assert dummy[0].shape == (4,)
        assert dummy[0].max() < 100

    def test_dense_feature_shape(self):
        """Test dummy input shape for dense features."""
        features = [DenseFeature("price", embed_dim=1)]
        dummy = generate_dummy_input(features, batch_size=4)

        assert len(dummy) == 1
        assert dummy[0].shape == (4, 1)

    def test_sequence_feature_shape(self):
        """Test dummy input shape for sequence features."""
        features = [SequenceFeature("hist", vocab_size=100, embed_dim=8)]
        dummy = generate_dummy_input(features, batch_size=4, seq_length=10)

        assert len(dummy) == 1
        assert dummy[0].shape == (4, 10)
        assert dummy[0].max() < 100

    def test_mixed_features(self, simple_features):
        """Test dummy input for mixed feature types."""
        features = simple_features["all"]
        dummy = generate_dummy_input(features, batch_size=2)

        assert len(dummy) == 4  # 2 sparse + 2 dense


class TestONNXWrapper:
    """Tests for ONNXWrapper class."""

    def test_wrapper_converts_args_to_dict(self):
        """Test that wrapper correctly converts positional args to dict."""

        class SimpleModel(torch.nn.Module):

            def forward(self, x):
                return x["a"] + x["b"]

        model = SimpleModel()
        wrapper = ONNXWrapper(model, ["a", "b"])

        a = torch.tensor([1.0, 2.0])
        b = torch.tensor([3.0, 4.0])
        result = wrapper(a, b)

        expected = a + b
        assert torch.allclose(result, expected)

    def test_wrapper_validates_input_count(self):
        """Test that wrapper raises error for wrong number of inputs."""

        class SimpleModel(torch.nn.Module):

            def forward(self, x):
                return x["a"]

        model = SimpleModel()
        wrapper = ONNXWrapper(model, ["a", "b"])

        with pytest.raises(ValueError, match="Expected 2 inputs"):
            wrapper(torch.tensor([1.0]))

    def test_wrapper_mode_handling(self):
        """Test that wrapper handles mode for dual-tower models."""

        class DualTowerModel(torch.nn.Module):

            def __init__(self):
                super().__init__()
                self.mode = None

        model = DualTowerModel()
        wrapper = ONNXWrapper(model, ["a"], mode="user")

        assert model.mode == "user"

        wrapper.restore_mode()
        assert model.mode is None


# ============================================================================
# Integration Tests with Real Models
# ============================================================================


class TestONNXExportRankingModels:
    """Integration tests for ONNX export of ranking models."""

    def test_export_widedeep(self):
        """Test ONNX export for WideDeep model."""
        from torch_rechub.models.ranking import WideDeep
        from torch_rechub.trainers import CTRTrainer

        wide_feats = [DenseFeature("d1"), DenseFeature("d2")]
        deep_feats = [SparseFeature("s1", vocab_size=100, embed_dim=8)]

        model = WideDeep(wide_features=wide_feats, deep_features=deep_feats, mlp_params={"dims": [16, 8]})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CTRTrainer(model, n_epoch=1, device='cpu', model_path=tmpdir)
            onnx_path = os.path.join(tmpdir, "widedeep.onnx")

            result = trainer.export_onnx(onnx_path, verbose=False)

            assert result is True
            assert os.path.exists(onnx_path)
            assert check_onnx_model_valid(onnx_path)

    def test_export_dcn(self):
        """Test ONNX export for DCN model."""
        from torch_rechub.models.ranking import DCN
        from torch_rechub.trainers import CTRTrainer

        features = [
            DenseFeature("d1"),
            SparseFeature("s1",
                          vocab_size=100,
                          embed_dim=8),
            SparseFeature("s2",
                          vocab_size=50,
                          embed_dim=8),
        ]

        model = DCN(features=features, n_cross_layers=2, mlp_params={"dims": [16, 8]})

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = CTRTrainer(model, n_epoch=1, device='cpu', model_path=tmpdir)
            onnx_path = os.path.join(tmpdir, "dcn.onnx")

            result = trainer.export_onnx(onnx_path, verbose=False)

            assert result is True
            assert os.path.exists(onnx_path)
            assert check_onnx_model_valid(onnx_path)


class TestONNXExportMatchingModels:
    """Integration tests for ONNX export of matching models."""

    def test_export_dssm_user_tower(self):
        """Test ONNX export for DSSM user tower."""
        from torch_rechub.models.matching import DSSM
        from torch_rechub.trainers import MatchTrainer

        # Note: All features must have the same embed_dim for DSSM's EmbeddingLayer
        user_feats = [
            SparseFeature("user_id",
                          vocab_size=100,
                          embed_dim=16),
            SparseFeature("gender",
                          vocab_size=3,
                          embed_dim=16),
        ]
        item_feats = [
            SparseFeature("item_id",
                          vocab_size=200,
                          embed_dim=16),
        ]

        model = DSSM(user_features=user_feats, item_features=item_feats, user_params={"dims": [32, 16]}, item_params={"dims": [32, 16]}, temperature=0.02)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = MatchTrainer(model, mode=0, n_epoch=1, device='cpu', model_path=tmpdir)

            # Export user tower
            user_onnx_path = os.path.join(tmpdir, "user_tower.onnx")
            result = trainer.export_onnx(user_onnx_path, mode="user", verbose=False)

            assert result is True
            assert os.path.exists(user_onnx_path)
            assert check_onnx_model_valid(user_onnx_path)

    def test_export_dssm_item_tower(self):
        """Test ONNX export for DSSM item tower."""
        from torch_rechub.models.matching import DSSM
        from torch_rechub.trainers import MatchTrainer

        user_feats = [SparseFeature("user_id", vocab_size=100, embed_dim=16)]
        item_feats = [SparseFeature("item_id", vocab_size=200, embed_dim=16)]

        model = DSSM(user_features=user_feats, item_features=item_feats, user_params={"dims": [32, 16]}, item_params={"dims": [32, 16]}, temperature=0.02)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = MatchTrainer(model, mode=0, n_epoch=1, device='cpu', model_path=tmpdir)

            # Export item tower
            item_onnx_path = os.path.join(tmpdir, "item_tower.onnx")
            result = trainer.export_onnx(item_onnx_path, mode="item", verbose=False)

            assert result is True
            assert os.path.exists(item_onnx_path)
            assert check_onnx_model_valid(item_onnx_path)


class TestONNXExportMultiTaskModels:
    """Integration tests for ONNX export of multi-task models."""

    def test_export_mmoe(self):
        """Test ONNX export for MMOE model."""
        from torch_rechub.models.multi_task import MMOE
        from torch_rechub.trainers import MTLTrainer

        features = [
            DenseFeature("d1"),
            SparseFeature("s1",
                          vocab_size=100,
                          embed_dim=8),
        ]
        task_types = ["classification", "classification"]

        model = MMOE(features=features, task_types=task_types, n_expert=3, expert_params={"dims": [16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = MTLTrainer(model, task_types=task_types, n_epoch=1, device='cpu', model_path=tmpdir)
            onnx_path = os.path.join(tmpdir, "mmoe.onnx")

            result = trainer.export_onnx(onnx_path, verbose=False)

            assert result is True
            assert os.path.exists(onnx_path)
            assert check_onnx_model_valid(onnx_path)
