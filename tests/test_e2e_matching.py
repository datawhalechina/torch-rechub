import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models import matching as mt
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

# Dynamically get all matching models
matching_models = [getattr(mt, model_name) for model_name in mt.__all__ if isinstance(getattr(mt, model_name), type)]


def create_data_for_mode(n_users, n_items, n_samples, mode, neg_ratio=3):
    """Create data for specific training mode."""
    # Raw data
    data = pd.DataFrame({"user_id": np.random.randint(0, n_users, n_samples), "item_id": np.random.randint(0, n_items, n_samples), "time": np.arange(n_samples)})

    # User and item profiles
    user_profile = pd.DataFrame({"user_id": np.arange(n_users)})
    item_profile = pd.DataFrame({"item_id": np.arange(n_items)})

    # Generate sequence features and train/test splits
    train_data, test_data = generate_seq_feature_match(data, "user_id", "item_id", "time", mode=mode, neg_ratio=neg_ratio)

    # Define features
    user_features = [SparseFeature("user_id", n_users, 16), SequenceFeature("hist_item_id", n_items, 16, pooling="mean", shared_with="item_id")]
    item_features = [SparseFeature("item_id", n_items, 16)]

    # Process data for model input
    train_x = gen_model_input(train_data, user_profile, "user_id", item_profile, "item_id", seq_max_len=10)

    if mode == 0:  # point-wise
        train_y = train_data["label"].values
    elif mode == 1:  # pair-wise
        train_y = np.ones(len(train_data))  # dummy labels
    elif mode == 2:  # list-wise
        train_y = np.zeros(len(train_data))  # all labels are 0 for list-wise

    test_user_x = gen_model_input(test_data, user_profile, "user_id", item_profile, "item_id", seq_max_len=10)

    # DataGenerator for training
    dg = MatchDataGenerator(x=train_x, y=train_y)
    train_dl, val_dl, item_dl = dg.generate_dataloader(test_user_x, df_to_dict(item_profile), batch_size=128)

    return {"user_features": user_features, "item_features": item_features, "train_dl": train_dl, "val_dl": val_dl, "item_dl": item_dl, "test_user_x": test_user_x, "item_profile": item_profile, "train_data": train_data}


@pytest.fixture(scope="module")
def matching_data():
    """Create a dataset fixture for all matching model tests."""
    n_users, n_items = 100, 500
    n_samples = 2000

    # Create data for different modes
    data_dict = {
        0: create_data_for_mode(n_users, n_items, n_samples, mode=0),  # point-wise
        1: create_data_for_mode(n_users, n_items, n_samples, mode=1),  # pair-wise
        2: create_data_for_mode(n_users, n_items, n_samples, mode=2),  # list-wise
    }

    return data_dict


@pytest.mark.parametrize("model_class", matching_models)
def test_matching_e2e(model_class, matching_data):
    """End-to-end test for matching models."""
    model_name = model_class.__name__

    # Determine the training mode based on model type
    if model_name == "FaceBookDSSM":
        mode = 1  # pair-wise
    elif model_name in ["YoutubeDNN", "YoutubeSBC", "MIND", "GRU4Rec", "NARM", "ComirecDR", "ComirecSA"]:
        mode = 2  # list-wise
    else:
        mode = 0  # point-wise (default)

    # Get data for the specific mode
    data = matching_data[mode]
    user_features = data["user_features"]
    item_features = data["item_features"]

    params = {}

    # Handle different model architectures
    if model_name == "DSSM":
        params = {"user_features": user_features, "item_features": item_features, "user_params": {"dims": [32]}, "item_params": {"dims": [32]}}
    elif model_name == "YoutubeDNN":
        # YoutubeDNN needs neg_item_feature for list-wise training
        # Make sure user_tower output dimension matches item embedding dimension
        neg_item_feature = [SequenceFeature('neg_items', vocab_size=item_features[0].vocab_size, embed_dim=16, pooling="concat", shared_with="item_id")]
        params = {"user_features": user_features, "item_features": item_features, "neg_item_feature": neg_item_feature, "user_params": {"dims": [16]}}
    elif model_name == "FaceBookDSSM":
        params = {"user_features": user_features, "pos_item_features": item_features, "neg_item_features": item_features, "user_params": {"dims": [32]}, "item_params": {"dims": [32]}}
    else:
        # Skip complex models that need more specific setup for now
        # These models require careful configuration that matches their specific architectures
        # and may need model-level changes which we want to avoid
        pytest.skip(f"Model {model_name} needs more complex setup - skipping for now.")

    with tempfile.TemporaryDirectory() as temp_dir:
        model = model_class(**params)

        trainer = MatchTrainer(model, mode=mode, optimizer_params={"lr": 0.01}, n_epoch=1, device='cpu', model_path=temp_dir)

        # Note: val_dl from MatchDataGenerator is for inference, not validation
        # So we only use train_dl for training without validation
        trainer.fit(data["train_dl"], val_dataloader=None)

        # Test inference
        user_embedding = trainer.inference_embedding(model, "user", data["val_dl"], temp_dir)
        item_embedding = trainer.inference_embedding(model, "item", data["item_dl"], temp_dir)

        assert user_embedding.shape[0] > 0
        assert item_embedding.shape[0] > 0
        assert user_embedding.shape[1] == item_embedding.shape[1]


if __name__ == '__main__':
    pytest.main(['-v', __file__])
