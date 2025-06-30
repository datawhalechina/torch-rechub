import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import pandas as pd
import tempfile

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models import matching as mt
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input

# Dynamically get all matching models
matching_models = [getattr(mt, model_name) for model_name in mt.__all__ if isinstance(getattr(mt, model_name), type)]

@pytest.fixture(scope="module")
def matching_data():
    """Create a dataset fixture for all matching model tests."""
    n_users, n_items = 100, 500
    n_samples = 2000
    
    # Raw data
    data = pd.DataFrame({
        "user_id": np.random.randint(0, n_users, n_samples),
        "item_id": np.random.randint(0, n_items, n_samples),
        "time": np.arange(n_samples)
    })
    
    # User and item profiles
    user_profile = pd.DataFrame({"user_id": np.arange(n_users)})
    item_profile = pd.DataFrame({"item_id": np.arange(n_items)})
    
    # Generate sequence features and train/test splits
    train_data, test_data = generate_seq_feature_match(data, "user_id", "item_id", "time", neg_ratio=3)
    
    # Define features
    user_features = [
        SparseFeature("user_id", n_users, 16),
        SequenceFeature("hist_item_id", n_items, 16, pooling="mean", shared_with="item_id")
    ]
    item_features = [SparseFeature("item_id", n_items, 16)]
    
    # Process data for model input
    train_x = gen_model_input(train_data, user_profile, "user_id", item_profile, "item_id", seq_max_len=10)
    train_y = train_data["label"].values
    
    test_user_x = gen_model_input(test_data, user_profile, "user_id", item_profile, "item_id", seq_max_len=10)
    
    # DataGenerator for training
    dg = MatchDataGenerator(x=train_x, y=train_y)
    train_dl, val_dl, item_dl = dg.generate_dataloader(test_user_x, df_to_dict(item_profile), batch_size=128)
    
    return {
        "user_features": user_features,
        "item_features": item_features,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "item_dl": item_dl,
        "test_user_x": test_user_x,
        "item_profile": item_profile
    }

@pytest.mark.parametrize("model_class", matching_models)
def test_matching_e2e(model_class, matching_data):
    """End-to-end test for matching models."""
    user_features = matching_data["user_features"]
    item_features = matching_data["item_features"]

    params = {}
    model_name = model_class.__name__
    
    # Handle different model architectures
    if model_name in ["DSSM", "YoutubeSBC", "MIND"]:
        params = {
            "user_features": user_features, "item_features": item_features,
            "user_params": {"dims": [32]}, "item_params": {"dims": [32]}
        }
        if model_name == "MIND":
            params["capsule_params"] = {"num_interest": 4, "dim_interest": 16}
    elif model_name == "YoutubeDNN":
        params = {
            "user_features": user_features, "item_features": item_features,
            "user_params": {"dims": [32]}
        }
    elif model_name == "FaceBookDSSM":
        params = {
            "user_features": user_features, "item_features": item_features, "neg_item_feature": item_features,
            "user_params": {"dims": [32]}, "item_params": {"dims": [32]}
        }
    elif model_name in ["GRU4Rec", "NARM", "SASRec", "STAMP"]:
        params = {"features": item_features}
        if model_name == "GRU4Rec":
            params["hidden_size"] = 16
        elif model_name == "NARM":
            params["hidden_size"] = 16; params["n_layers"]=1
        elif model_name == "SASRec":
            params["max_len"] = 10; params["hidden_dim"]=16; params["n_blocks"]=1; params["n_heads"]=1
        elif model_name == "STAMP":
            params["mlp_params"] = {"dims": [32]}
    elif model_name in ["ComirecDR", "ComirecSA", "SINE"]:
        params = {"features": user_features if model_name == "SINE" else item_features, "max_len": 10, "num_interest": 4}
        if model_name == "SINE":
             params["interest_extractor_params"] = {"dims": [32]}; params["learning_rate"] = 0.001
    else:
        pytest.skip(f"Model {model_name} needs specific setup.")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        model = model_class(**params)
        
        trainer = MatchTrainer(
            model,
            optimizer_params={"lr": 0.01},
            n_epoch=1,
            device='cpu',
            model_path=temp_dir
        )
        
        trainer.fit(matching_data["train_dl"], matching_data["val_dl"])
        
        # Test inference
        user_embedding = trainer.inference_embedding(model, "user", matching_data["val_dl"], temp_dir)
        item_embedding = trainer.inference_embedding(model, "item", matching_data["item_dl"], temp_dir)
        
        assert user_embedding.shape[0] > 0
        assert item_embedding.shape[0] > 0
        assert user_embedding.shape[1] == item_embedding.shape[1]

if __name__ == '__main__':
    pytest.main(['-v', __file__]) 