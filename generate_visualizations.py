import os
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from torch_rechub.basic.features import SparseFeature, SequenceFeature, DenseFeature

# Import models
from torch_rechub.models.matching import DSSM, YoutubeDNN
from torch_rechub.models.ranking import DeepFM
from torch_rechub.utils.visualization import visualize_model

# Ensure docs images directory exists
os.makedirs("docs/zh/images/models", exist_ok=True)

# 1. DSSM setup
user_features = [
    SparseFeature("user_id", vocab_size=100, embed_dim=16),
    SparseFeature("gender", vocab_size=3, embed_dim=16),
    SequenceFeature("hist_movie_id", vocab_size=100, embed_dim=16, pooling="mean", shared_with="movie_id")
]
item_features = [
    SparseFeature("movie_id", vocab_size=100, embed_dim=16),
    SparseFeature("cate_id", vocab_size=10, embed_dim=16)
]
dssm_model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [64, 32]}, # final output is 32. Item needs to be 32 too!
    item_params={"dims": [64, 32]}
)

# 2. YoutubeDNN setup
# The item features must have an embed dim that matches the output of the User Tower!
# The User Tower output comes from user_params.dims[-1] = 16.
# By default SparseFeature uses embed_dim=16 so it should match. Let's trace carefully.
neg_item_feature = [SequenceFeature("neg_items", vocab_size=100, embed_dim=16, pooling="concat", shared_with="movie_id")]
ytdnn_model = YoutubeDNN(
    user_features=user_features, 
    item_features=[SparseFeature("movie_id", vocab_size=100, embed_dim=16)], 
    neg_item_feature=neg_item_feature, 
    user_params={"dims": [64, 16]}, # Matches the 16 of the Item Embedding!
    temperature=0.02
)

# 3. DeepFM setup
dense_feas = [DenseFeature(f"I{i}") for i in range(1, 6)]
sparse_feas = [SparseFeature(f"C{i}", vocab_size=100, embed_dim=16) for i in range(1, 6)]
deepfm_model = DeepFM(
    deep_features=dense_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [64, 32], "dropout": 0.2, "activation": "relu"}
)

print("Generating DSSM graph...")
visualize_model(dssm_model, save_path="docs/zh/images/models/dssm_arch.png", dpi=300, show_shapes=True, expand_nested=True)

print("Generating YoutubeDNN graph...")
visualize_model(ytdnn_model, save_path="docs/zh/images/models/youtube_dnn_arch.png", dpi=300, show_shapes=True, expand_nested=True)

print("Generating DeepFM graph...")
visualize_model(deepfm_model, save_path="docs/zh/images/models/deepfm_arch.png", dpi=300, show_shapes=True, expand_nested=True)

print("Graphs generated successfully in docs/zh/images/models/")
