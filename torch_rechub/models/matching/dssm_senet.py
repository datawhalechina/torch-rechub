"""
Date: create on 12/19/2024
References:
    url: https://zhuanlan.zhihu.com/p/358779957
Authors: @1985312383
"""

import torch
import torch.nn.functional as F

from ...basic.features import SequenceFeature, SparseFeature
from ...basic.layers import MLP, EmbeddingLayer, SENETLayer


class DSSM(torch.nn.Module):
    """Deep Structured Semantic Model

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the item tower module.
        temperature (float): temperature factor for similarity score, default to 1.0.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
    """

    def __init__(self, user_features, item_features, user_params, item_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])

        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.user_num_features = len([fea.embed_dim for fea in self.user_features if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature) and fea.shared_with is None])
        self.item_num_features = len([fea.embed_dim for fea in self.item_features if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature) and fea.shared_with is None])
        self.user_senet = SENETLayer(self.user_num_features)
        self.item_senet = SENETLayer(self.item_num_features)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding


# calculate cosine score
        y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        y = y / self.temperature
        return torch.sigmoid(y)

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [batch_size, num_features * embed_dim]
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)
        # [batch_size, num_features, embed_dim]
        input_user = input_user.view(input_user.size(0), self.user_num_features, -1)
        # [batch_size, num_features, embed_dim]
        input_user = self.user_senet(input_user)
        # [batch_size, num_features * embed_dim]
        input_user = input_user.view(input_user.size(0), -1)
        # [batch_size, user_params["dims"][-1]]
        user_embedding = self.user_mlp(input_user)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)  # L2 normalize
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        # [batch_size, num_features * embed_dim]
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)
        # [batch_size, num_features, embed_dim]
        input_item = input_item.view(input_item.size(0), self.item_num_features, -1)
        # [batch_size, num_features, embed_dim]
        input_item = self.item_senet(input_item)
        # [batch_size, num_features * embed_dim]
        input_item = input_item.view(input_item.size(0), -1)
        # [batch_size, item_params["dims"][-1]]
        item_embedding = self.item_mlp(input_item)
        item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding
