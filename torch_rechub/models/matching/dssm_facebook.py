"""
Date: create on 24/05/2022
References:
    paper: (KDD'2020) Embedding-based Retrieval in Facebook Search
    url: https://arxiv.org/abs/2006.11632
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn.functional as F

from ...basic.layers import MLP, EmbeddingLayer


class FaceBookDSSM(torch.nn.Module):
    """Embedding-based Retrieval in Facebook Search
    It's a DSSM match model trained by hinge loss on pair-wise samples.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        pos_item_features (list[Feature Class]): negative sample features, training by the item tower module.
        neg_item_features (list[Feature Class]): positive sample features, training by the item tower module.
        temperature (float): temperature factor for similarity score, default to 1.0.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
    """

    def __init__(self, user_features, pos_item_features, neg_item_features, user_params, item_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.pos_item_features = pos_item_features
        self.neg_item_features = neg_item_features
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in pos_item_features])

        self.embedding = EmbeddingLayer(user_features + pos_item_features + neg_item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        pos_item_embedding, neg_item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return pos_item_embedding


# calculate cosine score
        pos_score = torch.mul(user_embedding, pos_item_embedding).sum(dim=1)
        neg_score = torch.mul(user_embedding, neg_item_embedding).sum(dim=1)

        return pos_score, neg_score

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [batch_size, num_features*deep_dims]
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)
        # [batch_size, user_params["dims"][-1]]
        user_embedding = self.user_mlp(input_user)
        user_embedding = F.normalize(user_embedding, p=2, dim=1)
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None, None
        input_item_pos = self.embedding(x, self.pos_item_features, squeeze_dim=True)
        if self.mode == "item":  # inference embedding mode, the zeros is just for placefolder
            return self.item_mlp(input_item_pos), None
        input_item_neg = self.embedding(x, self.neg_item_features, squeeze_dim=True)
        pos_embedding, neg_embedding = self.item_mlp(input_item_pos), self.item_mlp(input_item_neg)
        pos_embedding = F.normalize(pos_embedding, p=2, dim=1)
        neg_embedding = F.normalize(neg_embedding, p=2, dim=1)
        return pos_embedding, neg_embedding
