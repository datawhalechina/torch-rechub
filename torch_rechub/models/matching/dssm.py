"""
Date: create on 12/05/2022, update on 20/05/2022
References: 
    paper: (CIKM'2013) Learning Deep Structured Semantic Models for Web Search using Clickthrough Data
    url: https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf
    code: https://github.com/bbruceyuan/DeepMatch-Torch/blob/main/deepmatch_torch/models/dssm.py
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ...basic.layers import MLP, EmbeddingLayer


class DSSM(torch.nn.Module):
    """Deep Structured Semantic Model

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the item tower module.
        sim_func (str): similarity function, includes `["cosine", "dot"]`, default to "cosine".
        temperature (float): temperature factor for similarity score, default to 1.0.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
    """

    def __init__(self, user_features, item_features, user_params, item_params, sim_func="cosine", temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.sim_func = sim_func
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])

        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding
        if self.sim_func == "cosine":
            y = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        elif self.sim_func == "dot":
            y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        else:
            raise ValueError("similarity function only support %s, but got %s" % (["cosine", "dot"], self.sim_func))
        y = y / self.temperature
        return torch.sigmoid(y)

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user)  #[batch_size, user_params["dims"][-1]]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)  #[batch_size, num_features*embed_dim]
        item_embedding = self.item_mlp(input_item)  #[batch_size, item_params["dims"][-1]]
        return item_embedding