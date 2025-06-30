"""
Date: create on 12/05/2022
References:
    paper: (AKDD'2017) Deep & Cross Network for Ad Click Predictions
    url: https://arxiv.org/abs/1708.05123
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ...basic.layers import LR, MLP, CrossNetwork, EmbeddingLayer


class DCN(torch.nn.Module):
    """Deep & Cross Network

    Args:
        features (list[Feature Class]): training by the whole module.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, features, n_cross_layers, mlp_params):
        super().__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])

        self.embedding = EmbeddingLayer(features)
        self.cn = CrossNetwork(self.dims, n_cross_layers)
        self.mlp = MLP(self.dims, output_layer=False, **mlp_params)
        self.linear = LR(self.dims + mlp_params["dims"][-1])

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        cn_out = self.cn(embed_x)
        mlp_out = self.mlp(embed_x)
        x_stack = torch.cat([cn_out, mlp_out], dim=1)
        y = self.linear(x_stack)
        return torch.sigmoid(y.squeeze(1))
