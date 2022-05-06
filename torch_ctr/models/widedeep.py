"""
Created on April 22, 2022
Reference: "Wide & Deep Learning for Recommender Systems", DLRS, 2016
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ..basic.layers import LR, MLP, EmbeddingLayer


class WideDeep(torch.nn.Module):

    def __init__(self, wide_features, deep_features, mlp_params):
        super(WideDeep, self).__init__()
        self.wide_features = wide_features
        self.deep_features = deep_features
        self.wide_dims = sum([fea.embed_dim for fea in wide_features])
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.linear = LR(self.wide_dims)
        self.embedding = EmbeddingLayer(wide_features + deep_features)
        self.mlp = MLP(self.deep_dims, **mlp_params)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_dense_fields)``
        """
        input_wide = self.embedding(x, self.wide_features, squeeze_dim=True)  #[batch_size, wide_dims]
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  #[batch_size, deep_dims]

        y_wide = self.linear(input_wide)  #[batch_size, 1]
        y_deep = self.mlp(input_deep)  #[batch_size, 1]
        y = y_wide + y_deep
        y = torch.sigmoid(y.squeeze(1))
        return y
