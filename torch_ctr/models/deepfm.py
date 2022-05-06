"""
Created on April 22, 2022
Reference: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", IJCAI, 2017
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ..basic.layers import FactorizationMachine, EmbeddingLayer, MLP, LR


class DeepFM(torch.nn.Module):
    """
        Deep Factorization Machine Model
    """

    def __init__(self, deep_features, fm_features, mlp_params):
        super(DeepFM, self).__init__()
        self.deep_features = deep_features
        self.fm_features = fm_features
        self.deep_dims = sum([fea.embed_dim for fea in deep_features])
        self.fm_dims = sum([fea.embed_dim for fea in fm_features])
        self.linear = LR(self.fm_dims)  # 1-odrder interaction
        self.fm = FactorizationMachine(reduce_sum=True)  # 2-odrder interaction
        self.embedding = EmbeddingLayer(deep_features + fm_features)
        self.mlp = MLP(self.deep_dims, **mlp_params)

    def forward(self, x):
        """
        """
        input_deep = self.embedding(x, self.deep_features, squeeze_dim=True)  #[batch_size, deep_dims]
        input_fm = self.embedding(x, self.fm_features, squeeze_dim=False)  #[batch_size, num_fields, embed_dim]

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        y_deep = self.mlp(input_deep)  #[batch_size, 1]
        y = y_linear + y_fm + y_deep
        return torch.sigmoid(y.squeeze(1))