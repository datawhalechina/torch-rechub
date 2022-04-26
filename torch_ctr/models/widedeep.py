"""
Created on April 22, 2022
Reference: "Wide & Deep Learning for Recommender Systems", DLRS, 2016
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ..layers import FeaturesLinear, MultiLayerPerceptron, FeaturesEmbedding

class WideAndDeep(torch.nn.Module):

    def __init__(self, dense_field_nums, sparse_field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(dense_field_nums)
        self.embedding = FeaturesEmbedding(sparse_field_dims, embed_dim)
        self.mlp = MultiLayerPerceptron(len(sparse_field_dims) * embed_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x_dense: Long tensor of size ``(batch_size, num_dense_fields)``
        :param x_sparse: Long tensor of size ``(batch_size, num_sparse_fields)``
        """
        x_dense, x_sparse = x["x_dense"], x["x_sparse"]
        y_wide = self.linear(x_dense)
        embed_x = self.embedding(x_sparse)
        y_deep = self.mlp(embed_x.flatten(start_dim=1))
        y = y_wide + y_deep
        y = torch.sigmoid(y.squeeze(1))
        return y
