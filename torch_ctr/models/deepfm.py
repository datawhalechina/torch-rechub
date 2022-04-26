"""
Created on April 22, 2022
Reference: "DeepFM: A Factorization-Machine based Neural Network for CTR Prediction", IJCAI, 2017
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ..layers import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron


class DeepFM(torch.nn.Module):
    """
        Deep Factorization Machine Model
    """

    def __init__(self, dense_field_nums, sparse_field_dims, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.linear = FeaturesLinear(dense_field_nums)
        self.fm = FactorizationMachine(reduce_sum=True)
        self.embedding = FeaturesEmbedding(sparse_field_dims, embed_dim)
        self.mlp = MultiLayerPerceptron(len(sparse_field_dims) * embed_dim, mlp_dims, dropout)

    def forward(self, x):
        """
        :param x_dense: Long tensor of size ``(batch_size, num_dense_fields)``
        :param x_sparse: Long tensor of size ``(batch_size, num_sparse_fields)``
        """
        x_dense, x_sparse = x["x_dense"], x["x_sparse"]
        y_linear = self.linear(x_dense) #dense特征不参与交叉，建议提前离散化
        embed_x = self.embedding(x_sparse)
        y_fm = self.fm(embed_x)
        y_deep = self.mlp(embed_x.flatten(start_dim=1))
        y = y_linear + y_fm + y_deep
        return torch.sigmoid(y.squeeze(1))