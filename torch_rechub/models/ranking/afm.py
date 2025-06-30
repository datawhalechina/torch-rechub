"""
Date: create on 23/04/2024
References:
    paper: (IJCAI'2017) Attentional Factorization Machines：Learning the Weight of Feature Interactions via Attention Networks
    url: https://arxiv.org/abs/1708.04617
Authors: Tao Fan, thisisevy@foxmail.com
"""

import torch
from torch import nn
from torch.nn import Parameter, init

from ...basic.layers import FM, LR, MLP, EmbeddingLayer


class AFM(nn.Module):
    """Attentional Factorization Machine Model

    Args:
        fm_features (list): the list of `Feature Class`, training by the fm part module.
        embed_dim (int): the dimension of input embedding.
        t (int): the size of the hidden layer in the attention network.
    """

    def __init__(self, fm_features, embed_dim, t=64):
        super(AFM, self).__init__()
        self.fm_features = fm_features
        self.embed_dim = embed_dim
        self.fm_dims = sum([fea.embed_dim for fea in fm_features])
        self.linear = LR(self.fm_dims)  # 1-odrder interaction
        self.fm = FM(reduce_sum=False)  # 2-odrder interaction
        self.embedding = EmbeddingLayer(fm_features)

        # 注意力计算中的线性层
        self.attention_liner = nn.Linear(self.embed_dim, t)
        # AFM公式中的h
        self.h = init.xavier_uniform_(Parameter(torch.empty(t, 1)))
        # AFM公式中的p
        self.p = init.xavier_uniform_(Parameter(torch.empty(self.embed_dim, 1)))

    def attention(self, y_fm):
        # embs: [ batch_size, k ]
        # [ batch_size, t ]
        y_fm = self.attention_liner(y_fm)
        # [ batch_size, t ]
        y_fm = torch.relu(y_fm)
        # [ batch_size, 1 ]
        y_fm = torch.matmul(y_fm, self.h)
        # [ batch_size, 1 ]
        atts = torch.softmax(y_fm, dim=1)
        return atts

    def forward(self, x):
        # [batch_size, num_fields, embed_dim]
        input_fm = self.embedding(x, self.fm_features, squeeze_dim=False)

        y_linear = self.linear(input_fm.flatten(start_dim=1))
        y_fm = self.fm(input_fm)
        # 得到注意力
        atts = self.attention(y_fm)
        # [ batch_size, 1 ]
        outs = torch.matmul(atts * y_fm, self.p)
        # print(y_linear.size(), outs.size())
        y = y_linear + outs
        return torch.sigmoid(y.squeeze(1))
