"""
Date: create on 14/05/2022
References:
    paper: (KDD'2021) Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning in Targeted Display Advertising
    url: https://arxiv.org/abs/2105.08489
    code: https://github.com/adtalos/AITM-torch
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer


class AITM(nn.Module):
    """ Adaptive Information Transfer Multi-task (AITM) framework.
        all the task type must be binary classificatioon.

    Args:
        features (list[Feature Class]): training by the whole module.
        n_task (int): the number of binary classificatioon task.
        bottom_params (dict): the params of all the botwer expert module, keys include:`{"dims":list, "activation":str, "dropout":float}`.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, n_task, bottom_params, tower_params_list):
        super().__init__()
        self.features = features
        self.n_task = n_task
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)

        self.bottoms = nn.ModuleList(MLP(self.input_dims, output_layer=False, **bottom_params) for i in range(self.n_task))
        self.towers = nn.ModuleList(MLP(bottom_params["dims"][-1], **tower_params_list[i]) for i in range(self.n_task))

        self.info_gates = nn.ModuleList(MLP(bottom_params["dims"][-1], output_layer=False, dims=[bottom_params["dims"][-1]]) for i in range(self.n_task - 1))
        self.aits = nn.ModuleList(AttentionLayer(bottom_params["dims"][-1]) for _ in range(self.n_task - 1))

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  # [batch_size, *]
        input_towers = [self.bottoms[i](embed_x) for i in range(self.n_task)]  # [i]:[batch_size, bottom_dims[-1]]
        for i in range(1, self.n_task):  # for task 1:n-1
            # [batch_size,1,bottom_dims[-1]]
            info = self.info_gates[i - 1](input_towers[i - 1]).unsqueeze(1)
            # [batch_size, 2, bottom_dims[-1]]
            ait_input = torch.cat([input_towers[i].unsqueeze(1), info], dim=1)
            input_towers[i] = self.aits[i - 1](ait_input)

        ys = []
        for input_tower, tower in zip(input_towers, self.towers):
            y = tower(input_tower)
            ys.append(torch.sigmoid(y))
        return torch.cat(ys, dim=1)


class AttentionLayer(nn.Module):
    """attention for info tranfer

    Args:
        dim (int): attention dim

    Shape:
        Input: (batch_size, 2, dim)
        Output: (batch_size, dim)
    """

    def __init__(self, dim=32):
        super().__init__()
        self.dim = dim
        self.q_layer = nn.Linear(dim, dim, bias=False)
        self.k_layer = nn.Linear(dim, dim, bias=False)
        self.v_layer = nn.Linear(dim, dim, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)
        a = torch.sum(torch.mul(Q, K), -1) / torch.sqrt(torch.tensor(self.dim))
        a = self.softmax(a)
        outputs = torch.sum(torch.mul(torch.unsqueeze(a, -1), V), dim=1)
        return outputs
