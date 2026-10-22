"""
Date: create on 09/13/2022
References:
    paper: (KDD'21) EDCN: Enhancing Explicit and Implicit Feature Interactions via Information Sharing for Parallel Deep CTR Models
    url: https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf
Authors: lailai, lailai_zxy@tju.edu.cn
"""

import torch
from torch import nn

from ...basic.layers import LR, MLP, CrossLayer, EmbeddingLayer


class EDCN(torch.nn.Module):
    """Deep & Cross Network with a mixture of low-rank architecture

    Args:
        features (list[Feature Class]): training by the whole module.
        n_cross_layers (int) : the number of layers of feature intersection layers
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        bridge_type (str): the type interaction function, in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"]
        use_regulation_module (bool): True, whether to use regulation module
        temperature (int): the temperature coefficient to control distribution
    """

    def __init__(self, features, n_cross_layers, mlp_params, bridge_type="hadamard_product", use_regulation_module=True, temperature=1):
        super().__init__()
        self.features = features
        self.n_cross_layers = n_cross_layers
        self.num_fields = len(features)
        self.dims = sum([fea.embed_dim for fea in features])
        self.fea_dims = [fea.embed_dim for fea in features]
        self.embedding = EmbeddingLayer(features)
        self.cross_layers = nn.ModuleList([CrossLayer(self.dims) for _ in range(n_cross_layers)])
        self.bridge_modules = nn.ModuleList([BridgeModule(self.dims, bridge_type) for _ in range(n_cross_layers)])
        self.regulation_modules = nn.ModuleList([RegulationModule(self.num_fields, self.fea_dims, tau=temperature, use_regulation=use_regulation_module) for _ in range(n_cross_layers)])
        mlp_params["dims"] = [self.dims, self.dims]
        self.mlps = nn.ModuleList([MLP(self.dims, output_layer=False, **mlp_params) for _ in range(n_cross_layers)])
        self.linear = LR(self.dims * 3)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        cross_i, deep_i = self.regulation_modules[0](embed_x)
        cross_0 = cross_i
        for i in range(self.n_cross_layers):
            if i > 0:
                cross_i, deep_i = self.regulation_modules[i](bridge_i)
            cross_i = cross_i + self.cross_layers[i](cross_0, cross_i)
            deep_i = self.mlps[i](deep_i)
            bridge_i = self.bridge_modules[i](cross_i, deep_i)
        x_stack = torch.cat([cross_i, deep_i, bridge_i], dim=1)
        y = self.linear(x_stack)
        return torch.sigmoid(y.squeeze(1))


class BridgeModule(torch.nn.Module):

    def __init__(self, input_dim, bridge_type):
        super(BridgeModule, self).__init__()
        assert bridge_type in ["hadamard_product", "pointwise_addition", "concatenation", "attention_pooling"], 'bridge_type= is not supported'.format(bridge_type)
        self.bridge_type = bridge_type
        if bridge_type == "concatenation":
            self.concat_pooling = nn.Sequential(nn.Linear(input_dim * 2, input_dim), nn.ReLU())
        elif bridge_type == "attention_pooling":
            self.attention_x = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim, bias=False), nn.Softmax(dim=-1))
            self.attention_h = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, input_dim, bias=False), nn.Softmax(dim=-1))

    def forward(self, x, h):
        if self.bridge_type == "hadamard_product":
            out = x * h
        elif self.bridge_type == "pointwise_addition":
            out = x + h
        elif self.bridge_type == "concatenation":
            out = self.concat_pooling(torch.cat([x, h], dim=-1))
        elif self.bridge_type == "attention_pooling":
            out = self.attention_x(x) * x + self.attention_h(h) * h
        return out


class RegulationModule(torch.nn.Module):

    def __init__(self, num_fields, dims, tau, use_regulation=True):
        super(RegulationModule, self).__init__()
        self.use_regulation = use_regulation
        if self.use_regulation:
            self.num_fields = num_fields
            self.dims = dims
            self.tau = tau
            self.g1 = nn.Parameter(torch.ones(num_fields))
            self.g2 = nn.Parameter(torch.ones(num_fields))

    def forward(self, x):
        if self.use_regulation:
            g1 = torch.cat([(self.g1[i] / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.dims[i]) for i in range(self.num_fields)], dim=-1)
            g2 = torch.cat([(self.g2[i] / self.tau).softmax(dim=-1).unsqueeze(-1).repeat(1, self.dims[i]) for i in range(self.num_fields)], dim=-1)

            out1, out2 = g1 * x, g2 * x
        else:
            out1, out2 = x, x
        return out1, out2
