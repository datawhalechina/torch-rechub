"""
Date: create on 09/01/2022
References:
    paper: (WWW'21) Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems
    url: https://arxiv.org/abs/2008.13535
Authors: lailai, lailai_zxy@tju.edu.cn
"""
import torch

from ...basic.layers import LR, MLP, CrossNetMix, CrossNetV2, EmbeddingLayer


class DCNv2(torch.nn.Module):
    """Deep & Cross Network with a mixture of low-rank architecture

    Args:
        features (list[Feature Class]): training by the whole module.
        n_cross_layers (int) : the number of layers of feature intersection layers
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        use_low_rank_mixture (bool): True, whether to use a mixture of low-rank architecture
        low_rank (int): the rank size of low-rank matrices
        num_experts (int): the number of expert networks
    """

    def __init__(self, features, n_cross_layers, mlp_params, model_structure="parallel", use_low_rank_mixture=True, low_rank=32, num_experts=4, **kwargs):
        super(DCNv2, self).__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)
        if use_low_rank_mixture:
            self.crossnet = CrossNetMix(self.dims, n_cross_layers, low_rank=low_rank, num_experts=num_experts)
        else:
            self.crossnet = CrossNetV2(self.dims, n_cross_layers)
        self.model_structure = model_structure
        assert self.model_structure in ["crossnet_only", "stacked", "parallel"], \
            "model_structure={} not supported!".format(self.model_structure)
        if self.model_structure == "stacked":
            self.stacked_dnn = MLP(self.dims, output_layer=False, **mlp_params)
            final_dim = mlp_params["dims"][-1]
        if self.model_structure == "parallel":
            self.parallel_dnn = MLP(self.dims, output_layer=False, **mlp_params)
            final_dim = mlp_params["dims"][-1] + self.dims
        if self.model_structure == "crossnet_only":  # only CrossNet
            final_dim = self.dims
        self.linear = LR(final_dim)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        cross_out = self.crossnet(embed_x)
        if self.model_structure == "crossnet_only":
            final_out = cross_out
        elif self.model_structure == "stacked":
            final_out = self.stacked_dnn(cross_out)
        elif self.model_structure == "parallel":
            dnn_out = self.parallel_dnn(embed_x)
            final_out = torch.cat([cross_out, dnn_out], dim=1)
        y_pred = self.linear(final_out)
        y_pred = torch.sigmoid(y_pred.squeeze(1))
        return y_pred
