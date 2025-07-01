"""
Date: create on 10/19/2022
References:
    paper: (RecSys '19) FiBiNET: combining feature importance and bilinear feature interaction for click-through rate prediction
    url: https://dl.acm.org/doi/abs/10.1145/3298689.3347043
Authors: lailai, lailai_zxy@tju.edu.cn
"""
import torch
from torch import nn

from ...basic.features import SparseFeature
from ...basic.layers import MLP, BiLinearInteractionLayer, EmbeddingLayer, SENETLayer


class FiBiNet(torch.nn.Module):
    """
        Args:
        features (list[Feature Class]): training by the whole module.
        reduction_ratio (int) : Hidden layer reduction factor of SENET layer
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        bilinear_type (str): the type bilinear interaction function, in ["field_all", "field_each", "field_interaction"], field_all means that all features share a W, field_each means that a feature field corresponds to a W_i, field_interaction means that a feature field intersection corresponds to a W_ij
    """

    def __init__(self, features, mlp_params, reduction_ratio=3, bilinear_type="field_interaction", **kwargs):
        super(FiBiNet, self).__init__()
        self.features = features
        self.embedding = EmbeddingLayer(features)
        embedding_dim = max([fea.embed_dim for fea in features])
        num_fields = len([fea.embed_dim for fea in features if isinstance(fea, SparseFeature) and fea.shared_with is None])
        self.senet_layer = SENETLayer(num_fields, reduction_ratio)
        self.bilinear_interaction = BiLinearInteractionLayer(embedding_dim, num_fields, bilinear_type)
        self.dims = num_fields * (num_fields - 1) * embedding_dim
        self.mlp = MLP(self.dims, **mlp_params)

    def forward(self, x):
        embed_x = self.embedding(x, self.features)
        embed_senet = self.senet_layer(embed_x)
        embed_bi1 = self.bilinear_interaction(embed_x)
        embed_bi2 = self.bilinear_interaction(embed_senet)
        shallow_part = torch.flatten(torch.cat([embed_bi1, embed_bi2], dim=1), start_dim=1)
        mlp_out = self.mlp(shallow_part)
        return torch.sigmoid(mlp_out.squeeze(1))
