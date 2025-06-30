"""
Date: created on 31/07/2022
References:
    paper: FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine
    url: https://arxiv.org/abs/1905.06336
Authors: Bo Kang, klinux@live.com
"""

import torch
import torch.nn as nn

from ...basic.layers import CEN, FFM, MLP, EmbeddingLayer


class DeepFFM(nn.Module):
    """The DeepFFM model, mentioned on the `webpage
    <https://cs.nju.edu.cn/31/60/c1654a209248/page.htm>` which is the first
    work that introduces FFM model into neural CTR system. It is also described
    in the `FAT-DeepFFM paper <https://arxiv.org/abs/1905.06336>`.

    Args:
        linear_features (list): the list of `Feature Class`, fed to the linear module.
        cross_features (list): the list of `Feature Class`, fed to the ffm module.
        embed_dim (int): the dimensionality of categorical value embedding.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, linear_features, cross_features, embed_dim, mlp_params):
        super().__init__()
        self.linear_features = linear_features
        self.cross_features = cross_features

        self.num_fields = len(cross_features)
        self.num_field_cross = self.num_fields * (self.num_fields - 1) // 2

        self.ffm = FFM(num_fields=self.num_fields, reduce_sum=False)
        self.mlp_out = MLP(self.num_field_cross * embed_dim, **mlp_params)

        self.linear_embedding = EmbeddingLayer(linear_features)
        self.ffm_embedding = EmbeddingLayer(cross_features)

        self.b = torch.nn.Parameter(torch.zeros(1))

        # This keeping constant value in module on correct device
        # url:
        # https://discuss.pytorch.org/t/keeping-constant-value-in-module-on-correct-device/10129
        fields_offset = torch.arange(0, self.num_fields, dtype=torch.long)
        self.register_buffer('fields_offset', fields_offset)

    def forward(self, x):
        # compute scores from the linear part of the model, where input is the
        # raw features (Eq. 5, FAT-DeepFFM)
        y_linear = self.linear_embedding(x, self.linear_features, squeeze_dim=True).sum(1, keepdim=True)  # [batch_size, 1]

        # gather the embeddings. Each feature value corresponds to multiple embeddings, with multiplicity equal to number of features/fields.
        # output shape [batch_size, num_field, num_field, emb_dim]
        x_ffm = {fea.name: x[fea.name].unsqueeze(1) * self.num_fields + self.fields_offset for fea in self.cross_features}
        input_ffm = self.ffm_embedding(x_ffm, self.cross_features, squeeze_dim=False)

        # compute second order field-aware feature crossings, output shape
        # [batch_size, num_field_cross, emb_dim]
        em = self.ffm(input_ffm)

        # compute scores from the ffm part of the model, output shape
        # [batch_size, 1]
        y_ffm = self.mlp_out(em.flatten(start_dim=1))

        # compute final prediction
        y = y_linear + y_ffm
        return torch.sigmoid(y.squeeze(1) + self.b)


class FatDeepFFM(nn.Module):
    """The FAT-DeepFFM model, mentioned in the `FAT-DeepFFM paper
    <https://arxiv.org/abs/1905.06336>`. It combines DeepFFM with
    Compose-Excitation Network (CENet) field attention mechanism
    to highlight the importance of second-order feature crosses.

    Args:
        linear_features (list): the list of `Feature Class`, fed to the linear module.
        cross_features (list): the list of `Feature Class`, fed to the ffm module.
        embed_dim (int): the dimensionality of categorical value embedding.
        reduction_ratio (int): the between the dimensions of input layer and hidden layer of the CEN MLP module.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, linear_features, cross_features, embed_dim, reduction_ratio, mlp_params):
        super().__init__()
        self.linear_features = linear_features
        self.cross_features = cross_features

        self.num_fields = len(cross_features)
        self.num_field_cross = self.num_fields * (self.num_fields - 1) // 2

        self.ffm = FFM(num_fields=self.num_fields, reduce_sum=False)
        self.cen = CEN(embed_dim, self.num_field_cross, reduction_ratio)
        self.mlp_out = MLP(self.num_field_cross * embed_dim, **mlp_params)

        self.linear_embedding = EmbeddingLayer(linear_features)
        self.ffm_embedding = EmbeddingLayer(cross_features)

        self.b = torch.nn.Parameter(torch.zeros(1))

        fields_offset = torch.arange(0, self.num_fields, dtype=torch.long)
        self.register_buffer('fields_offset', fields_offset)

    def forward(self, x):
        # compute scores from the linear part of the model, where input is the
        # raw features (Eq. 5, FAT-DeepFFM)
        y_linear = self.linear_embedding(x, self.linear_features, squeeze_dim=True).sum(1, keepdim=True)  # [batch_size, 1]

        # gather the embeddings. Each feature value corresponds to multiple embeddings, with multiplicity is equal to the number of features/fields.
        # output shape [batch_size, num_field, num_field, emb_dim]
        x_ffm = {fea.name: x[fea.name].unsqueeze(1) * self.num_fields + self.fields_offset for fea in self.cross_features}
        input_ffm = self.ffm_embedding(x_ffm, self.cross_features, squeeze_dim=False)

        # compute second order field-aware feature crossings, output shape
        # [batch_size, num_field_cross, emb_dim]
        em = self.ffm(input_ffm)

        # rescale FFM embeddings with field attention (Eq.10), output shape
        # [batch_size, num_field_cross * emb_dim]
        aem = self.cen(em)

        # compute scores from the ffm part of the model, output shape
        # [batch_size, 1]
        y_ffm = self.mlp_out(aem)

        # compute final prediction
        y = y_linear + y_ffm
        return torch.sigmoid(y.squeeze(1) + self.b)
