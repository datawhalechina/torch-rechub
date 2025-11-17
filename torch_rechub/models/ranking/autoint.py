"""
Date: create on 14/11/2025
References:
    paper: (CIKM'2019) AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    url: https://arxiv.org/abs/1810.11921
Authors: Yang Zhou, zyaztec@gmail.com
"""

import torch
import torch.nn as nn

from ...basic.layers import LR, MLP, EmbeddingLayer, InteractingLayer


class AutoInt(torch.nn.Module):
    """AutoInt Model

    Args:
        sparse_features (list): the list of `SparseFeature` Class
        dense_features (list): the list of `DenseFeature` Class
        num_layers (int): number of interacting layers
        num_heads (int): number of attention heads
        dropout (float): dropout rate for attention
        mlp_params (dict): parameters for MLP, keys: {"dims":list, "activation":str,
                                             "dropout":float, "output_layer":bool"}
    """

    def __init__(self, sparse_features, dense_features, num_layers=3, num_heads=2, dropout=0.0, mlp_params=None):
        super(AutoInt, self).__init__()
        self.sparse_features = sparse_features

        self.dense_features = dense_features if dense_features is not None else []
        embed_dims = [fea.embed_dim for fea in self.sparse_features]
        self.embed_dim = embed_dims[0]
        if len(self.sparse_features) == 0:
            raise ValueError("AutoInt requires at least one sparse feature to determine embed_dim.")

        # field nums = sparse + dense
        self.num_sparse = len(self.sparse_features)
        self.num_dense = len(self.dense_features)
        self.num_fields = self.num_sparse + self.num_dense

        # total dims = num_fields * embed_dim
        self.dims = self.num_fields * self.embed_dim
        self.num_layers = num_layers

        self.sparse_embedding = EmbeddingLayer(self.sparse_features)

        # dense feature embedding
        self.dense_embeddings = nn.ModuleDict()
        for fea in self.dense_features:
            self.dense_embeddings[fea.name] = nn.Linear(1, self.embed_dim, bias=False)

        self.interacting_layers = torch.nn.ModuleList([InteractingLayer(self.embed_dim, num_heads=num_heads, dropout=dropout, residual=True) for _ in range(num_layers)])

        self.linear = LR(self.dims)

        self.attn_linear = nn.Linear(self.dims, 1)

        if mlp_params is not None:
            self.use_mlp = True
            self.mlp = MLP(self.dims, **mlp_params)
        else:
            self.use_mlp = False

    def forward(self, x):
        # sparse feature embedding: [B, num_sparse, embed_dim]
        sparse_emb = self.sparse_embedding(x, self.sparse_features, squeeze_dim=False)

        dense_emb_list = []
        for fea in self.dense_features:
            v = x[fea.name].float().view(-1, 1, 1)
            dense_emb = self.dense_embeddings[fea.name](v)  # [B, 1, embed_dim]
            dense_emb_list.append(dense_emb)

        if len(dense_emb_list) > 0:
            dense_emb = torch.cat(dense_emb_list, dim=1)  # [B, num_dense, d]
            embed_x = torch.cat([sparse_emb, dense_emb], dim=1)  # [B, num_fields, d]
        else:
            embed_x = sparse_emb  # [B, num_sparse, d]

        embed_x_flatten = embed_x.flatten(start_dim=1)  # [B, num_fields * embed_dim]

        # Multi-head self-attention layers
        attn_out = embed_x
        for layer in self.interacting_layers:
            attn_out = layer(attn_out)  # [B, num_fields, embed_dim]

        # Attention linear
        attn_out_flatten = attn_out.flatten(start_dim=1)  # [B, num_fields * embed_dim]
        y_attn = self.attn_linear(attn_out_flatten)  # [B, 1]

        # Linear part
        y_linear = self.linear(embed_x_flatten)  # [B, 1]

        # Deep MLP
        y = y_attn + y_linear
        if self.use_mlp:
            y_deep = self.mlp(embed_x_flatten)  # [B, 1]
            y = y + y_deep

        return torch.sigmoid(y.squeeze(1))
