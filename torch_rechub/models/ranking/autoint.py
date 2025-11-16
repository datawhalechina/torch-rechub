"""
Date: create on 14/11/2025
References:
    paper: (CIKM'2019) AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks
    url: https://arxiv.org/abs/1810.11921
Authors: Yang Zhou, zyaztec@gmail.com
"""

import torch

from ...basic.layers import LR, MLP, EmbeddingLayer, InteractingLayer


class AutoInt(torch.nn.Module):
    """AutoInt Model

    Args:
        features (list): the list of `Feature Class`, training by the entire model.
        num_layers (int): the number of interacting layers.
        num_heads (int): the number of attention heads (default=2).
        dropout (float): the dropout rate for attention layers (default=0.0).
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """

    def __init__(self, features, num_layers=3, num_heads=2, dropout=0.0, mlp_params=None):
        super(AutoInt, self).__init__()
        self.features = features
        self.dims = sum([fea.embed_dim for fea in features])
        self.num_layers = num_layers
        
        self.embedding = EmbeddingLayer(features)
        
        # Check if all features have the same embedding dimension
        embed_dims = [fea.embed_dim for fea in features]
        if len(set(embed_dims)) != 1:
            raise ValueError("All features must have the same embedding dimension for AutoInt")
        self.embed_dim = embed_dims[0]
        
        # Interacting layers (multi-head self-attention)
        self.interacting_layers = torch.nn.ModuleList([
            InteractingLayer(self.embed_dim, num_heads=num_heads, dropout=dropout, residual=True)
            for _ in range(num_layers)
        ])
        
        # Linear part for 1st-order feature interactions
        self.linear = LR(self.dims)
        
        # Optional MLP for deep learning
        if mlp_params is not None:
            self.use_mlp = True
            self.mlp = MLP(self.dims, **mlp_params)
        else:
            self.use_mlp = False

    def forward(self, x):
        # Embedding layer: [batch_size, num_fields, embed_dim]
        embed_x = self.embedding(x, self.features, squeeze_dim=False)
        
        # Multi-head self-attention layers
        attn_out = embed_x
        for layer in self.interacting_layers:
            attn_out = layer(attn_out)
        
        # Flatten attention output: [batch_size, num_fields * embed_dim]
        attn_out = attn_out.flatten(start_dim=1)
        
        # Linear part (1st-order)
        y_linear = self.linear(embed_x.flatten(start_dim=1))
        
        # Attention part (high-order)
        y_attn = torch.sum(attn_out, dim=1, keepdim=True)
        
        # Combine results
        y = y_linear + y_attn
        
        # Optional MLP
        if self.use_mlp:
            y_deep = self.mlp(embed_x.flatten(start_dim=1))
            y = y + y_deep
        
        return torch.sigmoid(y.squeeze(1))
