"""
Date: create on 26/02/2024, update on 30/04/2022
References:
    paper: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
    url: https://arxiv.org/pdf/1905.06874
    code: https://github.com/jiwidi/Behavior-Sequence-Transformer-Pytorch/blob/master/pytorch_bst.ipynb
Authors: Tao Fan, thisisevy@foxmail.com
"""

import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer


class BST(nn.Module):
    """Behavior Sequence Transformer

    Args:
        features (list): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
        history_features (list): the list of `Feature Class`,training by Transformer. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by Transformer. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        nhead (int): the number of heads in the multi-head-attention models.
        dropout (float): the dropout value in the multi-head-attention models.
        num_layers (Any): the number of sub-encoder-layers in the encoder.
        max_seq_len (int): maximum sequence length (history + 1 target). Used for positional encoding table size.
    """

    def __init__(self, features, history_features, target_features, mlp_params, nhead=8, dropout=0.2, num_layers=1, max_seq_len=51):
        super(BST, self).__init__()
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.max_seq_len = max_seq_len
        self.item_dim = sum([fea.embed_dim for fea in history_features])
        target_dim = sum([fea.embed_dim for fea in target_features])
        if self.item_dim != target_dim:
            raise ValueError(f"sum of history_features embed_dim ({self.item_dim}) must equal sum of target_features embed_dim ({target_dim})")
        if self.item_dim % nhead != 0:
            raise ValueError(f"item_dim ({self.item_dim}) must be divisible by nhead ({nhead})")
        self.all_dims = sum([fea.embed_dim for fea in features + target_features]) + self.item_dim
        self.embedding = EmbeddingLayer(features + history_features + target_features)
        # positional encoding: absolute index (simplified from paper's time-diff pos)
        self.pos_embedding = nn.Embedding(max_seq_len, self.item_dim)
        # paper uses LeakyReLU in FFN
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.item_dim, nhead=nhead, dropout=dropout, activation=nn.LeakyReLU(), batch_first=True)
        self.transformer_layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = MLP(self.all_dims, **mlp_params)

    def forward(self, x):
        embed_x_features = self.embedding(x, self.features)
        # (batch_size, num_history_features, seq_length, emb_dim)
        embed_x_history = self.embedding(x, self.history_features)
        # (batch_size, num_target_features, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)

        # fuse all history features into one item vector per timestep: [B, T, item_dim]
        hist = torch.cat([embed_x_history[:, i] for i in range(len(self.history_features))], dim=-1)
        # fuse target features into one vector: [B, item_dim]
        tgt = torch.cat([embed_x_target[:, i] for i in range(len(self.target_features))], dim=-1)

        # append target at end of sequence: [B, T+1, item_dim]
        seq = torch.cat([hist, tgt.unsqueeze(1)], dim=1)
        if seq.size(1) > self.max_seq_len:
            raise ValueError(f"sequence length {seq.size(1)} exceeds max_seq_len {self.max_seq_len}")
        positions = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        seq = seq + self.pos_embedding(positions)

        # padding mask: a position is padding only if ALL history features are padding there
        pad_mask = torch.ones(embed_x_history.size(0), embed_x_history.size(2), dtype=torch.bool, device=embed_x_history.device)
        for fea in self.history_features:
            pidx = fea.padding_idx if fea.padding_idx is not None else 0
            pad_mask = pad_mask & (x[fea.name].long() == pidx)
        tgt_mask = torch.zeros(pad_mask.size(0), 1, dtype=torch.bool, device=pad_mask.device)
        src_key_padding_mask = torch.cat([pad_mask, tgt_mask], dim=1)  # [B, T+1]

        out = self.transformer_layers(seq, src_key_padding_mask=src_key_padding_mask)
        # take target position output as interest representation
        interest = out[:, -1, :]  # [B, item_dim]

        mlp_in = torch.cat([
            interest,
            embed_x_target.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1),
        ],
                           dim=1)
        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))
