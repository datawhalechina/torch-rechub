"""
Date: create on 23/04/2022, update on 30/04/2022
References:
    paper: (KDD'2018) Deep Interest Network for Click-Through Rate Prediction
    url: https://arxiv.org/abs/1706.06978
    code: https://github.com/huawei-noah/benchmark/blob/main/FuxiCTR/fuxictr/pytorch/models/DIN.py
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer


class DIN(nn.Module):
    """Deep Interest Network
    Args:
        features (list): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
        history_features (list): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        attention_mlp_params (dict): the params of the ActivationUnit module, keys include:`{"dims":list, "activation":str, "dropout":float, "use_softmax":bool`}
    """

    def __init__(self, features, history_features, target_features, mlp_params, attention_mlp_params):
        super().__init__()
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.num_history_features = len(history_features)
        self.all_dims = sum([fea.embed_dim for fea in features + history_features + target_features])

        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.attention_layers = nn.ModuleList([ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features])
        self.mlp = MLP(self.all_dims, activation="dice", **mlp_params)

    def forward(self, x):
        # (batch_size, num_features, emb_dim)
        embed_x_features = self.embedding(x, self.features)
        # (batch_size, num_history_features, seq_length, emb_dim)
        embed_x_history = self.embedding(x, self.history_features)
        # (batch_size, num_target_features, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)
        attention_pooling = []
        for i in range(self.num_history_features):
            attention_seq = self.attention_layers[i](embed_x_history[:, i, :, :], embed_x_target[:, i, :])
            attention_pooling.append(attention_seq.unsqueeze(1))  # (batch_size, 1, emb_dim)
        # (batch_size, num_history_features, emb_dim)
        attention_pooling = torch.cat(attention_pooling, dim=1)

        mlp_in = torch.cat([attention_pooling.flatten(start_dim=1), embed_x_target.flatten(start_dim=1), embed_x_features.flatten(start_dim=1)], dim=1)  # (batch_size, N)

        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))


class ActivationUnit(nn.Module):
    """Activation Unit Layer mentioned in DIN paper, it is a Target Attention method.

    Args:
        embed_dim (int): the length of embedding vector.
        history (tensor):
    Shape:
        - Input: `(batch_size, seq_length, emb_dim)`
        - Output: `(batch_size, emb_dim)`
    """

    def __init__(self, emb_dim, dims=None, activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        if dims is None:
            dims = [36]
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.attention = MLP(4 * self.emb_dim, dims=dims, activation=activation)

    def forward(self, history, target):
        seq_length = history.size(1)
        # batch_size,seq_length,emb_dim
        target = target.unsqueeze(1).expand(-1, seq_length, -1)
        att_input = torch.cat([target, history, target - history, target * history], dim=-1)  # batch_size,seq_length,4*emb_dim
        # (batch_size*seq_length,4*emb_dim)
        att_weight = self.attention(att_input.view(-1, 4 * self.emb_dim))
        # (batch_size*seq_length, 1) -> (batch_size,seq_length)
        att_weight = att_weight.view(-1, seq_length)
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1)


# (batch_size, seq_length, 1) * (batch_size, seq_length, emb_dim)
# (batch_size,emb_dim)
        output = (att_weight.unsqueeze(-1) * history).sum(dim=1)
        return output
