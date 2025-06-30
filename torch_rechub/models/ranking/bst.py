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
        history_features (list): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        nhead (int): the number of heads in the multi-head-attention models.
        dropout (float): the dropout value in the multi-head-attention models.
        num_layers (Any): the number of sub-encoder-layers in the encoder.
    """

    def __init__(self, features, history_features, target_features, mlp_params, nhead=8, dropout=0.2, num_layers=1):
        super().__init__()
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.num_history_features = len(history_features)
        self.embed_dim = target_features[0].embed_dim
        self.seq_len = 50
        # TODO 在 'torch_rechub.basic.features.SequenceFeature' 中加入seq_len属性
        self.all_dims = (len(features) + len(history_features) * (self.seq_len + len(target_features))) * self.embed_dim
        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, dropout=dropout)
        self.transformer_layers = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # # 定义模型，模型的参数需要我们之前的feature类，用于构建模型的输入层，mlp指定模型后续DNN的结构
        self.mlp = MLP(self.all_dims, activation="leakyrelu", **mlp_params)

    def forward(self, x):
        # (batch_size, num_features, emb_dim)
        embed_x_features = self.embedding(x, self.features)
        # (batch_size, num_history_features, seq_length, emb_dim)
        embed_x_history = self.embedding(x, self.history_features)
        # (batch_size, num_target_features, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)
        attention_pooling = []
        for i in range(self.num_history_features):
            attention_seq = self.transformer_layers(torch.cat([torch.squeeze(embed_x_history[:, i, :, :], 1), embed_x_target], dim=1))
            # (batch_size, seq_length + num_target_features, emb_dim)
            attention_pooling.append(attention_seq)
        # (batch_size, num_history_features * (seq_length + num_target_features), emb_dim)
        attention_pooling = torch.cat(attention_pooling, dim=1)

        mlp_in = torch.cat([attention_pooling.flatten(start_dim=1), embed_x_features.flatten(start_dim=1)], dim=1)  # (batch_size, N)
        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))
