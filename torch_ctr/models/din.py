"""
Created on April 23, 2022
Update on April 30, 2022
Reference: "Deep Interest Network for Click-Through Rate Prediction", KDD, 2018
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""
import torch
import torch.nn as nn

from ..basic.layers import EmbeddingLayer, MultiLayerPerceptron


class DIN(nn.Module):
    """Deep Interest Network
    Args:
        features (list of Feature Class):the user profile features and context features in origin paper, exclude history and target features
    """

    def __init__(self,
                 features,
                 history_features,
                 target_features,
                 mlp_params={
                     "dims": [200, 80],
                     "activation": "dice"
                 },
                 attention_mlp_params={
                     "dims": [36],
                     "activation": "dice"
                 }):
        super(DIN, self).__init__()
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.num_history_features = len(history_features)

        self.all_dims = sum([fea.embed_dim for fea in features + history_features + target_features])
        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.mlp = MultiLayerPerceptron(self.all_dims, **mlp_params)

        self.attention_layers = nn.ModuleList([ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features])

    def forward(self, x):
        """
        The original paper without dense feature.
        :param x_sequence: Long tensor of size ``(batch_size, num_seq_fields, seq_length)``
                        # each fields is a tensor list
        :param x_candidate: Long tensor of size ``(batch_size, num_seq_fields)``
                        # each fields is a candidate field id for x_sequence
        :param x_sparse: Long tensor of size ``(batch_size, num_sparse_fields)``
                        # sparse: user_profile_feature, candidate_feature, context_feature
        """

        embed_x_features = self.embedding(x, self.features)  #(batch_size, num_features, emb_dim)
        embed_x_history = self.embedding(x, self.history_features)  #(batch_size, num_history_features, seq_length, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)  #(batch_size, num_target_features, emb_dim)
        attention_pooling = []
        for i in range(self.num_history_features):  #为每一个序列特征都训练一个独立的激活单元 eg.历史序列item id，历史序列item id品牌
            attention_seq = self.attention_layers[i](embed_x_history[:, i, :, :], embed_x_target[:, i, :])
            attention_pooling.append(attention_seq.unsqueeze(1))  #(batch_size, 1, emb_dim)
        attention_pooling = torch.cat(attention_pooling, dim=1)  #(batch_size, num_history_features, emb_dim)

        mlp_in = torch.cat([attention_pooling.flatten(start_dim=1),
                            embed_x_target.flatten(start_dim=1),
                            embed_x_features.flatten(start_dim=1)],
                           dim=1)  #(batch_size, N)

        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))


class ActivationUnit(torch.nn.Module):
    """
        DIN Attention Layer
    """

    def __init__(self, emb_dim, dims=[36], activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax
        self.attention = MultiLayerPerceptron(4 * self.emb_dim, dims, activation=activation)

    def forward(self, history, candidate):
        """
        :param history: Long tensor of size ``(batch_size, seq_length, emb_dim) ``
        :param candidate: Long tensor of size ``(batch_size, emb_dim)``
        """
        seq_length = history.size(1)
        candidate = candidate.unsqueeze(1).expand(-1, seq_length, -1)  #batch_size,seq_length,emb_dim
        att_input = torch.cat([candidate, history, candidate - history, candidate * history], dim=-1)  # batch_size,seq_length,4*emb_dim
        att_weight = self.attention(att_input.view(-1, 4 * self.emb_dim))  #  #(batch_size*seq_length,4*emb_dim)
        att_weight = att_weight.view(-1, seq_length)  #(batch_size*seq_length, 1) -> (batch_size,seq_length)
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1)
        # (batch_size, seq_length, 1) * (batch_size, seq_length, emb_dim)
        output = (att_weight.unsqueeze(-1) * history).sum(dim=1)  #(batch_size,emb_dim)
        return output