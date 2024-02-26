"""
Date: create on 26/02/2024, update on 30/04/2022
References: 
    paper: Behavior Sequence Transformer for E-commerce Recommendation in Alibaba
    url: https://arxiv.org/pdf/1905.06874
    code: https://github.com/jiwidi/Behavior-Sequence-Transformer-Pytorch/blob/master/pytorch_bst.ipynb
Authors: Tao Fan, thisisevy@foxmail.com
"""

import torch
# import torch.utils.data as data
# from torchvision import transforms
# import ast
# from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
from ...basic.layers import EmbeddingLayer, MLP




class BST(nn.Module):
    """Behavior Sequence Transformer
    features (list): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
        history_features (list): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
    """
    def __init__(self, features, history_features, target_features, mlp_params):
        super().__init__()
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.num_history_features = len(history_features)
        # self.positional_embedding = PositionalEmbedding(8, 9)
        self.all_dims = sum([fea.embed_dim for fea in features + history_features + target_features])
        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.attention_layers = nn.TransformerEncoderLayer(64, 8, dropout=0.2) # nn.ModuleList([ActivationUnit(fea.embed_dim, **attention_mlp_params) for fea in self.history_features])
        self.mlp = MLP(self.all_dims, activation="leakyrelu", **mlp_params) # # 定义模型，模型的参数需要我们之前的feature类，用于构建模型的输入层，mlp指定模型后续DNN的结构


    def forward(self, x):
        embed_x_features = self.embedding(x, self.features)  #(batch_size, num_features, emb_dim)
        embed_x_history = self.embedding(x, self.history_features)  #(batch_size, num_history_features, seq_length, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)  #(batch_size, num_target_features, emb_dim)
        # positional_embedding = self.positional_embedding(torch.cat([embed_x_history,embed_x_target],dim=2))
        # embed_x_history = torch.cat((embed_x_history, positional_embedding), dim=2)
        attention_pooling = []
        for i in range(self.num_history_features):
            attention_seq = self.attention_layers(embed_x_history[:, i, :, :])
            attention_pooling.append(attention_seq)  #(batch_size, seq_length, emb_dim)
        attention_pooling = torch.stack(attention_pooling,dim=1).mean(dim=2)  #(batch_size, num_history_features, emb_dim)
        # print(attention_pooling.shape, embed_x_target.shape, embed_x_features.shape)
        mlp_in = torch.cat([
            attention_pooling.flatten(start_dim=1),
            embed_x_target.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1)
        ],
                           dim=1)  #(batch_size, N)
        # print(mlp_in.shape)
        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))
    
class PositionalEmbedding(nn.Module):
    """
    Computes positional embedding following "Attention is all you need"
    """

    def __init__(self, max_len, d_model):
        super().__init__()

        # Compute the positional encodings once in log space.
        self.pe = nn.Embedding(max_len, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)

