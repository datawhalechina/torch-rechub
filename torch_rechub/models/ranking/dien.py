"""
Date: create on 01/05/2024
References:
    paper: (AAAI'2019) Deep Interest Evolution Network for Click-Through Rate Prediction
    url: https://arxiv.org/pdf/1809.03672
Authors: Tao Fan, thisisevy@foxmail.com
"""

import torch
from torch import nn
from torch.nn import Parameter, init

from ...basic.layers import MLP, EmbeddingLayer


class AUGRU(nn.Module):

    def __init__(self, embed_dim):
        super(AUGRU, self).__init__()
        self.embed_dim = embed_dim
        # 初始化AUGRU单元
        self.augru_cell = AUGRU_Cell(self.embed_dim)

    def forward(self, x, item):
        '''
        :param x: 输入的序列向量，维度为 [ batch_size, seq_lens, embed_dim ]
        :param item: 目标物品的向量
        :return: outs: 所有AUGRU单元输出的隐藏向量[ batch_size, seq_lens, embed_dim ]
                 h: 最后一个AUGRU单元输出的隐藏向量[ batch_size, embed_dim ]
        '''
        outs = []
        h = None
        # 开始循环，x.shape[1]是序列的长度
        for i in range(x.shape[1]):
            if h is None:
                # 初始化第一层的输入h
                h = Parameter(torch.rand(x.shape[0], self.embed_dim).to(x.device))
            h = self.augru_cell(x[:, i], h, item)
            outs.append(torch.unsqueeze(h, dim=1))
        outs = torch.cat(outs, dim=1)
        return outs, h


# AUGRU单元
class AUGRU_Cell(nn.Module):

    def __init__(self, embed_dim):
        """
        :param embed_dim: 输入向量的维度
        """
        super(AUGRU_Cell, self).__init__()

        # 初始化更新门的模型参数
        self.Wu = Parameter(torch.rand(embed_dim, embed_dim))
        self.Uu = Parameter(torch.rand(embed_dim, embed_dim))
        self.bu = init.xavier_uniform_(Parameter(torch.empty(1, embed_dim)))

        # 初始化重置门的模型参数
        self.Wr = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.Ur = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.br = init.xavier_uniform_(Parameter(torch.empty(1, embed_dim)))

        # 初始化计算h~的模型参数
        self.Wh = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.Uh = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.bh = init.xavier_uniform_(Parameter(torch.empty(1, embed_dim)))

        # 初始化注意计算里的模型参数
        self.Wa = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))


# 注意力的计算

    def attention(self, x, item):
        '''
        :param x: 输入的序列中第t个向量 [ batch_size, embed_dim ]
        :param item: 目标物品的向量 [ batch_size, embed_dim ]
        :return: 注意力权重 [ batch_size, 1 ]
        '''
        hW = torch.matmul(x, self.Wa)
        hWi = torch.sum(hW * item, dim=1)
        hWi = torch.unsqueeze(hWi, 1)
        return torch.softmax(hWi, dim=1)

    def forward(self, x, h_1, item):
        '''
        :param x:  输入的序列中第t个物品向量 [ batch_size, embed_dim ]
        :param h_1:  上一个AUGRU单元输出的隐藏向量 [ batch_size, embed_dim ]
        :param item: 目标物品的向量 [ batch_size, embed_dim ]
        :return: h 当前层输出的隐藏向量 [ batch_size, embed_dim ]
        '''
        # [ batch_size, embed_dim ]
        u = torch.sigmoid(torch.matmul(x, self.Wu) + torch.matmul(h_1, self.Uu) + self.bu)
        # [ batch_size, embed_dim ]
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_1, self.Ur) + self.br)
        # [ batch_size, embed_dim ]
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + r * torch.matmul(h_1, self.Uh) + self.bh)
        # [ batch_size, 1 ]
        a = self.attention(x, item)
        # [ batch_size, embed_dim ]
        u_hat = a * u
        # [ batch_size, embed_dim ]
        h = (1 - u_hat) * h_1 + u_hat * h_hat
        # [ batch_size, embed_dim ]
        return h


class DIEN(nn.Module):
    """Deep Interest Evolution Network
    Args:
        features (list): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
        history_features (list): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        history_labels (list): the list of history_features whether it is clicked history or not. It should be 0 or 1.
        alpha (float): the weighting of auxiliary loss.
    """

    def __init__(self, features, history_features, target_features, mlp_params, history_labels, alpha=0.2):
        super().__init__()
        self.alpha = alpha  # 计算辅助损失函数时的权重
        self.features = features
        self.history_features = history_features
        self.target_features = target_features
        self.num_history_features = len(history_features)
        self.all_dims = sum([fea.embed_dim for fea in features + history_features + target_features])
        # self.GRU = nn.GRU(batch_first=True)
        self.embedding = EmbeddingLayer(features + history_features + target_features)
        self.interest_extractor_layers = nn.ModuleList([nn.GRU(fea.embed_dim, fea.embed_dim, batch_first=True) for fea in self.history_features])
        self.interest_evolving_layers = nn.ModuleList([AUGRU(fea.embed_dim) for fea in self.history_features])

        self.mlp = MLP(self.all_dims, activation="dice", **mlp_params)
        self.history_labels = torch.Tensor(history_labels)
        self.BCELoss = nn.BCELoss()


# # 注意力计算中的线性层
# self.attention_liner = nn.Linear(self.embed_dim, t)
# # AFM公式中的h
# self.h = init.xavier_uniform_(Parameter(torch.empty(t, 1)))
# # AFM公式中的p
# self.p = init.xavier_uniform_(Parameter(torch.empty(self.embed_dim, 1)))

    def auxiliary(self, outs, history_features, history_labels):
        '''
        :param history_features: 历史序列物品的向量 [ batch_size, len_seqs, dim ]
        :param outs: 兴趣抽取层GRU网络输出的outs [ batch_size, len_seqs, dim ]
        :param history_labels: 历史序列物品标注 [ batch_size, len_seqs, 1 ]
        :return: 辅助损失函数
        '''
        # [ batch_size * len_seqs, dim ]
        history_features = history_features.reshape(-1, history_features.shape[2])
        # [ batch_size * len_seqs, dim ]
        outs = outs.reshape(-1, outs.shape[2])
        # [ batch_size * len_seqs ]
        out = torch.sum(outs * history_features, dim=1)
        # [ batch_size * len_seqs, 1 ]
        out = torch.unsqueeze(torch.sigmoid(out), 1)
        # [ batch_size * len_seqs,1 ]
        history_labels = history_labels.reshape(-1, 1).float()
        return self.BCELoss(out, history_labels)

    def forward(self, x):
        # (batch_size, num_features, emb_dim)
        embed_x_features = self.embedding(x, self.features)
        # (batch_size, num_history_features, seq_length, emb_dim)
        embed_x_history = self.embedding(x, self.history_features)
        # (batch_size, num_target_features, emb_dim)
        embed_x_target = self.embedding(x, self.target_features)

        interest_extractor = []
        auxi_loss = 0
        for i in range(self.num_history_features):
            outs, _ = self.interest_extractor_layers[i](embed_x_history[:, i, :, :])
            # 利用GRU输出的outs得到辅助损失函数
            auxi_loss += self.auxiliary(outs, embed_x_history[:, i, :, :], self.history_labels)
            # (batch_size, 1, seq_length, emb_dim)
            interest_extractor.append(outs.unsqueeze(1))
        # (batch_size, num_history_features, seq_length, emb_dim)
        interest_extractor = torch.cat(interest_extractor, dim=1)
        interest_evolving = []
        for i in range(self.num_history_features):
            _, h = self.interest_evolving_layers[i](interest_extractor[:, i, :, :], embed_x_target[:, i, :])
            interest_evolving.append(h.unsqueeze(1))  # (batch_size, 1, emb_dim)
        # (batch_size, num_history_features, emb_dim)
        interest_evolving = torch.cat(interest_evolving, dim=1)

        mlp_in = torch.cat([interest_evolving.flatten(start_dim=1), embed_x_target.flatten(start_dim=1), embed_x_features.flatten(start_dim=1)], dim=1)  # (batch_size, N)
        y = self.mlp(mlp_in)

        return torch.sigmoid(y.squeeze(1)), self.alpha * auxi_loss
