"""
Date: create on 03/06/2022
References:
    paper: SESSION-BASED RECOMMENDATIONS WITH RECURRENT NEURAL NETWORKS
    url: http://arxiv.org/abs/1511.06939
Authors: Kai Wang, 306178200@qq.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from ...basic.layers import MLP, EmbeddingLayer


class GRU4Rec(torch.nn.Module):
    """The match model mentioned in `Deep Neural Networks for YouTube Recommendations` paper.
    It's a DSSM match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        history_features (list[Feature Class]): training history
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, history_features, item_features, neg_item_feature, user_params, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features + history_features])

        self.embedding = EmbeddingLayer(user_features + item_features + history_features)
        self.gru = nn.GRU(input_size=history_features[0].embed_dim, hidden_size=history_features[0].embed_dim, num_layers=user_params.get('num_layers', 2), batch_first=True, bias=False)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        y = torch.mul(user_embedding, item_embedding).sum(dim=1)

        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [batch_size, num_features*deep_dims]
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)

        history_emb = self.embedding(x, self.history_features).squeeze(1)
        _, history_emb = self.gru(history_emb)
        history_emb = history_emb[-1]

        input_user = torch.cat([input_user, history_emb], dim=-1)

        user_embedding = self.user_mlp(input_user).unsqueeze(1)  # [batch_size, 1, embed_dim]
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "user":
            # inference embedding mode -> [batch_size, embed_dim]
            return user_embedding.squeeze(1)
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  # [batch_size, 1, embed_dim]
        pos_embedding = F.normalize(pos_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "item":  # inference embedding mode
            return pos_embedding.squeeze(1)  # [batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature, squeeze_dim=False).squeeze(1)  # [batch_size, n_neg_items, embed_dim]
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)  # L2 normalize
        # [batch_size, 1+n_neg_items, embed_dim]
        return torch.cat((pos_embedding, neg_embeddings), dim=1)
