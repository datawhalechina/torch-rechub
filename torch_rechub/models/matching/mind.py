"""
Date: create on 08/06/2022
References:
    paper: Multi-Interest Network with Dynamic Routing
    url: https://arxiv.org/pdf/1904.08030v1
    code: https://github.com/ShiningCosmos/pytorch_ComiRec/blob/main/MIND.py
Authors: Kai Wang, 306178200@qq.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from ...basic.layers import MLP, CapsuleNetwork, EmbeddingLayer, MultiInterestSA


class MIND(torch.nn.Module):
    """The match model mentioned in `Multi-Interest Network with Dynamic Routing` paper.
    It's a ComirecDR match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        history_features (list[Feature Class]): training history
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        max_length (int): max sequence length of input item sequence
        temperature (float): temperature factor for similarity score, default to 1.0.
        interest_num （int): interest num
    """

    def __init__(self, user_features, history_features, item_features, neg_item_feature, max_length, temperature=1.0, interest_num=4):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.interest_num = interest_num
        self.max_length = max_length
        self.user_dims = sum([fea.embed_dim for fea in user_features + history_features])

        self.embedding = EmbeddingLayer(user_features + item_features + history_features)
        self.capsule = CapsuleNetwork(self.history_features[0].embed_dim, self.max_length, bilinear_type=0, interest_num=self.interest_num)
        self.convert_user_weight = nn.Parameter(torch.rand(self.user_dims, self.history_features[0].embed_dim), requires_grad=True)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        pos_item_embedding = item_embedding[:, 0, :]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        k_index = torch.argmax(dot_res, dim=1)
        best_interest_emb = torch.rand(user_embedding.shape[0], user_embedding.shape[2]).to(user_embedding.device)
        for k in range(user_embedding.shape[0]):
            best_interest_emb[k, :] = user_embedding[k, k_index[k], :]
        best_interest_emb = best_interest_emb.unsqueeze(1)

        y = torch.mul(best_interest_emb, item_embedding).sum(dim=1)
        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True).unsqueeze(1)  # [batch_size, num_features*deep_dims]
        input_user = input_user.expand([input_user.shape[0], self.interest_num, input_user.shape[-1]])

        history_emb = self.embedding(x, self.history_features).squeeze(1)
        mask = self.gen_mask(x)
        multi_interest_emb = self.capsule(history_emb, mask)

        input_user = torch.cat([input_user, multi_interest_emb], dim=-1)

        # user_embedding = self.user_mlp(input_user).unsqueeze(1)
        # #[batch_size, interest_num, embed_dim]
        user_embedding = torch.matmul(input_user, self.convert_user_weight)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "user":
            # inference embedding mode -> [batch_size, interest_num, embed_dim]
            return user_embedding
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

    def gen_mask(self, x):
        his_list = x[self.history_features[0].name]
        mask = (his_list > 0).long()
        return mask
