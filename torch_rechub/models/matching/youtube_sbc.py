"""
Date: create on 23/05/2022
References:
    paper: (RecSys'2019) Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
    url: https://dl.acm.org/doi/10.1145/3298689.3346996
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import numpy as np
import torch
import torch.nn.functional as F

from ...basic.layers import MLP, EmbeddingLayer


class YoutubeSBC(torch.nn.Module):
    """Sampling-Bias-Corrected Neural Modeling for Matching by Youtube.
    It's a DSSM match model trained by In-batch softmax loss on list-wise samples, and add sample debias module.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the item tower module.
        sample_weight_feature (list[Feature Class]): used for sampling bias corrected in training.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        batch_size (int): same as batch size of DataLoader, used in in-batch sampling
        n_neg (int): the number of negative sample for every positive sample, default to 3. Note it's must smaller than batch_size.
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, item_features, sample_weight_feature, user_params, item_params, batch_size, n_neg=3, temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.sample_weight_feature = sample_weight_feature
        self.n_neg = n_neg
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])
        self.batch_size = batch_size
        self.embedding = EmbeddingLayer(user_features + item_features + sample_weight_feature)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = None

        # in-batch sampling index
        self.index0 = np.repeat(np.arange(batch_size), n_neg + 1)
        self.index1 = np.concatenate([np.arange(i, i + n_neg + 1) for i in range(batch_size)])
        self.index1[np.where(self.index1 >= batch_size)] -= batch_size

    def forward(self, x):
        user_embedding = self.user_tower(x)  # (batch_size, embedding_dim)
        item_embedding = self.item_tower(x)  # (batch_size, embedding_dim)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding


# pred[i, j] means predicted score that user_i give to item_j
        pred = torch.cosine_similarity(user_embedding.unsqueeze(1), item_embedding, dim=2)  # (batch_size, batch_size)

        # get sample weight of items in this batch
        sample_weight = self.embedding(x, self.sample_weight_feature, squeeze_dim=True).squeeze(1)  # (batch_size)
        # Sampling Bias Corrected, using broadcast. (batch_size, batch_size)
        scores = pred - torch.log(sample_weight)

        if user_embedding.shape[0] * (self.n_neg + 1) != self.index0.shape[0]:  # last batch
            batch_size = user_embedding.shape[0]
            index0 = self.index0[:batch_size * (self.n_neg + 1)]
            index1 = self.index1[:batch_size * (self.n_neg + 1)]
            index0[np.where(index0 >= batch_size)] -= batch_size
            index1[np.where(index1 >= batch_size)] -= batch_size
            scores = scores[index0, index1]  # (batch_size, 1 + self.n_neg)
        else:
            # (batch_size, 1 + self.n_neg)
            scores = scores[self.index0, self.index1]

        scores = scores / self.temperature
        return scores.view(-1, self.n_neg + 1)  # (batch_size, 1 + self.n_neg)

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [batch_size, num_features*deep_dims]
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)
        # [batch_size, user_params["dims"][-1]]
        user_embedding = self.user_mlp(input_user)
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        # [batch_size, num_features*embed_dim]
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)
        # [batch_size, item_params["dims"][-1]]
        item_embedding = self.item_mlp(input_item)
        return item_embedding
