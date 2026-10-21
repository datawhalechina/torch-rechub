"""
Date: create on 23/05/2022
References:
    paper: (RecSys'2019) Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
    url: https://dl.acm.org/doi/10.1145/3298689.3346996
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

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

    def forward(self, x):
        user_embedding = self.user_tower(x)  # (batch_size, embedding_dim)
        item_embedding = self.item_tower(x)  # (batch_size, embedding_dim)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding
        if self.mode == "in_batch":
            return user_embedding, item_embedding


# pred[i, j] means predicted score that user_i give to item_j
        pred = torch.cosine_similarity(user_embedding.unsqueeze(1), item_embedding, dim=2)  # (batch_size, batch_size)

        scores = self.correct_inbatch_logits(pred, x)
        rows = torch.arange(scores.size(0), device=scores.device).unsqueeze(1)
        columns = (rows + torch.arange(self.n_neg + 1, device=scores.device)) % scores.size(0)
        return scores[rows, columns] / self.temperature  # [batch_size, 1 + n_neg]

    def correct_inbatch_logits(self, scores, x):
        """Keep sampling-bias correction for both original and full-batch training."""
        weights = self.embedding(x, self.sample_weight_feature, squeeze_dim=True).squeeze(1)
        return scores - torch.log(weights.clamp_min(torch.finfo(scores.dtype).tiny))

    def user_tower(self, x):
        if self.mode == "item":
            return None
        # [batch_size, num_features*deep_dims]
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)
        # [batch_size, user_params["dims"][-1]]
        user_embedding = self.user_mlp(input_user)
        if self.mode == 'in_batch':
            user_embedding = F.normalize(user_embedding, p=2, dim=1)
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        # [batch_size, num_features*embed_dim]
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)
        # [batch_size, item_params["dims"][-1]]
        item_embedding = self.item_mlp(input_item)
        if self.mode == 'in_batch':
            item_embedding = F.normalize(item_embedding, p=2, dim=1)
        return item_embedding
