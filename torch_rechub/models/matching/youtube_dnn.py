"""
Date: create on 23/05/2022
References: 
    paper: (RecSys'2016) Deep Neural Networks for YouTube Recommendations
    url: https://dl.acm.org/doi/10.1145/2959100.2959190
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ...basic.layers import MLP, EmbeddingLayer


class YoutubeDNN(torch.nn.Module):
    """The match model mentioned in `Deep Neural Networks for YouTube Recommendations` paper.
    It's a DSSM match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        sim_func (str): similarity function, includes `["cosine", "dot"]`, default to "cosine".
        temperature (float): temperature factor for similarity score, default to 1.0.
    """

    def __init__(self, user_features, item_features, neg_item_feature, user_params, sim_func="cosine", temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.neg_item_feature = neg_item_feature
        self.sim_func = sim_func
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])

        self.embedding = EmbeddingLayer(user_features + item_features)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding
        if self.sim_func == "cosine":
            y = torch.cosine_similarity(user_embedding, item_embedding, dim=-1)  #[batch_size, 1+n_neg_items, embed_dim]
        elif self.sim_func == "dot":
            y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        else:
            raise ValueError("similarity function only support %s, but got %s" % (["cosine", "dot"], self.sim_func))
        y = y / self.temperature
        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user).unsqueeze(1)  #[batch_size, 1, embed_dim]
        if self.mode == "user":
            return user_embedding.squeeze(1)  #inference embedding mode -> [batch_size, embed_dim]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  #[batch_size, 1, embed_dim]
        if self.mode == "item":  #inference embedding mode
            return pos_embedding.squeeze(1)  #[batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature,
                                        squeeze_dim=False).squeeze(1)  #[batch_size, n_neg_items, embed_dim]
        return torch.cat((pos_embedding, neg_embeddings), dim=1)  #[batch_size, 1+n_neg_items, embed_dim]
