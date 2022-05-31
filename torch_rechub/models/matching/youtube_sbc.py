"""
Date: create on 23/05/2022
References: 
    paper: (RecSys'2019) Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations
    url: https://dl.acm.org/doi/10.1145/3298689.3346996
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch

from ...basic.layers import MLP, EmbeddingLayer


class YoutubeSBC(torch.nn.Module):
    """Sampling-Bias-Corrected Neural Modeling for Matching by Youtube. 
    It's a DSSM match model trained by In-batch softmax loss on list-wise samples, and add sample debias module.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        item_features (list[Feature Class]): training by the item tower module.
        sample_weight_feature (list[Feature Class]): used for sampleing bias corrected in training.
        user_params (dict): the params of the User Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        item_params (dict): the params of the Item Tower module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}.
        n_neg (int): the number of negative sample for every positive sample, default to 3. Note it's must smaller than batch_size.
        sim_func (str): similarity function, includes `["cosine", "dot"]`, default to "cosine".
        temperature (float): temperature factor for similarity score, default to 1.0.

    """

    def __init__(self,
                 user_features,
                 item_features,
                 sample_weight_feature,
                 user_params,
                 item_params,
                 n_neg=3,
                 sim_func="cosine",
                 temperature=1.0):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.sample_weight_feature = sample_weight_feature
        self.n_neg = n_neg
        self.sim_func = sim_func
        self.temperature = temperature
        self.user_dims = sum([fea.embed_dim for fea in user_features])
        self.item_dims = sum([fea.embed_dim for fea in item_features])

        self.embedding = EmbeddingLayer(user_features + item_features + sample_weight_feature)
        self.user_mlp = MLP(self.user_dims, output_layer=False, **user_params)
        self.item_mlp = MLP(self.item_dims, output_layer=False, **item_params)
        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding
        if self.sim_func == "cosine":
            y = torch.cosine_similarity(user_embedding, item_embedding, dim=1)
        elif self.sim_func == "dot":
            y = torch.mul(user_embedding, item_embedding).sum(dim=1)
        else:
            raise ValueError("similarity function only support %s, but got %s" % (["cosine", "dot"], self.sim_func))

        sample_weight = self.embedding(x, self.sample_weight_feature, squeeze_dim=True).squeeze(1)

        y = y - torch.log(sample_weight)  #Sampling Bias Corrected

        #in-batch negative sample
        #!! TODO: use mask matrix. It's slow now.
        batch_size = y.size(0)
        scores = torch.ones(batch_size, 1 + self.n_neg, device=y.device)  #positive sample in the first position.
        y_expand = torch.cat((y, y))
        for i in range(batch_size):
            scores[i, :] = torch.cat((y_expand[i].view(-1), y_expand[i + 1:i + 1 + self.n_neg]))
        scores = scores / self.temperature
        return scores  #(batch_size, 4)

    def user_tower(self, x):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True)  #[batch_size, num_features*deep_dims]
        user_embedding = self.user_mlp(input_user)  #[batch_size, user_params["dims"][-1]]
        return user_embedding

    def item_tower(self, x):
        if self.mode == "user":
            return None
        input_item = self.embedding(x, self.item_features, squeeze_dim=True)  #[batch_size, num_features*embed_dim]
        item_embedding = self.item_mlp(input_item)  #[batch_size, item_params["dims"][-1]]
        return item_embedding