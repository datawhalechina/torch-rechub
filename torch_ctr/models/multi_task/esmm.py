"""
Created on 4 May, 2022
Reference: Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate (SIGIR'2018)
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""
import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer, PredictionLayer


class ESMM(nn.Module):

    def __init__(self, user_features, item_features, cvr_params, ctr_params):
        super(ESMM, self).__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.tower_dims = user_features[0].embed_dim + item_features[0].embed_dim  #the dims after Field-wise Pooling Layer
        self.tower_cvr = MLP(self.tower_dims, **cvr_params)
        self.tower_ctr = MLP(self.tower_dims, **ctr_params)

    def forward(self, x):
        embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).sum(dim=1)  #[batch_size, embed_dim]
        embed_item_features = self.embedding(x, self.item_features, squeeze_dim=False).sum(dim=1)  #[batch_size, embed_dim]
        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)
        cvr_logit = self.tower_cvr(input_tower)
        ctr_logit = self.tower_ctr(input_tower)
        cvr_pred = torch.sigmoid(cvr_logit)
        ctr_pred = torch.sigmoid(ctr_logit)
        ctcvr_pred = torch.mul(cvr_pred, cvr_pred)

        ys = [cvr_pred, ctr_pred, ctcvr_pred]
        return torch.cat(ys, dim=1)
