"""
Created on 4 May, 2022
Reference: Caruana, R. (1997). Multitask learning. Machine learning, 28(1), 41-75.
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""
import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer, PredictionLayer


class SharedBottom(nn.Module):

    def __init__(self, features, task_types, bottom_params, tower_params_list):
        super(SharedBottom, self).__init__()
        self.features = features
        self.task_types = task_types
        self.embedding = EmbeddingLayer(features)
        self.bottom_dims = sum([fea.embed_dim for fea in features])
        self.bottom_mlp = MLP(self.bottom_dims, **{**bottom_params, **{"output_layer": False}})
        self.towers = nn.ModuleList(MLP(bottom_params["dims"][-1], **tower_params_list[i]) for i in range(len(task_types)))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        input_bottom = self.embedding(x, self.features, squeeze_dim=True)
        x = self.bottom_mlp(input_bottom)

        ys = []
        for tower, predict_layer in zip(self.towers, self.predict_layers):
            tower_out = tower(x)
            y = predict_layer(tower_out)  #regression->keep, binary classification->sigmoid
            ys.append(y)
        return torch.cat(ys, dim=1)
