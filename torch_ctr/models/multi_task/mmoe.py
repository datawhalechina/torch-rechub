"""
Created on 4 May, 2022
Reference: Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD'2018)
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""
import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer, PredictionLayer


class MMOE(nn.Module):

    def __init__(self, features, task_types, n_expert, expert_params, tower_params_list):
        super(MMOE, self).__init__()
        self.features = features
        self.task_types = task_types
        self.n_task = len(task_types)
        self.n_expert = n_expert
        self.embedding = EmbeddingLayer(features)
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.experts = nn.ModuleList(MLP(self.input_dims, **{**expert_params, **{"output_layer": False}}) for i in range(self.n_expert))
        self.gates = nn.ModuleList(MLP(self.input_dims, **{
            "dims": [self.n_expert],
            "activation": "softmax",
            "output_layer": False
        }) for i in range(self.n_task))  #n_gate = n_task
        self.towers = nn.ModuleList(MLP(expert_params["dims"][-1], **tower_params_list[i]) for i in range(self.n_task))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        embed_x = self.embedding(x, self.features, squeeze_dim=True)  #[batch_size, input_dims]
        expert_outs = [expert(embed_x).unsqueeze(1) for expert in self.experts]  #expert_out[i]: [batch_size, 1, expert_dims[-1]]
        expert_outs = torch.cat(expert_outs, dim=1)  #[batch_size, n_expert, expert_dims[-1]]
        gate_outs = [gate(embed_x).unsqueeze(-1) for gate in self.gates]  #gate_out[i]: [batch_size, n_expert, 1]

        ys = []
        for gate_out, tower, predict_layer in zip(gate_outs, self.towers, self.predict_layers):
            expert_weight = torch.mul(gate_out, expert_outs)  #[batch_size, n_expert, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)  #[batch_size, expert_dims[-1]]
            tower_out = tower(expert_pooling)  #[batch_size, 1]
            y = predict_layer(tower_out)  #logit -> proba
            ys.append(y)
        return torch.cat(ys, dim=1)
