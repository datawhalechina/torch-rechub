"""
Date: create on 05/05/2022
References:
    paper: (RecSys'2020) Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations
    url: https://dl.acm.org/doi/abs/10.1145/3383313.3412236
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn as nn

from ...basic.layers import MLP, EmbeddingLayer, PredictionLayer


class PLE(nn.Module):
    """Progressive Layered Extraction model.

    Args:
        features (list): the list of `Feature Class`, training by the expert and tower module.
        task_types (list): types of tasks, only support `["classfication", "regression"]`.
        n_level (int): the  number of CGC layer.
        n_expert_specific (int): the number of task-specific expert net.
        n_expert_shared (int): the number of task-shared expert net.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
        tower_params_list (list): the list of tower params dict, the keys same as expert_params.
    """

    def __init__(self, features, task_types, n_level, n_expert_specific, n_expert_shared, expert_params, tower_params_list):
        super().__init__()
        self.features = features
        self.n_task = len(task_types)
        self.task_types = task_types
        self.n_level = n_level
        self.input_dims = sum([fea.embed_dim for fea in features])
        self.embedding = EmbeddingLayer(features)
        self.cgc_layers = nn.ModuleList(CGC(i + 1, n_level, self.n_task, n_expert_specific, n_expert_shared, self.input_dims, expert_params) for i in range(n_level))
        self.towers = nn.ModuleList(MLP(expert_params["dims"][-1], output_layer=False, **tower_params_list[i]) for i in range(self.n_task))
        self.predict_layers = nn.ModuleList(PredictionLayer(task_type) for task_type in task_types)

    def forward(self, x):
        # [batch_size, input_dims]
        embed_x = self.embedding(x, self.features, squeeze_dim=True)
        ple_inputs = [embed_x] * (self.n_task + 1)
        ple_outs = []
        for i in range(self.n_level):
            # ple_outs[i]: [batch_size, expert_dims[-1]]
            ple_outs = self.cgc_layers[i](ple_inputs)
            ple_inputs = ple_outs


# predict
        ys = []
        for ple_out, tower, predict_layer in zip(ple_outs, self.towers, self.predict_layers):
            tower_out = tower(ple_out)  # [batch_size, 1]
            y = predict_layer(tower_out)  # logit -> proba
            ys.append(y)
        return torch.cat(ys, dim=1)


class CGC(nn.Module):
    """Customized Gate Control (CGC) Model mentioned in PLE paper.

    Args:
        cur_level (int): the current level of CGC in PLE.
        n_level (int): the  number of CGC layer.
        n_task (int): the number of tasks.
        n_expert_specific (int): the number of task-specific expert net.
        n_expert_shared (int): the number of task-shared expert net.
        input_dims (int): the input dims of the xpert module in current CGC layer.
        expert_params (dict): the params of all the expert module, keys include:`{"dims":list, "activation":str, "dropout":float}.
    """

    def __init__(self, cur_level, n_level, n_task, n_expert_specific, n_expert_shared, input_dims, expert_params):
        super().__init__()
        self.cur_level = cur_level  # the CGC level of PLE
        self.n_level = n_level
        self.n_task = n_task
        self.n_expert_specific = n_expert_specific
        self.n_expert_shared = n_expert_shared
        self.n_expert_all = n_expert_specific * self.n_task + n_expert_shared
        # the first layer expert dim is the input data dim other expert dim
        input_dims = input_dims if cur_level == 1 else expert_params["dims"][-1]
        self.experts_specific = nn.ModuleList(MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.n_task * self.n_expert_specific))
        self.experts_shared = nn.ModuleList(MLP(input_dims, output_layer=False, **expert_params) for _ in range(self.n_expert_shared))
        self.gates_specific = nn.ModuleList(MLP(input_dims, **{"dims": [self.n_expert_specific + self.n_expert_shared], "activation": "softmax", "output_layer": False}) for _ in range(self.n_task))  # n_gate_specific = n_task
        if cur_level < n_level:
            self.gate_shared = MLP(input_dims, **{"dims": [self.n_expert_all], "activation": "softmax", "output_layer": False})  # n_gate_specific = n_task

    def forward(self, x_list):
        expert_specific_outs = []  # expert_out[i]: [batch_size, 1, expert_dims[-1]]
        for i in range(self.n_task):
            expert_specific_outs.extend([expert(x_list[i]).unsqueeze(1) for expert in self.experts_specific[i * self.n_expert_specific:(i + 1) * self.n_expert_specific]])
        # x_list[-1]: the input for shared experts
        expert_shared_outs = [expert(x_list[-1]).unsqueeze(1) for expert in self.experts_shared]
        # gate_out[i]: [batch_size, n_expert_specific+n_expert_shared, 1]
        gate_specific_outs = [gate(x_list[i]).unsqueeze(-1) for i, gate in enumerate(self.gates_specific)]
        cgc_outs = []
        for i, gate_out in enumerate(gate_specific_outs):
            cur_expert_list = expert_specific_outs[i * self.n_expert_specific:(i + 1) * self.n_expert_specific] + expert_shared_outs
            # [batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_concat = torch.cat(cur_expert_list, dim=1)
            # [batch_size, n_expert_specific+n_expert_shared, expert_dims[-1]]
            expert_weight = torch.mul(gate_out, expert_concat)
            # [batch_size, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)
            cgc_outs.append(expert_pooling)  # length: n_task
        if self.cur_level < self.n_level:  # not the last layer
            gate_shared_out = self.gate_shared(x_list[-1]).unsqueeze(-1)  # [batch_size, n_expert_all, 1]
            expert_concat = torch.cat(expert_specific_outs + expert_shared_outs, dim=1)  # [batch_size, n_expert_all, expert_dims[-1]]
            # [batch_size, n_expert_all, expert_dims[-1]]
            expert_weight = torch.mul(gate_shared_out, expert_concat)
            # [batch_size, expert_dims[-1]]
            expert_pooling = torch.sum(expert_weight, dim=1)
            cgc_outs.append(expert_pooling)  # length: n_task+1

        return cgc_outs
