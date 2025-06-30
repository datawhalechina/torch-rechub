"""
Date: created on 17/09/2022
References:
    paper: STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation
    url: https://dl.acm.org/doi/10.1145/3219819.3219950
    official Tensorflow implementation: https://github.com/uestcnlp/STAMP
Authors: Bo Kang, klinux@live.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STAMP(nn.Module):

    def __init__(self, item_history_feature, weight_std, emb_std):
        super(STAMP, self).__init__()

        # item embedding layer
        self.item_history_feature = item_history_feature
        n_items, item_emb_dim, = item_history_feature.vocab_size, item_history_feature.embed_dim
        self.item_emb = nn.Embedding(n_items, item_emb_dim, padding_idx=0)

        # weights and biases for attention computation
        self.w_0 = nn.Parameter(torch.zeros(item_emb_dim, 1))
        self.w_1_t = nn.Parameter(torch.zeros(item_emb_dim, item_emb_dim))
        self.w_2_t = nn.Parameter(torch.zeros(item_emb_dim, item_emb_dim))
        self.w_3_t = nn.Parameter(torch.zeros(item_emb_dim, item_emb_dim))
        self.b_a = nn.Parameter(torch.zeros(item_emb_dim))
        self._init_parameter_weights(weight_std)

        # mlp layers
        self.f_s = nn.Sequential(nn.Tanh(), nn.Linear(item_emb_dim, item_emb_dim))
        self.f_t = nn.Sequential(nn.Tanh(), nn.Linear(item_emb_dim, item_emb_dim))
        self.emb_std = emb_std
        self.apply(self._init_module_weights)

    def _init_parameter_weights(self, weight_std):
        nn.init.normal_(self.w_0, std=weight_std)
        nn.init.normal_(self.w_1_t, std=weight_std)
        nn.init.normal_(self.w_2_t, std=weight_std)
        nn.init.normal_(self.w_3_t, std=weight_std)

    def _init_module_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(std=self.emb_std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(std=self.emb_std)

    def forward(self, input_dict):
        # Index the embeddings for the items in the session
        input = input_dict[self.item_history_feature.name]
        value_mask = (input != 0).unsqueeze(-1)
        value_counts = value_mask.sum(dim=1, keepdim=True).squeeze(-1)
        item_emb_batch = self.item_emb(input) * value_mask

        # Index the embeddings of the latest clicked items
        x_t = self.item_emb(torch.gather(input, 1, value_counts - 1))

        # Eq. 2, user's general interest in the current session
        m_s = ((item_emb_batch).sum(1) / value_counts).unsqueeze(1)

        # Eq. 7, compute attention coefficient
        a = F.normalize(torch.exp(torch.sigmoid(item_emb_batch @ self.w_1_t + x_t @ self.w_2_t + m_s @ self.w_3_t + self.b_a) @ self.w_0) * value_mask, p=1, dim=1)

        # Eq. 8, compute user's attention-based interests
        m_a = (a * item_emb_batch).sum(1) + m_s.squeeze(1)

        # Eq. 3, compute the output state of the general interest
        h_s = self.f_s(m_a)

        # Eq. 9, compute the output state of the short-term interest
        h_t = self.f_t(x_t).squeeze(1)

        # Eq. 4, compute candidate scores
        z = h_s * h_t @ self.item_emb.weight.T

        return z
