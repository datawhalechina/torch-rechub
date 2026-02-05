"""
Date: created on 06/09/2022
References:
    paper: Neural Attentive Session-based Recommendation
    url: http://arxiv.org/abs/1711.04725
    official Theano implementation: https://github.com/lijingsdu/sessionRec_NARM
    another Pytorch implementation: https://github.com/Wang-Shuo/Neural-Attentive-Session-Based-Recommendation-PyTorch
Authors: Bo Kang, klinux@live.com
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from torch import sigmoid
from torch.nn import GRU, Dropout, Embedding, Parameter


class NARM(nn.Module):

    def __init__(self, item_history_feature, hidden_dim, emb_dropout_p, session_rep_dropout_p, item_feature=None):
        super(NARM, self).__init__()

        # item embedding layer
        self.item_history_feature = item_history_feature
        self.item_feature = item_feature  # Optional: for in-batch negative sampling
        self.item_emb = Embedding(item_history_feature.vocab_size, item_history_feature.embed_dim, padding_idx=0)
        self.mode = None  # For inference: "user" or "item"

        # embedding dropout layer
        self.emb_dropout = Dropout(emb_dropout_p)

        # gru unit
        self.gru = GRU(input_size=item_history_feature.embed_dim, hidden_size=hidden_dim)

        # attention projection matrices
        self.a_1, self.a_2 = Parameter(torch.randn(hidden_dim, hidden_dim)), Parameter(torch.randn(hidden_dim, hidden_dim))

        # attention context vector
        self.v = Parameter(torch.randn(hidden_dim, 1))

        # session representation dropout layer
        self.session_rep_dropout = Dropout(session_rep_dropout_p)

        # bilinear projection matrix
        self.b = Parameter(torch.randn(item_history_feature.embed_dim, hidden_dim * 2))

    def _compute_session_repr(self, input_dict):
        """Compute session representation (user embedding before bilinear transform)."""
        input = input_dict[self.item_history_feature.name]
        value_mask = (input != 0)
        value_counts = value_mask.sum(dim=1, keepdim=False).to("cpu").detach()
        embs = rnn_utils.pack_padded_sequence(self.emb_dropout(self.item_emb(input)), value_counts, batch_first=True, enforce_sorted=False)

        h, h_t = self.gru(embs)
        h_t = h_t.permute(1, 0, 2)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)

        c_g = h_t.squeeze(1)
        q = sigmoid(h_t @ self.a_1.T + h @ self.a_2.T) @ self.v
        alpha = torch.exp(q) * value_mask.unsqueeze(-1)
        alpha /= alpha.sum(dim=1, keepdim=True)
        c_l = (alpha * h).sum(1)

        c = self.session_rep_dropout(torch.hstack((c_g, c_l)))
        return c

    def user_tower(self, x):
        """Compute user embedding for in-batch negative sampling."""
        if self.mode == "item":
            return None
        c = self._compute_session_repr(x)
        user_emb = c @ self.b.T  # [batch_size, embed_dim]
        if self.mode == "user":
            return user_emb
        return user_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]

    def item_tower(self, x):
        """Compute item embedding for in-batch negative sampling."""
        if self.mode == "user":
            return None
        if self.item_feature is not None:
            item_ids = x[self.item_feature.name]
            item_emb = self.item_emb(item_ids)  # [batch_size, embed_dim]
            if self.mode == "item":
                return item_emb
            return item_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]
        return None

    def forward(self, input_dict):
        # Support inference mode
        if self.mode == "user":
            return self.user_tower(input_dict)
        if self.mode == "item":
            return self.item_tower(input_dict)

        # In-batch negative sampling mode
        if self.item_feature is not None:
            user_emb = self.user_tower(input_dict)  # [batch_size, 1, embed_dim]
            item_emb = self.item_tower(input_dict)  # [batch_size, 1, embed_dim]
            return torch.mul(user_emb, item_emb).sum(dim=-1).squeeze()

        # Original behavior: compute scores for all items
        c = self._compute_session_repr(input_dict)
        s = c @ self.b.T @ self.item_emb.weight.T
        return s
