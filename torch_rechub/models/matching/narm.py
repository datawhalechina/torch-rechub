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

    def __init__(self, item_history_feature, hidden_dim, emb_dropout_p, session_rep_dropout_p):
        super(NARM, self).__init__()

        # item embedding layer
        self.item_history_feature = item_history_feature
        self.item_emb = Embedding(item_history_feature.vocab_size, item_history_feature.embed_dim, padding_idx=0)

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

    def forward(self, input_dict):
        # Eq. 1-4, index item embeddings and pass through gru
        # # Fetch the embeddings for items in the session
        input = input_dict[self.item_history_feature.name]
        value_mask = (input != 0)
        value_counts = value_mask.sum(dim=1, keepdim=False).to("cpu").detach()
        embs = rnn_utils.pack_padded_sequence(self.emb_dropout(self.item_emb(input)), value_counts, batch_first=True, enforce_sorted=False)

        # # compute hidden states at each time step
        h, h_t = self.gru(embs)
        h_t = h_t.permute(1, 0, 2)
        h, _ = rnn_utils.pad_packed_sequence(h, batch_first=True)

        # Eq. 5, set last hidden state of gru as the output of the global
        # encoder
        c_g = h_t.squeeze(1)

        # Eq. 8, compute similarity between final hidden state and previous
        # hidden states
        q = sigmoid(h_t @ self.a_1.T + h @ self.a_2.T) @ self.v

        # Eq. 7, compute attention
        alpha = torch.exp(q) * value_mask.unsqueeze(-1)
        alpha /= alpha.sum(dim=1, keepdim=True)

        # Eq. 6, compute the output of the local encoder
        c_l = (alpha * h).sum(1)

        # Eq. 9, compute session representation by concatenating user
        # sequential behavior (global) and main purpose in the current session
        # (local)
        c = self.session_rep_dropout(torch.hstack((c_g, c_l)))

        # Eq. 10, compute bilinear similarity between current session and each
        # candidate items
        s = c @ self.b.T @ self.item_emb.weight.T

        return s
