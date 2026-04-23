"""
Date: create on 01/05/2024
References:
    paper: (AAAI'2019) Deep Interest Evolution Network for Click-Through Rate Prediction
    url: https://arxiv.org/pdf/1809.03672
Authors: Tao Fan, thisisevy@foxmail.com
"""

import torch
from torch import nn
from torch.nn import Parameter, init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ...basic.layers import MLP, EmbeddingLayer


class AUGRU_Cell(nn.Module):

    def __init__(self, embed_dim):
        super(AUGRU_Cell, self).__init__()
        self.Wu = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.Uu = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.bu = init.xavier_uniform_(Parameter(torch.empty(1, embed_dim)))
        self.Wr = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.Ur = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.br = init.xavier_uniform_(Parameter(torch.empty(1, embed_dim)))
        self.Wh = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.Uh = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))
        self.bh = init.xavier_uniform_(Parameter(torch.empty(1, embed_dim)))

    def forward(self, x, h_1, a):
        # a: [ batch_size, 1 ] — pre-computed attention score for this step
        u = torch.sigmoid(torch.matmul(x, self.Wu) + torch.matmul(h_1, self.Uu) + self.bu)
        r = torch.sigmoid(torch.matmul(x, self.Wr) + torch.matmul(h_1, self.Ur) + self.br)
        h_hat = torch.tanh(torch.matmul(x, self.Wh) + r * torch.matmul(h_1, self.Uh) + self.bh)
        u_hat = a * u  # attentional update gate (paper Eq.16)
        return (1 - u_hat) * h_1 + u_hat * h_hat


class AUGRU(nn.Module):

    def __init__(self, embed_dim):
        super(AUGRU, self).__init__()
        self.embed_dim = embed_dim
        self.augru_cell = AUGRU_Cell(embed_dim)
        self.Wa = init.xavier_uniform_(Parameter(torch.empty(embed_dim, embed_dim)))

    def forward(self, x, item, mask=None):
        # x: [B, T, D], item: [B, D], mask: [B, T] bool (True=valid)
        scores = torch.matmul(x, self.Wa)  # [B, T, D]
        scores = (scores * item.unsqueeze(1)).sum(-1)  # [B, T]
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=1)
        # replace nan rows (all-padding) with uniform attention
        nan_rows = attn.isnan().any(dim=1)
        if nan_rows.any():
            attn[nan_rows] = 1.0 / attn.size(1)
        attn = attn.unsqueeze(-1)  # [B, T, 1]

        h = torch.zeros(x.size(0), self.embed_dim, device=x.device)
        outs = []
        for i in range(x.size(1)):
            h = self.augru_cell(x[:, i], h, attn[:, i])
            outs.append(h.unsqueeze(1))
        return torch.cat(outs, dim=1), h


class DIEN(nn.Module):
    """Deep Interest Evolution Network (AAAI 2019).

    Args:
        features (list): user profile / context features fed into the top MLP.
        history_features (list): positive behaviour sequence features (SequenceFeature, pooling="concat").
            Must set padding_idx=0 and shared_with=<target_feature_name> so the embedding table
            is owned by the corresponding target feature.
        neg_history_features (list): negative-sampled behaviour sequence features, one per history feature.
            Must set padding_idx=0 and shared_with=<target_feature_name> (same root as history_features).
            EmbeddingLayer only registers features with shared_with=None as root keys in embed_dict,
            so shared_with must point to the target feature, NOT the history feature.
        target_features (list): target item features used by the AUGRU attention.
            Must set padding_idx=0 so the shared embedding table's row 0 is a true zero vector.
        mlp_params (dict): params for the top MLP, e.g. {"dims": [256, 128]}.
            activation is fixed to "dice" as in the paper.
        alpha (float): weight of the auxiliary loss. Default 0.2.

    Returns (forward):
        tuple(prediction: Tensor[B], aux_loss: Tensor[])

    Notes:
        - Sequences are padded with 0 (convention from generate_seq_feature).
        - Samples with all-padding history keep zero hidden state throughout GRU and AUGRU.
        - Auxiliary loss uses next-step positive/negative supervision (paper Eq.7), skipping padding positions.
        - AUGRU attention is softmax-normalised over the full valid sequence (paper Eq.14-15).
    """

    def __init__(self, features, history_features, neg_history_features, target_features, mlp_params, alpha=0.2):
        super().__init__()
        self.alpha = alpha
        self.features = features
        self.history_features = history_features
        self.neg_history_features = neg_history_features
        self.target_features = target_features
        self.all_dims = sum([fea.embed_dim for fea in features + history_features + target_features])
        self.embedding = EmbeddingLayer(features + history_features + neg_history_features + target_features)
        self.interest_extractor_layers = nn.ModuleList([nn.GRU(fea.embed_dim, fea.embed_dim, batch_first=True) for fea in history_features])
        self.interest_evolving_layers = nn.ModuleList([AUGRU(fea.embed_dim) for fea in history_features])
        self.mlp = MLP(self.all_dims, activation="dice", **mlp_params)
        self.BCELoss = nn.BCELoss()

    def auxiliary(self, outs, pos_emb, neg_emb, mask=None):
        # outs, pos_emb, neg_emb: [B, T, D]
        # mask: [B, T] bool, True=valid; shift by 1: use h[t] to predict e[t+1]
        h = outs[:, :-1]
        pos = pos_emb[:, 1:]
        neg = neg_emb[:, 1:]
        if mask is not None:
            valid = mask[:, :-1] & mask[:, 1:]  # [B, T-1]
        else:
            valid = torch.ones(h.size(0), h.size(1), dtype=torch.bool, device=h.device)
        h, pos, neg = h[valid], pos[valid], neg[valid]
        if h.size(0) == 0:
            return torch.tensor(0.0, device=outs.device)
        pos_score = torch.sigmoid((h * pos).sum(-1, keepdim=True))
        neg_score = torch.sigmoid((h * neg).sum(-1, keepdim=True))
        return (self.BCELoss(pos_score, torch.ones_like(pos_score)) + self.BCELoss(neg_score, torch.zeros_like(neg_score)))

    def forward(self, x):
        embed_x_features = self.embedding(x, self.features)
        embed_x_history = self.embedding(x, self.history_features)
        embed_x_neg_history = self.embedding(x, self.neg_history_features)
        embed_x_target = self.embedding(x, self.target_features)

        interest_extractor = []
        auxi_loss = 0
        for i, fea in enumerate(self.history_features):
            seq = embed_x_history[:, i, :, :]
            mask = self.embedding.input_mask(x, fea).squeeze(1).bool()  # [B, T]
            seq_lens = mask.sum(dim=1).cpu()

            has_hist = seq_lens > 0
            outs = torch.zeros_like(seq)
            if has_hist.any():
                packed = pack_padded_sequence(seq[has_hist], seq_lens[has_hist].clamp(min=1), batch_first=True, enforce_sorted=False)
                packed_out, _ = self.interest_extractor_layers[i](packed)
                unpacked, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=seq.size(1))
                outs[has_hist] = unpacked

            auxi_loss += self.auxiliary(outs, seq, embed_x_neg_history[:, i, :, :], mask)
            interest_extractor.append(outs.unsqueeze(1))

        interest_extractor = torch.cat(interest_extractor, dim=1)

        interest_evolving = []
        for i, fea in enumerate(self.history_features):
            mask = self.embedding.input_mask(x, fea).squeeze(1).bool()  # [B, T]
            has_hist = mask.any(dim=1)
            h = torch.zeros(mask.size(0), self.interest_evolving_layers[i].embed_dim, device=mask.device)
            if has_hist.any():
                _, h_valid = self.interest_evolving_layers[i](interest_extractor[has_hist, i, :, :], embed_x_target[has_hist, i, :], mask[has_hist])
                h[has_hist] = h_valid
            interest_evolving.append(h.unsqueeze(1))

        interest_evolving = torch.cat(interest_evolving, dim=1)
        mlp_in = torch.cat([
            interest_evolving.flatten(start_dim=1),
            embed_x_target.flatten(start_dim=1),
            embed_x_features.flatten(start_dim=1),
        ],
                           dim=1)
        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1)), self.alpha * auxi_loss
