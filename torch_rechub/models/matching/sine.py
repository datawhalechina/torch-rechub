"""
Date: created on 03/07/2022
References:
    paper: Sparse-Interest Network for Sequential Recommendation
    url: https://arxiv.org/abs/2102.09267
    code: https://github.com/Qiaoyut/SINE/blob/master/model.py
Authors: Bo Kang, klinux@live.com
"""

import torch
import torch.nn.functional as F
from torch import einsum


class SINE(torch.nn.Module):
    """The match model was proposed in `Sparse-Interest Network for Sequential Recommendation` paper.

    Args:
        history_features (list[str]): training history feature names, this is for indexing the historical sequences from input dictionary
        item_features (list[str]): item feature names, this is for indexing the items from input dictionary
        neg_item_features (list[str]): neg item feature names, this for indexing negative items from input dictionary
        num_items (int): number of items in the data
        embedding_dim (int): dimensionality of the embeddings
        hidden_dim (int): dimensionality of the hidden layer in self attention modules
        num_concept (int): number of concept, also called conceptual prototypes
        num_intention (int): number of (user) specific intentions out of the concepts
        seq_max_len (int): max sequence length of input item sequence
        num_heads (int): number of attention heads in self attention modules, default to 1
        temperature (float): temperature factor in the similarity measure, default to 1.0
    """

    def __init__(self, history_features, item_features, neg_item_features, num_items, embedding_dim, hidden_dim, num_concept, num_intention, seq_max_len, num_heads=1, temperature=1.0):
        super().__init__()
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_features = neg_item_features
        self.temperature = temperature
        self.num_concept = num_concept
        self.num_intention = num_intention
        self.seq_max_len = seq_max_len

        std = 1e-4
        self.item_embedding = torch.nn.Embedding(num_items, embedding_dim)
        torch.nn.init.normal_(self.item_embedding.weight, 0, std)
        self.concept_embedding = torch.nn.Embedding(num_concept, embedding_dim)
        torch.nn.init.normal_(self.concept_embedding.weight, 0, std)
        self.position_embedding = torch.nn.Embedding(seq_max_len, embedding_dim)
        torch.nn.init.normal_(self.position_embedding.weight, 0, std)

        self.w_1 = torch.nn.Parameter(torch.rand(embedding_dim, hidden_dim), requires_grad=True)
        self.w_2 = torch.nn.Parameter(torch.rand(hidden_dim, num_heads), requires_grad=True)

        self.w_3 = torch.nn.Parameter(torch.rand(embedding_dim, embedding_dim), requires_grad=True)

        self.w_k1 = torch.nn.Parameter(torch.rand(embedding_dim, hidden_dim), requires_grad=True)
        self.w_k2 = torch.nn.Parameter(torch.rand(hidden_dim, num_intention), requires_grad=True)

        self.w_4 = torch.nn.Parameter(torch.rand(embedding_dim, hidden_dim), requires_grad=True)
        self.w_5 = torch.nn.Parameter(torch.rand(hidden_dim, num_heads), requires_grad=True)

        self.mode = None

    def forward(self, x):
        user_embedding = self.user_tower(x)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        y = torch.mul(user_embedding, item_embedding).sum(dim=-1)

        # # compute covariance regularizer
        # M = torch.cov(self.concept_embedding.weight, correction=0)
        # l_c = (torch.norm(M, p='fro')**2 - torch.norm(torch.diag(M), p='fro')**2)/2

        return y

    def user_tower(self, x):
        if self.mode == "item":
            return None


# sparse interests extraction
# # user specific historical item embedding X^u
        hist_item = x[self.history_features[0]]
        x_u = self.item_embedding(hist_item) + \
            self.position_embedding.weight.unsqueeze(0)
        x_u_mask = (x[self.history_features[0]] > 0).long()

        # # user specific conceptual prototypes C^u
        # ## attention a
        h_1 = einsum('bse, ed -> bsd', x_u, self.w_1).tanh()
        a_hist = F.softmax(einsum('bsd, dh -> bsh', h_1, self.w_2) + -1.e9 * (1 - x_u_mask.unsqueeze(-1).float()), dim=1)

        # ## virtual concept vector z_u
        z_u = einsum("bse, bsh -> be", x_u, a_hist)

        # ## similarity between user's concept vector and entire conceptual prototypes s^u
        s_u = einsum("be, te -> bt", z_u, self.concept_embedding.weight)
        s_u_top_k = torch.topk(s_u, self.num_intention)

        # ## final C^u
        c_u = einsum("bk, bke -> bke", torch.sigmoid(s_u_top_k.values), self.concept_embedding(s_u_top_k.indices))

        # # user intention assignment P_{k|t}
        p_u = F.softmax(einsum("bse, bke -> bks", F.normalize(x_u @ self.w_3, dim=-1), F.normalize(c_u, p=2, dim=-1)), dim=1)

        # # attention weighing P_{t|k}
        h_2 = einsum('bse, ed -> bsd', x_u, self.w_k1).tanh()
        a_concept_k = F.softmax(einsum('bsd, dk -> bsk', h_2, self.w_k2) + -1.e9 * (1 - x_u_mask.unsqueeze(-1).float()), dim=1)

        # # multiple interests encoding \phi_\theta^k(x^u)
        phi_u = einsum("bks, bse -> bke", p_u * a_concept_k.permute(0, 2, 1), x_u)

        # adaptive interest aggregation
        # # intention aware input behavior \hat{X^u}
        x_u_hat = einsum('bks, bke -> bse', p_u, c_u)

        # # user's next intention C^u_{apt}
        h_3 = einsum('bse, ed -> bsd', x_u_hat, self.w_4).tanh()
        c_u_apt = F.normalize(einsum("bs, bse -> be", F.softmax(einsum('bsd, dh -> bsh', h_3, self.w_5).reshape(-1, self.seq_max_len) + -1.e9 * (1 - x_u_mask.float()), dim=1), x_u_hat), -1)

        # # aggregation weights e_k^u
        e_u = F.softmax(einsum('be, bke -> bk', c_u_apt, phi_u) / self.temperature, dim=1)

        # final user representation v^u
        v_u = einsum('bk, bke -> be', e_u, phi_u)

        if self.mode == "user":
            return v_u
        return v_u.unsqueeze(1)

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.item_embedding(x[self.item_features[0]]).unsqueeze(1)
        if self.mode == "item":  # inference embedding mode
            return pos_embedding.squeeze(1)  # [batch_size, embed_dim]
        neg_embeddings = self.item_embedding(x[self.neg_item_features[0]]).squeeze(1)  # [batch_size, n_neg_items, embed_dim]

        # [batch_size, 1+n_neg_items, embed_dim]
        return torch.cat((pos_embedding, neg_embeddings), dim=1)

    def gen_mask(self, x):
        his_list = x[self.history_features[0].name]
        mask = (his_list > 0).long()
        return mask
