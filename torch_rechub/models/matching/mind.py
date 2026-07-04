"""
Date: create on 08/06/2022
References:
    paper: Multi-Interest Network with Dynamic Routing
    url: https://arxiv.org/pdf/1904.08030v1
    code: https://github.com/ShiningCosmos/pytorch_ComiRec/blob/main/MIND.py
Authors: Kai Wang, 306178200@qq.com
"""

import torch
import torch.nn.functional as F
from torch import nn

from ...basic.layers import MLP, CapsuleNetwork, EmbeddingLayer, MultiInterestSA


def dynamic_interest_mask(mask, interest_num):
    """Per-user active-interest mask following the MIND paper's heuristic.

    Sizes the number of interests by history length
    ``K'_u = max(1, min(interest_num, floor(log2(|I_u|))))`` where ``|I_u|`` is
    each user's interacted-item count (``mask.sum(-1)``). This is MIND-specific
    and lives with the model, not in the shared ``CapsuleNetwork`` routing layer.

    Args:
        mask (Tensor): history mask ``(B, L)`` with 1 for real items, 0 for padding.
        interest_num (int): upper bound ``K`` on the number of interests.

    Returns:
        Tensor: boolean ``(B, interest_num)`` mask, ``True`` for the first
        ``K'_u`` (active) interests of each user.
    """
    hist_len = mask.sum(dim=1).float().clamp(min=1.0)  # |I_u|, >=1 to keep log2 finite
    k_u = torch.floor(torch.log2(hist_len)).clamp(min=1, max=interest_num)  # (B,)
    idx = torch.arange(interest_num, device=mask.device).view(1, -1)  # (1, K)
    return idx < k_u.unsqueeze(1)  # (B, K)


class MIND(torch.nn.Module):
    """The match model mentioned in `Multi-Interest Network with Dynamic Routing` paper.
    It's a ComirecDR match model trained by global softmax loss on list-wise samples.
    Note in origin paper, it's without item dnn tower and train item embedding directly.

    Args:
        user_features (list[Feature Class]): training by the user tower module.
        history_features (list[Feature Class]): training history
        item_features (list[Feature Class]): training by the embedding table, it's the item id feature.
        neg_item_feature (list[Feature Class]): training by the embedding table, it's the negative items id feature.
        max_length (int): max sequence length of input item sequence
        temperature (float): temperature factor for similarity score, default to 1.0.
        interest_num （int): interest num (the upper bound ``K`` when ``dynamic_interest``).
        dynamic_interest (bool): if ``True``, size the active interests per user by the
            paper's heuristic ``K'_u = max(1, min(interest_num, floor(log2(|I_u|))))``
            instead of a fixed ``interest_num``. Defaults to ``False`` (unchanged behaviour).

    Note:
        With ``dynamic_interest=True`` the user-tower inference embedding zeroes out
        each user's inactive interests (rows ``>= K'_u`` are zero vectors). Downstream
        retrieval/indexing should skip those zero rows; :meth:`active_interest_mask`
        returns the boolean ``(B, interest_num)`` mask of the interests to keep.
    """

    def __init__(self, user_features, history_features, item_features, neg_item_feature, max_length, temperature=1.0, interest_num=4, dynamic_interest=False):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.history_features = history_features
        self.neg_item_feature = neg_item_feature
        self.temperature = temperature
        self.interest_num = interest_num
        self.dynamic_interest = dynamic_interest
        self.max_length = max_length
        self.user_dims = sum([fea.embed_dim for fea in user_features + history_features])

        self.embedding = EmbeddingLayer(user_features + item_features + history_features)
        self.capsule = CapsuleNetwork(self.history_features[0].embed_dim, self.max_length, bilinear_type=0, interest_num=self.interest_num)
        self.convert_user_weight = nn.Parameter(torch.rand(self.user_dims, self.history_features[0].embed_dim), requires_grad=True)
        self.mode = None

    def forward(self, x):
        # Compute the per-user active-interest mask once and reuse it for routing
        # (via the user tower), label-aware selection, and inference.
        interest_mask = self.active_interest_mask(x)
        user_embedding = self.user_tower(x, interest_mask=interest_mask)
        item_embedding = self.item_tower(x)
        if self.mode == "user":
            return user_embedding
        if self.mode == "item":
            return item_embedding

        pos_item_embedding = item_embedding[:, 0, :]
        dot_res = torch.bmm(user_embedding, pos_item_embedding.squeeze(1).unsqueeze(-1))
        if interest_mask is not None:
            # Never route the label-aware attention to a surplus (inactive) interest.
            dot_res = dot_res.masked_fill(~interest_mask.unsqueeze(-1), float("-inf"))
        k_index = torch.argmax(dot_res, dim=1).squeeze(-1)
        batch_index = torch.arange(user_embedding.shape[0], device=user_embedding.device)
        best_interest_emb = user_embedding[batch_index, k_index, :].unsqueeze(1)

        y = torch.mul(best_interest_emb, item_embedding).sum(dim=-1)
        return y

    def user_tower(self, x, interest_mask=None):
        if self.mode == "item":
            return None
        input_user = self.embedding(x, self.user_features, squeeze_dim=True).unsqueeze(1)  # [batch_size, num_features*deep_dims]
        input_user = input_user.expand([input_user.shape[0], self.interest_num, input_user.shape[-1]])

        history_emb = self.embedding(x, self.history_features).squeeze(1)
        mask = self.gen_mask(x)
        # Compute the mask here when called standalone (e.g. inference); forward()
        # passes it in so it is computed only once per step.
        if self.dynamic_interest and interest_mask is None:
            interest_mask = self.active_interest_mask(x)
        multi_interest_emb = self.capsule(history_emb, mask, interest_mask=interest_mask)

        input_user = torch.cat([input_user, multi_interest_emb], dim=-1)

        # user_embedding = self.user_mlp(input_user).unsqueeze(1)
        # #[batch_size, interest_num, embed_dim]
        user_embedding = torch.matmul(input_user, self.convert_user_weight)
        user_embedding = F.normalize(user_embedding, p=2, dim=-1)  # L2 normalize
        if interest_mask is not None:
            # Zero the surplus interests so downstream selection/retrieval sees only K'_u.
            user_embedding = user_embedding * interest_mask.unsqueeze(-1)
        if self.mode == "user":
            # inference embedding mode -> [batch_size, interest_num, embed_dim]
            return user_embedding
        return user_embedding

    def active_interest_mask(self, x):
        """Per-user boolean mask ``(B, interest_num)`` of active interests.

        Returns ``None`` when ``dynamic_interest`` is off (all interests active).
        With ``dynamic_interest=True`` the inference user embedding zeroes the
        inactive interests; downstream retrieval can use this mask to drop them.
        """
        if not self.dynamic_interest:
            return None
        return dynamic_interest_mask(self.gen_mask(x), self.interest_num)

    def item_tower(self, x):
        if self.mode == "user":
            return None
        pos_embedding = self.embedding(x, self.item_features, squeeze_dim=False)  # [batch_size, 1, embed_dim]
        pos_embedding = F.normalize(pos_embedding, p=2, dim=-1)  # L2 normalize
        if self.mode == "item":  # inference embedding mode
            return pos_embedding.squeeze(1)  # [batch_size, embed_dim]
        neg_embeddings = self.embedding(x, self.neg_item_feature, squeeze_dim=False).squeeze(1)  # [batch_size, n_neg_items, embed_dim]
        neg_embeddings = F.normalize(neg_embeddings, p=2, dim=-1)  # L2 normalize
        # [batch_size, 1+n_neg_items, embed_dim]
        return torch.cat((pos_embedding, neg_embeddings), dim=1)

    def gen_mask(self, x):
        his_list = x[self.history_features[0].name]
        mask = (his_list > 0).long()
        return mask
