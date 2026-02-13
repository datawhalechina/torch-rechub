"""
Date: create on 2022/5/8, update on 2022/5/8
References:
    paper: (ICDM'2018) Self-attentive sequential recommendation
    url: https://arxiv.org/pdf/1808.09781.pdf
    code: https://github.com/kang205/SASRec
Authors: Yuchen Wang, 615922749@qq.com
"""
import numpy as np
import torch
import torch.nn as nn

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.basic.layers import MLP, EmbeddingLayer


class SASRec(torch.nn.Module):
    """SASRec: Self-Attentive Sequential Recommendation
    Args:
        features (list): the list of `Feature Class`. In sasrec, the features list needs to have three elements in order: user historical behavior sequence features, positive sample sequence, and negative sample sequence.
        max_len: The length of the sequence feature.
        num_blocks: The number of stacks of attention modules.
        num_heads: The number of heads in MultiheadAttention.
        item_feature: Optional item feature for in-batch negative sampling mode.

    """

    def __init__(
        self,
        features,
        max_len=50,
        dropout_rate=0.5,
        num_blocks=2,
        num_heads=1,
        item_feature=None,
    ):
        super(SASRec, self).__init__()

        self.features = features
        self.item_feature = item_feature  # Optional: for in-batch negative sampling
        self.mode = None  # For inference: "user" or "item"
        self.max_len = max_len

        self.features = features

        self.item_num = self.features[0].vocab_size
        self.embed_dim = self.features[0].embed_dim

        self.item_emb = EmbeddingLayer(self.features)
        self.position_emb = torch.nn.Embedding(max_len, self.embed_dim)
        self.emb_dropout = torch.nn.Dropout(p=dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()
        self.last_layernorm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)

        for _ in range(num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(self.embed_dim, num_heads, dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.embed_dim, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.embed_dim, dropout_rate)
            self.forward_layers.append(new_fwd_layer)

    def seq_forward(self, x, embed_x_feature):
        x = x['seq']

        embed_x_feature *= self.features[0].embed_dim**0.5
        embed_x_feature = embed_x_feature.squeeze()  # (bacth_size, max_len, embed_dim)

        positions = np.tile(np.array(range(x.shape[1])), [x.shape[0], 1])

        embed_x_feature += self.position_emb(torch.LongTensor(positions))
        embed_x_feature = self.emb_dropout(embed_x_feature)

        timeline_mask = torch.BoolTensor(x == 0)
        embed_x_feature *= ~timeline_mask.unsqueeze(-1)

        attention_mask = ~torch.tril(torch.ones((embed_x_feature.shape[1], embed_x_feature.shape[1]), dtype=torch.bool))

        for i in range(len(self.attention_layers)):
            embed_x_feature = torch.transpose(embed_x_feature, 0, 1)
            Q = self.attention_layernorms[i](embed_x_feature)
            mha_outputs, _ = self.attention_layers[i](Q, embed_x_feature, embed_x_feature, attn_mask=attention_mask)

            embed_x_feature = Q + mha_outputs
            embed_x_feature = torch.transpose(embed_x_feature, 0, 1)

            embed_x_feature = self.forward_layernorms[i](embed_x_feature)
            embed_x_feature = self.forward_layers[i](embed_x_feature)
            embed_x_feature *= ~timeline_mask.unsqueeze(-1)

        seq_output = self.last_layernorm(embed_x_feature)

        return seq_output

    def user_tower(self, x):
        """Compute user embedding for in-batch negative sampling.
        Takes the last valid position's output as user representation.
        """
        if self.mode == "item":
            return None
        # Get sequence embedding
        seq_embed = self.item_emb(x, self.features[:1])[:, 0]  # Only use seq feature
        seq_output = self.seq_forward(x, seq_embed)  # [batch_size, max_len, embed_dim]

        # Get the last valid position for each sequence
        seq = x['seq']
        seq_lens = (seq != 0).sum(dim=1) - 1  # Last valid index
        seq_lens = seq_lens.clamp(min=0)
        batch_idx = torch.arange(seq_output.size(0), device=seq_output.device)
        user_emb = seq_output[batch_idx, seq_lens]  # [batch_size, embed_dim]

        if self.mode == "user":
            return user_emb
        return user_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]

    def item_tower(self, x):
        """Compute item embedding for in-batch negative sampling."""
        if self.mode == "user":
            return None
        if self.item_feature is not None:
            item_ids = x[self.item_feature.name]
            # Use the embedding layer to get item embeddings
            item_emb = self.item_emb.embedding[self.features[0].name](item_ids)
            if self.mode == "item":
                return item_emb
            return item_emb.unsqueeze(1)  # [batch_size, 1, embed_dim]
        return None

    def forward(self, x):
        # Support inference mode
        if self.mode == "user":
            return self.user_tower(x)
        if self.mode == "item":
            return self.item_tower(x)

        # In-batch negative sampling mode
        if self.item_feature is not None:
            user_emb = self.user_tower(x)  # [batch_size, 1, embed_dim]
            item_emb = self.item_tower(x)  # [batch_size, 1, embed_dim]
            return torch.mul(user_emb, item_emb).sum(dim=-1).squeeze()

        # Original behavior: pairwise loss with pos/neg sequences
        embedding = self.item_emb(x, self.features)
        seq_embed, pos_embed, neg_embed = embedding[:, 0], embedding[:, 1], embedding[:, 2]
        seq_output = self.seq_forward(x, seq_embed)

        pos_logits = (seq_output * pos_embed).sum(dim=-1)
        neg_logits = (seq_output * neg_embed).sum(dim=-1)

        return pos_logits, neg_logits


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2)
        outputs += inputs
        return outputs


if __name__ == '__main__':
    seq = SequenceFeature('seq', vocab_size=17, embed_dim=7, pooling='concat')
    pos = SequenceFeature('pos', vocab_size=17, embed_dim=7, pooling='concat', shared_with='seq')
    neg = SequenceFeature('neg', vocab_size=17, embed_dim=7, pooling='concat', shared_with='seq')

    seq = [seq, pos, neg]

    hist_seq = torch.tensor([[1, 2, 3, 4], [2, 3, 7, 8]])
    pos_seq = hist_seq
    neg_seq = hist_seq

    x = {'seq': hist_seq, 'pos': pos_seq, 'neg': neg_seq}
    model = SASRec(features=seq)
    print('out', model(x))
