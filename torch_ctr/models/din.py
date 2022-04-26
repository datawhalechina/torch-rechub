"""
Created on April 23, 2022
Reference: "Deep Interest Network for Click-Through Rate Prediction", KDD, 2018
@author: Mincai Lai, laimincai@shanghaitech.edu.cn
"""
import torch

from torch_ctr.layers import FeaturesEmbedding, SequenceFeaturesEmbedding, MultiLayerPerceptron
from torch.nn import ModuleList, Sequential, Linear,  PReLU
from ..activation import Dice

class DIN(torch.nn.Module):
    def __init__(self, sequence_field_dims, sparse_field_dims, embed_dim, mlp_dims=None, attention_mlp_dims=[36], activation='dice', dropout=None):
        super(DIN, self).__init__()

        self.num_seq_fields = len(sequence_field_dims)
        self.embedding_sequence = SequenceFeaturesEmbedding(sequence_field_dims, embed_dim, mode="concat")
        self.embedding_sparse = FeaturesEmbedding(sparse_field_dims, embed_dim)
        self.embed_output_dim = (len(sparse_field_dims) + 2*self.num_seq_fields)*embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, activation="dice")
        self.attention_layers = ModuleList([
            ActivationUnit(embed_dim, attention_mlp_dims, activation, use_softmax=False) for i in range(self.num_seq_fields)]) 

    def forward(self, x):
        """
        The original paper without dense feature.
        :param x_sequence: Long tensor of size ``(batch_size, num_seq_fields, seq_length)``
                        # each fields is a tensor list
        :param x_candidate: Long tensor of size ``(batch_size, num_seq_fields)``
                        # each fields is a candidate field id for x_sequence
        :param x_sparse: Long tensor of size ``(batch_size, num_sparse_fields)``
                        # sparse: user_profile_feature, candidate_feature, context_feature
        """
        x_sequence, x_candidate, x_sparse = x["x_sequence"], x["x_candidate"], x["x_sparse"]

        #embed_x_sequence: (batch_size, num_seq_fields, seq_length, emb_dim) 
        #embed_x_candidate:  (batch_size, num_seq_fields, emb_dim) 
        embed_x_sequence, embed_x_candidate = self.embedding_sequence(x_sequence, x_candidate)
        embed_x_sparse = self.embedding_sparse(x_sparse) #(batch_size, num_sparse_fields, embed_dim)
        pooling_concat = []
        for i in range(self.num_seq_fields):
            #为每一个序列特征都训练一个独立的激活单元 eg.历史序列item id，历史序列item id品牌
            pooling_seq = self.attention_layers[i](embed_x_sequence[:,i,:,:], embed_x_candidate[:,i,:])
            pooling_concat.append(pooling_seq.unsqueeze(1)) #(batch_size, 1, emb_dim)
        pooling_concat = torch.cat(pooling_concat, dim=1)  #(batch_size, num_seq_fields, emb_dim)
        
        mlp_in = torch.cat([pooling_concat.flatten(start_dim=1), embed_x_candidate.flatten(start_dim=1), embed_x_sparse.flatten(start_dim=1)], dim=1) #(batch_size, N)

        y = self.mlp(mlp_in)
        return torch.sigmoid(y.squeeze(1))


class ActivationUnit(torch.nn.Module):
    """
        DIN Attention Layer
    """
    def __init__(self, emb_dim, attention_mlp_dims=[36], activation="dice", use_softmax=False):
        super(ActivationUnit, self).__init__()
        self.emb_dim = emb_dim
        self.use_softmax = use_softmax 
        self.attention = MultiLayerPerceptron(4 * self.emb_dim, attention_mlp_dims, activation=activation)

    def forward(self, history, candidate):
        """
        :param history: Long tensor of size ``(batch_size, seq_length, emb_dim) ``
        :param candidate: Long tensor of size ``(batch_size, emb_dim)``
        """
        seq_length = history.size(1)
        candidate = candidate.unsqueeze(1).expand(-1, seq_length, -1) #batch_size,seq_length,emb_dim 
        att_input = torch.cat([candidate, history, candidate - history, candidate * history], dim=-1) # batch_size,seq_length,4*emb_dim 
        att_weight = self.attention(att_input.view(-1, 4*self.emb_dim))  #  #(batch_size*seq_length,4*emb_dim)
        att_weight = att_weight.view(-1, seq_length)    #(batch_size*seq_length, 1) -> (batch_size,seq_length)
        if self.use_softmax:
            att_weight = att_weight.softmax(dim=-1) 
        # (batch_size, seq_length, 1) * (batch_size, seq_length, emb_dim)
        output = (att_weight.unsqueeze(-1) * history).sum(dim=1) #(batch_size,emb_dim)
        return output