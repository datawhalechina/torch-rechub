import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import ModuleList, Sequential, Linear, Embedding, BatchNorm1d, ReLU, Dropout
from .activation import activation_layer

class FeaturesLinear(torch.nn.Module):
    """
        Logistic Regression
    """
    def __init__(self, num_fields, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = Linear(num_fields, 1, bias=True) 

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, 1)``
        """
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class FeaturesEmbedding(torch.nn.Module):
    """
        所有sparse特征 统一emb维度=embed_dim
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding_list = ModuleList([
            Embedding(field_dims[i], embed_dim) for i in range(len(field_dims))]) 
        for emb in self.embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: ``(batch_size, num_fields, embed_dim)`` 
        """
        sparse_emb = [emb(x[:,i].unsqueeze(1)) for i, emb in enumerate(self.embedding_list)]
        return torch.cat(sparse_emb, dim=1)

class MaskedAveragePooling(torch.nn.Module):
    def __init__(self):
        super(MaskedAveragePooling, self).__init__()

    def forward(self, embedding_matrix):
        """
        :param x: matrix tensor of size ``(batch_size, seq_length, embed_dim)``
        :return ``(batch_size, embed_dim)``
        """
        sum_pooling_matrix = torch.sum(embedding_matrix, dim=1)
        non_padding_length = (embedding_matrix != 0).sum(dim=1)
        embedding_vec = sum_pooling_matrix / (non_padding_length.float() + 1e-16)
        return embedding_vec # batch_size, 1, embedding_dim

class MaskedSumPooling(torch.nn.Module):
    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        """
        :param x: matrix tensor of size ``(batch_size, seq_length, embed_dim)``
        :return ``(batch_size, embed_dim)``
        """
        return torch.sum(embedding_matrix, dim=1)

class SequenceFeaturesEmbedding(torch.nn.Module):
    """
        sequence embedding
        TODO：specific sparse fields shared embedding with specific sequence embedding 
    """

    def __init__(self, field_dims, embed_dim, mode="mean"):
        super().__init__()
        assert mode in ['concat', 'sum', 'mean'], "mode must in {'concat', 'sum', 'mean'}"
        self.field_dims = field_dims
        self.mode = mode
        if self.mode == "mean":
            self.pool = MaskedAveragePooling()
        elif self.mode == "sum":
            self.pool = MaskedSumPooling()
        elif self.mode == "concat":
            self.pool = None

        self.embedding_list = ModuleList([
            Embedding(field_dims[i], embed_dim) for i in range(len(field_dims))]) 
        for emb in self.embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x, x_candidate=None):
        """
        :param x: Long tensor of size ``(batch_size, num_seq_fields, seq_length)``
        :param x_candidate: Long tensor of size ``(batch_size, num_seq_fields)``
                            for DIN model
        :return: ``(batch_size, num_seq_fields, emb_dim)`` 
                 ``(batch_size, num_seq_fields, seq_length, emb_dim) ``
        """
        if self.pool != None: #(batch_size, num_seq_fields, emb_dim) 
            sequence_emb = [self.pool(emb(x[:,i])).unsqueeze(1) for i, emb in enumerate(self.embedding_list)]
        else: #(batch_size, num_seq_fields, seq_length, emb_dim) 
            sequence_emb = [emb(x[:,i].unsqueeze(1)) for i, emb in enumerate(self.embedding_list)]
        
        sequence_emb = torch.cat(sequence_emb, dim=1) #concat all sequence fields
        
        if x_candidate != None: #(batch_size, num_seq_fields, emb_dim) 
            candidate_emb = [emb(x_candidate[:,i].unsqueeze(1)) for i, emb in enumerate(self.embedding_list)]
            candidate_emb = torch.cat(candidate_emb, dim=1)
            return sequence_emb, candidate_emb
        else:
            return sequence_emb

class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, activation="relu", output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(Linear(input_dim, embed_dim))
            layers.append(BatchNorm1d(embed_dim))
            layers.append(activation_layer(activation))
            layers.append(Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(Linear(input_dim, 1))
        self.mlp = Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)

class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix