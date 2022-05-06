import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from .activation import activation_layer
from .features import DenseFeature, SparseFeature, SequenceFeature


class PredictionLayer(nn.Module):
    """
      Arguments
         - **task**: str, ``"classification"`` for  binary logloss or  ``"regression"`` for regression loss
    """

    def __init__(self, task_type='classification'):
        super(PredictionLayer, self).__init__()
        if task_type not in ["classification", "regression"]:
            raise ValueError("task_type must be classification or regression")
        self.task_type = task_type

    def forward(self, x):
        if self.task_type == "classification":
            x = torch.sigmoid(x)
        return x


class EmbeddingLayer(nn.Module):
    """General Embedding Layer
    Args:
        field_dims: list, the number of tokens in each token fields
        offsets: list, the dimension offset of each token field
        embed_dim: int, the dimension of output embedding vectors

    Input:
        input_x (dict): {feature_name: A 3D tensor with shape:``(batch_size,field_size)``}.

    Returns:
        output: tensor,  A 3D tensor with shape: ``(batch_size,field_size,embed_dim)``.

    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  #exist
                continue
            if isinstance(fea, SparseFeature):
                self.embed_dict[fea.name] = nn.Embedding(fea.vocab_size, fea.embed_dim)
            elif isinstance(fea, SequenceFeature) and fea.shared_with == None:
                self.embed_dict[fea.name] = nn.Embedding(fea.vocab_size, fea.embed_dim)
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

        for matrix in self.embed_dict.values():  #init embedding weight
            torch.nn.init.xavier_normal_(matrix.weight)

    def forward(self, x, features, squeeze_dim=False):
        """_summary_

        Args:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:``(batch_size, seq_len)``, sparse/dense value is a 1D tensor with shape ``(batch_size)``
            
        Returns:
            dense_values (tensor): A 2D tensor with shape: ``(batch_size, field_size)``
            sparse_emb (tensor): A 3D tensor with shape: ``(batch_size, field_size, embed_dim)``
        """
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = MaskedSumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = MaskedAveragePooling()
                elif fea.pooling == "concat":
                    #(batch_size, num_seq_fields, seq_length, emb_dim)
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." % (["sum", "mean"], fea.pooling))
                if fea.shared_with == None:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.name](x[fea.name].long())).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long())).unsqueeze(1))  #shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float().unsqueeze(1))  #.unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)  # torch.stack(list(embedding_dict.values()), dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            sparse_emb = torch.cat(sparse_emb, dim=1)  #[batch_size, num_features, embed_dim]

        if squeeze_dim:  #Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  #only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                return sparse_emb.flatten(start_dim=1)  #squeeze dim to : [batch_size, num_features*embed_dim]
            elif dense_exists and sparse_exists:
                #concat dense value with sparse embedding,
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values), dim=1)
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  #[batch_size, num_features, embed_dim]
            else:
                raise ValueError("If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" % ("SparseFeatures", features))


class LR(nn.Module):
    """
        Logistic Regression
    """

    def __init__(self, num_fields, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(num_fields, 1, bias=True)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, 1)``
        """
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class FeaturesEmbedding(nn.Module):
    """
        所有sparse特征 统一emb维度=embed_dim
    """

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding_list = nn.ModuleList([nn.Embedding(field_dims[i], embed_dim) for i in range(len(field_dims))])
        for emb in self.embedding_list:
            torch.nn.init.xavier_uniform_(emb.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        :return: ``(batch_size, num_fields, embed_dim)`` 
        """
        sparse_emb = [emb(x[:, i].unsqueeze(1)) for i, emb in enumerate(self.embedding_list)]
        return torch.cat(sparse_emb, dim=1)


class ConcatPooling(nn.Module):
    """
        keep the data shape
    """

    def __init__(self):
        super(ConcatPooling, self).__init__()

    def forward(self, x):
        """
        :param x: matrix tensor of size ``(batch_size, num_seq_fields, seq_length, emb_dim)``
        :return ``(batch_size, num_seq_fields, seq_length, emb_dim)``
        """
        return x
        #return x.view(batch_size, -1, emb_dim)  # batch_size, 1, embedding_dim


class MaskedAveragePooling(nn.Module):
    r"""Calculate FM result over the embeddings

    Args:
        reduce_sum: bool, whether to sum the result, default is True.

    Input:
        input_x: tensor, A 3D tensor with shape:``(batch_size,field_size,embed_dim)``.

    Output
        output: tensor, A 3D tensor with shape: ``(batch_size,1)`` or ``(batch_size, embed_dim)``.
    """

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
        return embedding_vec  # batch_size, 1, embedding_dim


class MaskedSumPooling(nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """

    def __init__(self):
        super(MaskedSumPooling, self).__init__()

    def forward(self, embedding_matrix):
        """
        :param x: matrix tensor of size ``(batch_size, seq_length, embed_dim)``
        :return ``(batch_size, embed_dim)``
        """
        return torch.sum(embedding_matrix, dim=1)


class SequenceFeaturesEmbedding(nn.Module):
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

        self.embedding_list = nn.ModuleList([nn.Embedding(field_dims[i], embed_dim) for i in range(len(field_dims))])
        for emb in self.embedding_list:
            xavier_normal_(emb.weight.data)

    def forward(self, x, x_candidate=None):
        """
        :param x: Long tensor of size ``(batch_size, num_seq_fields, seq_length)``
        :param x_candidate: Long tensor of size ``(batch_size, num_seq_fields)``
                            for DIN model
        :return: ``(batch_size, num_seq_fields, emb_dim)`` 
                 ``(batch_size, num_seq_fields, seq_length, emb_dim) ``
        """
        if self.pool != None:  #(batch_size, num_seq_fields, emb_dim)
            sequence_emb = [self.pool(emb(x[:, i])).unsqueeze(1) for i, emb in enumerate(self.embedding_list)]
        else:  #(batch_size, num_seq_fields, seq_length, emb_dim)
            sequence_emb = [emb(x[:, i].unsqueeze(1)) for i, emb in enumerate(self.embedding_list)]

        sequence_emb = torch.cat(sequence_emb, dim=1)  #concat all sequence fields

        if x_candidate != None:  #(batch_size, num_seq_fields, emb_dim)
            candidate_emb = [emb(x_candidate[:, i].unsqueeze(1)) for i, emb in enumerate(self.embedding_list)]
            candidate_emb = torch.cat(candidate_emb, dim=1)
            return sequence_emb, candidate_emb
        else:
            return sequence_emb


class MLP(nn.Module):
    """Multi Layer Perceptron

    Args:
        nn (_type_): _description_
    """

    def __init__(self, input_dim, dims, dropout=0, activation="relu", output_layer=True):
        super().__init__()
        layers = list()
        for i_dim in dims:
            layers.append(nn.Linear(input_dim, i_dim))
            layers.append(nn.BatchNorm1d(i_dim))
            layers.append(activation_layer(activation))
            layers.append(nn.Dropout(p=dropout))
            input_dim = i_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix