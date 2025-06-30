from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activation import activation_layer
from .features import DenseFeature, SequenceFeature, SparseFeature


class PredictionLayer(nn.Module):
    """Prediction Layer.

    Args:
        task_type (str): if `task_type='classification'`, then return sigmoid(x),
                    change the input logits to probability. if`task_type='regression'`, then return x.
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
    """General Embedding Layer.
    We save all the feature embeddings in embed_dict: `{feature_name : embedding table}`.


    Args:
        features (list): the list of `Feature Class`. It is means all the features which we want to create a embedding table.

    Shape:
        - Input:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list): the list of `Feature Class`. It is means the current features which we want to do embedding lookup.
            squeeze_dim (bool): whether to squeeze dim of output (default = `False`).
        - Output:
            - if input Dense: `(batch_size, num_features_dense)`.
            - if input Sparse: `(batch_size, num_features, embed_dim)` or  `(batch_size, num_features * embed_dim)`.
            - if input Sequence: same with input sparse or `(batch_size, num_features_seq, seq_length, embed_dim)` when `pooling=="concat"`.
            - if input Dense and Sparse/Sequence: `(batch_size, num_features_sparse * embed_dim)`. Note we must squeeze_dim for concat dense value with sparse embedding.
    """

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.embed_dict = nn.ModuleDict()
        self.n_dense = 0

        for fea in features:
            if fea.name in self.embed_dict:  # exist
                continue
            if isinstance(fea, SparseFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, SequenceFeature) and fea.shared_with is None:
                self.embed_dict[fea.name] = fea.get_embedding_layer()
            elif isinstance(fea, DenseFeature):
                self.n_dense += 1

    def forward(self, x, features, squeeze_dim=False):
        sparse_emb, dense_values = [], []
        sparse_exists, dense_exists = False, False
        for fea in features:
            if isinstance(fea, SparseFeature):
                if fea.shared_with is None:
                    sparse_emb.append(self.embed_dict[fea.name](x[fea.name].long()).unsqueeze(1))
                else:
                    sparse_emb.append(self.embed_dict[fea.shared_with](x[fea.name].long()).unsqueeze(1))
            elif isinstance(fea, SequenceFeature):
                if fea.pooling == "sum":
                    pooling_layer = SumPooling()
                elif fea.pooling == "mean":
                    pooling_layer = AveragePooling()
                elif fea.pooling == "concat":
                    pooling_layer = ConcatPooling()
                else:
                    raise ValueError("Sequence pooling method supports only pooling in %s, got %s." % (["sum", "mean"], fea.pooling))
                fea_mask = InputMask()(x, fea)
                if fea.shared_with is None:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.name](x[fea.name].long()), fea_mask).unsqueeze(1))
                else:
                    sparse_emb.append(pooling_layer(self.embed_dict[fea.shared_with](x[fea.name].long()), fea_mask).unsqueeze(1))  # shared specific sparse feature embedding
            else:
                dense_values.append(x[fea.name].float() if x[fea.name].float().dim() > 1 else x[fea.name].float().unsqueeze(1))  # .unsqueeze(1).unsqueeze(1)

        if len(dense_values) > 0:
            dense_exists = True
            dense_values = torch.cat(dense_values, dim=1)
        if len(sparse_emb) > 0:
            sparse_exists = True
            # TODO: support concat dynamic embed_dim in dim 2
            # [batch_size, num_features, embed_dim]
            sparse_emb = torch.cat(sparse_emb, dim=1)

        if squeeze_dim:  # Note: if the emb_dim of sparse features is different, we must squeeze_dim
            if dense_exists and not sparse_exists:  # only input dense features
                return dense_values
            elif not dense_exists and sparse_exists:
                # squeeze dim to : [batch_size, num_features*embed_dim]
                return sparse_emb.flatten(start_dim=1)
            elif dense_exists and sparse_exists:
                # concat dense value with sparse embedding
                return torch.cat((sparse_emb.flatten(start_dim=1), dense_values), dim=1)
            else:
                raise ValueError("The input features can note be empty")
        else:
            if sparse_exists:
                return sparse_emb  # [batch_size, num_features, embed_dim]
            else:
                raise ValueError("If keep the original shape:[batch_size, num_features, embed_dim], expected %s in feature list, got %s" % ("SparseFeatures", features))


class InputMask(nn.Module):
    """Return inputs mask from given features

    Shape:
        - Input:
            x (dict): {feature_name: feature_value}, sequence feature value is a 2D tensor with shape:`(batch_size, seq_len)`,\
                      sparse/dense feature value is a 1D tensor with shape `(batch_size)`.
            features (list or SparseFeature or SequenceFeature): Note that the elements in features are either all instances of SparseFeature or all instances of SequenceFeature.
        - Output:
            - if input Sparse: `(batch_size, num_features)`
            - if input Sequence: `(batch_size, num_features_seq, seq_length)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, features):
        mask = []
        if not isinstance(features, list):
            features = [features]
        for fea in features:
            if isinstance(fea, SparseFeature) or isinstance(fea, SequenceFeature):
                if fea.padding_idx is not None:
                    fea_mask = x[fea.name].long() != fea.padding_idx
                else:
                    fea_mask = x[fea.name].long() != -1
                mask.append(fea_mask.unsqueeze(1).float())
            else:
                raise ValueError("Only SparseFeature or SequenceFeature support to get mask.")
        return torch.cat(mask, dim=1)


class LR(nn.Module):
    """Logistic Regression Module. It is the one Non-linear
    transformation for input feature.

    Args:
        input_dim (int): input size of Linear module.
        sigmoid (bool): whether to add sigmoid function before output.

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim, sigmoid=False):
        super().__init__()
        self.sigmoid = sigmoid
        self.fc = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        if self.sigmoid:
            return torch.sigmoid(self.fc(x))
        else:
            return self.fc(x)


class ConcatPooling(nn.Module):
    """Keep the origin sequence embedding shape

    Shape:
    - Input: `(batch_size, seq_length, embed_dim)`
    - Output: `(batch_size, seq_length, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        return x


class AveragePooling(nn.Module):
    """Pooling the sequence embedding matrix by `mean`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.mean(x, dim=1)
        else:
            sum_pooling_matrix = torch.bmm(mask, x).squeeze(1)
            non_padding_length = mask.sum(dim=-1)
            return sum_pooling_matrix / (non_padding_length.float() + 1e-16)


class SumPooling(nn.Module):
    """Pooling the sequence embedding matrix by `sum`.

    Shape:
        - Input
            x: `(batch_size, seq_length, embed_dim)`
            mask: `(batch_size, 1, seq_length)`
        - Output: `(batch_size, embed_dim)`
    """

    def __init__(self):
        super().__init__()

    def forward(self, x, mask=None):
        if mask is None:
            return torch.sum(x, dim=1)
        else:
            return torch.bmm(mask, x).squeeze(1)


class MLP(nn.Module):
    """Multi Layer Perceptron Module, it is the most widely used module for
    learning feature. Note we default add `BatchNorm1d` and `Activation`
    `Dropout` for each `Linear` Module.

    Args:
        input dim (int): input size of the first Linear Layer.
        output_layer (bool): whether this MLP module is the output layer. If `True`, then append one Linear(*,1) module.
        dims (list): output size of Linear Layer (default=[]).
        dropout (float): probability of an element to be zeroed (default = 0.5).
        activation (str): the activation function, support `[sigmoid, relu, prelu, dice, softmax]` (default='relu').

    Shape:
        - Input: `(batch_size, input_dim)`
        - Output: `(batch_size, 1)` or `(batch_size, dims[-1])`
    """

    def __init__(self, input_dim, output_layer=True, dims=None, dropout=0, activation="relu"):
        super().__init__()
        if dims is None:
            dims = []
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
        return self.mlp(x)


class FM(nn.Module):
    """The Factorization Machine module, mentioned in the `DeepFM paper
    <https://arxiv.org/pdf/1703.04247.pdf>`. It is used to learn 2nd-order
    feature interactions.

    Args:
        reduce_sum (bool): whether to sum in embed_dim (default = `True`).

    Shape:
        - Input: `(batch_size, num_features, embed_dim)`
        - Output: `(batch_size, 1)`` or ``(batch_size, embed_dim)`
    """

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        square_of_sum = torch.sum(x, dim=1)**2
        sum_of_square = torch.sum(x**2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class CIN(nn.Module):
    """Compressed Interaction Network

    Args:
        input_dim (int): input dim of input tensor.
        cin_size (list[int]): out channels of Conv1d.

    Shape:
        - Input: `(batch_size, num_features, embed_dim)`
        - Output: `(batch_size, 1)`
    """

    def __init__(self, input_dim, cin_size, split_half=True):
        super().__init__()
        self.num_layers = len(cin_size)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cin_size[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1, stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class CrossLayer(nn.Module):
    """
        Cross layer.
    Args:
        input_dim (int): input dim of input tensor
    """

    def __init__(self, input_dim):
        super(CrossLayer, self).__init__()
        self.w = torch.nn.Linear(input_dim, 1, bias=False)
        self.b = torch.nn.Parameter(torch.zeros(input_dim))

    def forward(self, x_0, x_i):
        x = self.w(x_i) * x_0 + self.b
        return x


class CrossNetwork(nn.Module):
    """CrossNetwork  mentioned in the DCN paper.

    Args:
        input_dim (int): input dim of input tensor

    Shape:
        - Input: `(batch_size, *)`
        - Output: `(batch_size, *)`

    """

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class CrossNetV2(nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([torch.nn.Linear(input_dim, input_dim, bias=False) for _ in range(num_layers)])
        self.b = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)])

    def forward(self, x):
        x0 = x
        for i in range(self.num_layers):
            x = x0 * self.w[i](x) + self.b[i] + x
        return x


class CrossNetMix(nn.Module):
    """ CrossNetMix improves CrossNetwork by:
        1. add MOE to learn feature interactions in different subspaces
        2. add nonlinear transformations in low-dimensional space
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
    """

    def __init__(self, input_dim, num_layers=2, low_rank=32, num_experts=4):
        super(CrossNetMix, self).__init__()
        self.num_layers = num_layers
        self.num_experts = num_experts

        # U: (input_dim, low_rank)
        self.u_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank))) for i in range(self.num_layers)])
        # V: (input_dim, low_rank)
        self.v_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, input_dim, low_rank))) for i in range(self.num_layers)])
        # C: (low_rank, low_rank)
        self.c_list = torch.nn.ParameterList([nn.Parameter(nn.init.xavier_normal_(torch.empty(num_experts, low_rank, low_rank))) for i in range(self.num_layers)])
        self.gating = nn.ModuleList([nn.Linear(input_dim, 1, bias=False) for i in range(self.num_experts)])

        self.bias = torch.nn.ParameterList([nn.Parameter(nn.init.zeros_(torch.empty(input_dim, 1))) for i in range(self.num_layers)])

    def forward(self, x):
        x_0 = x.unsqueeze(2)  # (bs, in_features, 1)
        x_l = x_0
        for i in range(self.num_layers):
            output_of_experts = []
            gating_score_experts = []
            for expert_id in range(self.num_experts):
                # (1) G(x_l)
                # compute the gating score by x_l
                gating_score_experts.append(self.gating[expert_id](x_l.squeeze(2)))

                # (2) E(x_l)
                # project the input x_l to $\mathbb{R}^{r}$
                v_x = torch.matmul(self.v_list[i][expert_id].t(), x_l)  # (bs, low_rank, 1)

                # nonlinear activation in low rank space
                v_x = torch.tanh(v_x)
                v_x = torch.matmul(self.c_list[i][expert_id], v_x)
                v_x = torch.tanh(v_x)

                # project back to $\mathbb{R}^{d}$
                uv_x = torch.matmul(self.u_list[i][expert_id], v_x)  # (bs, in_features, 1)

                dot_ = uv_x + self.bias[i]
                dot_ = x_0 * dot_  # Hadamard-product

                output_of_experts.append(dot_.squeeze(2))


# (3) mixture of low-rank experts
            output_of_experts = torch.stack(output_of_experts, 2)  # (bs, in_features, num_experts)
            gating_score_experts = torch.stack(gating_score_experts, 1)  # (bs, num_experts, 1)
            moe_out = torch.matmul(output_of_experts, gating_score_experts.softmax(1))
            x_l = moe_out + x_l  # (bs, in_features, 1)

        x_l = x_l.squeeze()  # (bs, in_features)
        return x_l


class SENETLayer(nn.Module):
    """
    A weighted feature gating system in the SENet paper
    Args:
        num_fields (int): number of feature fields

    Shape:
        - num_fields: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """

    def __init__(self, num_fields, reduction_ratio=3):
        super(SENETLayer, self).__init__()
        reduced_size = max(1, int(num_fields / reduction_ratio))
        self.mlp = nn.Sequential(nn.Linear(num_fields, reduced_size, bias=False), nn.ReLU(), nn.Linear(reduced_size, num_fields, bias=False), nn.ReLU())

    def forward(self, x):
        z = torch.mean(x, dim=-1, out=None)
        a = self.mlp(z)
        v = x * a.unsqueeze(-1)
        return v


class BiLinearInteractionLayer(nn.Module):
    """
    Bilinear feature interaction module, which is an improved model of the FFM model
     Args:
        num_fields (int): number of feature fields
        bilinear_type(str): the type bilinear interaction function
    Shape:
        - num_fields: `(batch_size, *)`
        - Output: `(batch_size, *)`
    """

    def __init__(self, input_dim, num_fields, bilinear_type="field_interaction"):
        super(BiLinearInteractionLayer, self).__init__()
        self.bilinear_type = bilinear_type
        if self.bilinear_type == "field_all":
            self.bilinear_layer = nn.Linear(input_dim, input_dim, bias=False)
        elif self.bilinear_type == "field_each":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i in range(num_fields)])
        elif self.bilinear_type == "field_interaction":
            self.bilinear_layer = nn.ModuleList([nn.Linear(input_dim, input_dim, bias=False) for i, j in combinations(range(num_fields), 2)])
        else:
            raise NotImplementedError()

    def forward(self, x):
        feature_emb = torch.split(x, 1, dim=1)
        if self.bilinear_type == "field_all":
            bilinear_list = [self.bilinear_layer(v_i) * v_j for v_i, v_j in combinations(feature_emb, 2)]
        elif self.bilinear_type == "field_each":
            bilinear_list = [self.bilinear_layer[i](feature_emb[i]) * feature_emb[j] for i, j in combinations(range(len(feature_emb)), 2)]
        elif self.bilinear_type == "field_interaction":
            bilinear_list = [self.bilinear_layer[i](v[0]) * v[1] for i, v in enumerate(combinations(feature_emb, 2))]
        return torch.cat(bilinear_list, dim=1)


class MultiInterestSA(nn.Module):
    """MultiInterest Attention mentioned in the Comirec paper.

    Args:
        embedding_dim (int): embedding dim of item embedding
        interest_num (int): num of interest
        hidden_dim (int): hidden dim

    Shape:
        - Input: seq_emb : (batch,seq,emb)
                 mask : (batch,seq,1)
        - Output: `(batch_size, interest_num, embedding_dim)`

    """

    def __init__(self, embedding_dim, interest_num, hidden_dim=None):
        super(MultiInterestSA, self).__init__()
        self.embedding_dim = embedding_dim
        self.interest_num = interest_num
        if hidden_dim is None:
            self.hidden_dim = self.embedding_dim * 4
        self.W1 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.hidden_dim), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.rand(self.hidden_dim, self.interest_num), requires_grad=True)
        self.W3 = torch.nn.Parameter(torch.rand(self.embedding_dim, self.embedding_dim), requires_grad=True)

    def forward(self, seq_emb, mask=None):
        H = torch.einsum('bse, ed -> bsd', seq_emb, self.W1).tanh()
        if mask is not None:
            A = torch.einsum('bsd, dk -> bsk', H, self.W2) + - \
                1.e9 * (1 - mask.float())
            A = F.softmax(A, dim=1)
        else:
            A = F.softmax(torch.einsum('bsd, dk -> bsk', H, self.W2), dim=1)
        A = A.permute(0, 2, 1)
        multi_interest_emb = torch.matmul(A, seq_emb)
        return multi_interest_emb


class CapsuleNetwork(nn.Module):
    """CapsuleNetwork mentioned in the Comirec and MIND paper.

    Args:
        hidden_size (int): embedding dim of item embedding
        seq_len (int): length of the item sequence
        bilinear_type (int): 0 for MIND, 2 for ComirecDR
        interest_num (int): num of interest
        routing_times (int): routing times

    Shape:
        - Input: seq_emb : (batch,seq,emb)
                 mask : (batch,seq,1)
        - Output: `(batch_size, interest_num, embedding_dim)`

    """

    def __init__(self, embedding_dim, seq_len, bilinear_type=2, interest_num=4, routing_times=3, relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.embedding_dim = embedding_dim  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        self.interest_num = interest_num
        self.routing_times = routing_times

        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim, bias=False), nn.ReLU())
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.embedding_dim, self.embedding_dim * self.interest_num, bias=False)
        else:
            self.w = nn.Parameter(torch.Tensor(1, self.seq_len, self.interest_num * self.embedding_dim, self.embedding_dim))

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:
            item_eb_hat = self.linear(item_eb)
            item_eb_hat = item_eb_hat.repeat(1, 1, self.interest_num)
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:
            u = torch.unsqueeze(item_eb, dim=2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u, dim=3)

        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.embedding_dim))
        item_eb_hat = torch.transpose(item_eb_hat, 1, 2).contiguous()
        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.embedding_dim))

        if self.stop_grad:
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        if self.bilinear_type > 0:
            capsule_weight = torch.zeros(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=item_eb.device, requires_grad=False)
        else:
            capsule_weight = torch.randn(item_eb_hat.shape[0], self.interest_num, self.seq_len, device=item_eb.device, requires_grad=False)

        for i in range(self.routing_times):  # 动态路由传播3次
            atten_mask = torch.unsqueeze(mask, 1).repeat(1, self.interest_num, 1)
            paddings = torch.zeros_like(atten_mask, dtype=torch.float)

            capsule_softmax_weight = F.softmax(capsule_weight, dim=-1)
            capsule_softmax_weight = torch.where(torch.eq(atten_mask, 0), paddings, capsule_softmax_weight)
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat_iter)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / \
                    (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

                delta_weight = torch.matmul(item_eb_hat_iter, torch.transpose(interest_capsule, 2, 3).contiguous())
                delta_weight = torch.reshape(delta_weight, (-1, self.interest_num, self.seq_len))
                capsule_weight = capsule_weight + delta_weight
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, True)
                scalar_factor = cap_norm / \
                    (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.embedding_dim))

        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule


class FFM(nn.Module):
    """The Field-aware Factorization Machine module, mentioned in the `FFM paper
    <https://dl.acm.org/doi/abs/10.1145/2959100.2959134>`. It explicitly models
    multi-channel second-order feature interactions, with each feature filed
    corresponding to one channel.

    Args:
        num_fields (int): number of feature fields.
        reduce_sum (bool): whether to sum in embed_dim (default = `True`).

    Shape:
        - Input: `(batch_size, num_fields, num_fields, embed_dim)`
        - Output: `(batch_size, num_fields*(num_fields-1)/2, 1)` or `(batch_size, num_fields*(num_fields-1)/2, embed_dim)`
    """

    def __init__(self, num_fields, reduce_sum=True):
        super().__init__()
        self.num_fields = num_fields
        self.reduce_sum = reduce_sum

    def forward(self, x):
        # compute (non-redundant) second order field-aware feature crossings
        crossed_embeddings = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                crossed_embeddings.append(x[:, i, j, :] * x[:, j, i, :])
        crossed_embeddings = torch.stack(crossed_embeddings, dim=1)

        # if reduce_sum is true, the crossing operation is effectively inner
        # product, other wise Hadamard-product
        if self.reduce_sum:
            crossed_embeddings = torch.sum(crossed_embeddings, dim=-1, keepdim=True)
        return crossed_embeddings


class CEN(nn.Module):
    """The Compose-Excitation Network module, mentioned in the `FAT-DeepFFM paper
    <https://arxiv.org/abs/1905.06336>`, a modified version of
    `Squeeze-and-Excitation Network” (SENet) (Hu et al., 2017)`. It is used to
    highlight the importance of second-order feature crosses.

    Args:
        embed_dim (int): the dimensionality of categorical value embedding.
        num_field_crosses (int): the number of second order crosses between feature fields.
        reduction_ratio (int): the between the dimensions of input layer and hidden layer of the MLP module.

    Shape:
        - Input: `(batch_size, num_fields, num_fields, embed_dim)`
        - Output: `(batch_size, num_fields*(num_fields-1)/2 * embed_dim)`
    """

    def __init__(self, embed_dim, num_field_crosses, reduction_ratio):
        super().__init__()

        # convolution weight (Eq.7 FAT-DeepFFM)
        self.u = torch.nn.Parameter(torch.rand(num_field_crosses, embed_dim), requires_grad=True)

        # two FC layers that computes the field attention
        self.mlp_att = MLP(num_field_crosses, dims=[num_field_crosses // reduction_ratio, num_field_crosses], output_layer=False, activation="relu")

    def forward(self, em):
        # compute descriptor vector (Eq.7 FAT-DeepFFM), output shape
        # [batch_size, num_field_crosses]
        d = F.relu((self.u.squeeze(0) * em).sum(-1))

        # compute field attention (Eq.9), output shape [batch_size,
        # num_field_crosses]
        s = self.mlp_att(d)

        # rescale original embedding with field attention (Eq.10), output shape
        # [batch_size, num_field_crosses, embed_dim]
        aem = s.unsqueeze(-1) * em
        return aem.flatten(start_dim=1)
