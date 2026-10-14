import collections

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from ...basic.layers import MLP


def kmeans(
    samples,
    num_clusters,
    num_iters=10,
):
    """
    Perform K-Means clustering on input samples and return cluster centers.

    This function applies the scikit-learn implementation of K-Means
    to cluster the input samples and returns the resulting cluster
    centers as a PyTorch tensor on the original device.

    Parameters
    ----------
    samples : torch.Tensor
        Input tensor of shape (N, D), where N is the number of samples
        and D is the feature dimension.
    num_clusters : int
        The number of clusters to form.
    num_iters : int, optional (default=10)
        Maximum number of iterations of the K-Means algorithm.

    Returns
    -------
    tensor_centers : torch.Tensor
        A tensor of shape (num_clusters, D) containing the cluster
        centers, located on the same device as the input samples.

    Notes
    -----
    This function converts the input tensor to a NumPy array and runs
    K-Means on the CPU using scikit-learn. Gradients are not preserved.
    """
    B, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device
    x = samples.cpu().detach().numpy()

    cluster = KMeans(n_clusters=num_clusters, max_iter=num_iters).fit(x)

    centers = cluster.cluster_centers_
    tensor_centers = torch.from_numpy(centers).to(device)

    return tensor_centers


@torch.no_grad()
def sinkhorn_algorithm(distances, epsilon, sinkhorn_iterations):
    Q = torch.exp(-distances / epsilon)

    B = Q.shape[0]  # number of samples to assign
    K = Q.shape[1]  # how many centroids per block (usually set to 256)

    # make the matrix sums to 1
    sum_Q = Q.sum(-1, keepdim=True).sum(-2, keepdim=True)
    Q /= sum_Q
    # print(Q.sum())
    for it in range(sinkhorn_iterations):

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= B

        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= K

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q


class VectorQuantizer(nn.Module):
    """VectorQuantizer: Single-stage vector quantization module.

    Quantizes input features using a learned codebook and optionally
    applies Sinkhorn-based soft assignment. Computes codebook and
    commitment losses for training.

    Parameters
    ----------
    n_e : int
        Number of embeddings (codebook size).
    e_dim : int
        Dimensionality of each embedding vector.
    beta : float, default=0.25
        Weight for the commitment loss term.
    kmeans_init : bool, default=False
        Whether to initialize embeddings with K-Means.
    kmeans_iters : int, default=10
        Number of K-Means iterations for initialization.
    sk_epsilon : float, default=0.003
        Entropy regularization coefficient for Sinkhorn assignment.
    sk_iters : int, default=100
        Number of Sinkhorn iterations.

    Shape
    -----
    Input
        x : torch.Tensor of shape (batch_size, ..., e_dim)
    Output
        x_q : torch.Tensor of shape (batch_size, ..., e_dim)
        loss : torch.Tensor, scalar quantization loss
        indices : torch.Tensor of shape (batch_size, ...), codebook indices

    Examples
    --------
    >>> vq = VectorQuantizer(n_e=512, e_dim=64)
    >>> x = torch.randn(32, 10, 64)
    >>> x_q, loss, indices = vq(x)
    >>> x_q.shape
    torch.Size([32, 10, 64])
    """

    def __init__(
        self,
        n_e,
        e_dim,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=10,
        sk_epsilon=0.003,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilon = sk_epsilon
        self.sk_iters = sk_iters

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if not kmeans_init:
            self.initted = True
            self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        else:
            self.initted = False
            self.embedding.weight.data.zero_()

    def get_codebook(self):
        """
        Return the current codebook embeddings.

        Returns
        -------
        torch.Tensor
            A tensor of shape (n_e, e_dim) containing the embedding vectors.
        """
        return self.embedding.weight

    def get_codebook_entry(self, indices, shape=None):
        """
        Retrieve codebook entries corresponding to given indices.

        Parameters
        ----------
        indices : torch.Tensor
            Tensor of indices selecting codebook entries.
        shape : tuple of int, optional
            Desired output shape after reshaping the retrieved embeddings.

        Returns
        -------
        torch.Tensor
            Quantized vectors corresponding to the provided indices.
        """
        z_q = self.embedding(indices)
        if shape is not None:
            z_q = z_q.view(shape)

        return z_q

    def init_emb(self, data):
        """
        Initialize the codebook embeddings using K-Means clustering.
        """
        centers = kmeans(
            data,
            self.n_e,
            self.kmeans_iters,
        )

        self.embedding.weight.data.copy_(centers)
        self.initted = True

    @staticmethod
    def center_distance_for_constraint(distances):
        """
        Center and normalize distance values for constrained optimization.
        """
        # distances: B, K
        max_distance = distances.max()
        min_distance = distances.min()

        middle = (max_distance + min_distance) / 2
        amplitude = max_distance - middle + 1e-5
        assert amplitude > 0
        centered_distances = (distances - middle) / amplitude
        return centered_distances

    def forward(self, x, use_sk=True):
        """
        Apply vector quantization to the input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension corresponds to the embedding
            dimension.
        use_sk : bool, optional (default=True)
            Whether to use Sinkhorn-based soft assignment instead of
            hard nearest-neighbor assignment.

        Returns
        -------
        x_q : torch.Tensor
            Quantized output tensor with the same shape as the input.
        loss : torch.Tensor
            Vector quantization loss consisting of codebook and commitment
            terms.
        indices : torch.Tensor
            Indices of the selected codebook entries for each input vector.

        Notes
        -----
        During training, the codebook may be initialized using K-Means
        if it has not been initialized yet. Gradients are preserved using
        the straight-through estimator.
        """
        # Flatten input
        latent = x.view(-1, self.e_dim)

        if not self.initted and self.training:
            self.init_emb(latent)

        # Calculate the L2 Norm between latent and Embedded weights
        d = (torch.sum(latent**2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1, keepdim=True).t() - 2 * torch.matmul(latent, self.embedding.weight.t()))
        if not use_sk or self.sk_epsilon <= 0:
            indices = torch.argmin(d, dim=-1)
        else:
            d = self.center_distance_for_constraint(d)
            d = d.double()
            Q = sinkhorn_algorithm(d, self.sk_epsilon, self.sk_iters)

            if torch.isnan(Q).any() or torch.isinf(Q).any():
                print("Sinkhorn Algorithm returns nan/inf values.")
            indices = torch.argmax(Q, dim=-1)

        # indices = torch.argmin(d, dim=-1)

        x_q = self.embedding(indices).view(x.shape)

        # compute loss for embedding
        commitment_loss = F.mse_loss(x_q.detach(), x)
        codebook_loss = F.mse_loss(x_q, x.detach())
        loss = codebook_loss + self.beta * commitment_loss

        # preserve gradients
        x_q = x + (x_q - x).detach()

        indices = indices.view(x.shape[:-1])

        return x_q, loss, indices


class ResidualVectorQuantizer(nn.Module):
    """ResidualVectorQuantizer: Multi-stage residual vector quantization.

    Applies a sequence of VectorQuantizer modules to progressively
    quantize the residuals of the input. Computes mean quantization
    loss across all stages. References:SoundStream: An End-to-End Neural Audio Codec
    https://arxiv.org/pdf/2107.03312.pdf

    Parameters
    ----------
    n_e_list : list of int
        Number of embeddings for each residual quantization stage.
    e_dim : int
        Dimensionality of each embedding vector.
    sk_epsilons : list of float
        Entropy regularization coefficients for Sinkhorn assignment
        at each stage.
    beta : float, default=0.25
        Weight for the commitment loss term.
    kmeans_init : bool, default=False
        Whether to initialize embeddings with K-Means.
    kmeans_iters : int, default=100
        Number of K-Means iterations for initialization.
    sk_iters : int, default=100
        Number of Sinkhorn iterations.

    Shape
    -----
    Input
        x : torch.Tensor of shape (batch_size, ..., e_dim)
    Output
        x_q : torch.Tensor of shape (batch_size, ..., e_dim)
        mean_losses : torch.Tensor, scalar mean quantization loss
        all_indices : torch.Tensor of shape (batch_size, ..., num_quantizers)

    Examples
    --------
    >>> rvq = ResidualVectorQuantizer(n_e_list=[512, 512], e_dim=64, sk_epsilons=[0.003, 0.003])
    >>> x = torch.randn(32, 10, 64)
    >>> x_q, loss, indices = rvq(x)
    >>> x_q.shape
    torch.Size([32, 10, 64])
    """

    def __init__(
        self,
        n_e_list,
        e_dim,
        sk_epsilons,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_iters=100,
    ):
        super().__init__()
        self.n_e_list = n_e_list
        self.e_dim = e_dim
        self.num_quantizers = len(n_e_list)
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        sk_epsilons = [0.0] * len(n_e_list) if sk_epsilons is None else sk_epsilons
        if len(sk_epsilons) != len(n_e_list):
            raise ValueError("sk_epsilons must have one value per codebook")
        self.vq_layers = nn.ModuleList([VectorQuantizer(n_e, e_dim, beta=self.beta, kmeans_init=self.kmeans_init, kmeans_iters=self.kmeans_iters, sk_epsilon=sk_epsilon, sk_iters=sk_iters) for n_e, sk_epsilon in zip(n_e_list, sk_epsilons)])

    def get_codebook(self):
        """
        Return the stacked codebooks from all residual quantizers.
        """
        all_codebook = []
        for quantizer in self.vq_layers:
            codebook = quantizer.get_codebook()
            all_codebook.append(codebook)
        return torch.stack(all_codebook)

    def forward(self, x, use_sk=True):
        """
        Apply residual vector quantization to the input features.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor whose last dimension corresponds to the embedding
            dimension.
        use_sk : bool, optional (default=True)
            Whether to use Sinkhorn-based soft assignment for each
            quantization stage.

        Returns
        -------
        x_q : torch.Tensor
            Quantized output obtained by summing the outputs of all
            residual quantizers.
        mean_losses : torch.Tensor
            Mean vector quantization loss averaged over all stages.
        all_indices : torch.Tensor
            Tensor containing codebook indices from all quantizers,
            stacked along the last dimension.

        Notes
        -----
        Each quantization stage operates on the residual from the
        previous stage, enabling progressive refinement of the
        quantized representation.
        """
        all_losses = []
        all_indices = []

        x_q = 0
        residual = x
        for quantizer in self.vq_layers:
            x_res, loss, indices = quantizer(residual, use_sk=use_sk)
            residual = residual - x_res
            x_q = x_q + x_res

            all_losses.append(loss)
            all_indices.append(indices)

        mean_losses = torch.stack(all_losses).mean()
        all_indices = torch.stack(all_indices, dim=-1)

        return x_q, mean_losses, all_indices


class RQVAEModel(nn.Module):
    """RQVAEModel: Residual Quantized Variational Autoencoder.

    Implements a VAE with a multi-stage residual vector quantizer
    (ResidualVectorQuantizer) for latent discretization.

    Parameters
    ----------
    in_dim : int, default=768
        Input feature dimension.
    num_emb_list : list of int
        Number of embeddings for each residual quantization stage.
    e_dim : int, default=64
        Dimension of each embedding vector.
    layers : list of int
        Hidden layer sizes for the encoder/decoder MLP.
    dropout_prob : float, default=0.0
        Dropout probability applied to MLP layers.
    bn : bool, default=False
        Whether to use batch normalization in MLP layers.
    loss_type : str, default="mse"
        Reconstruction loss type, either "mse" or "l1".
    quant_loss_weight : float, default=1.0
        Weight for the vector quantization loss.
    beta : float, default=0.25
        Commitment loss weight in the vector quantizers.
    kmeans_init : bool, default=False
        Whether to initialize codebooks using K-Means.
    kmeans_iters : int, default=100
        Number of K-Means iterations for initialization.
    sk_epsilons : list of float
        Entropy regularization coefficients for Sinkhorn assignment.
    sk_iters : int, default=100
        Number of Sinkhorn iterations for each quantizer.

    Shape
    -----
    Input
        x : torch.Tensor of shape (batch_size, in_dim)
    Output
        out : torch.Tensor of shape (batch_size, in_dim)
        rq_loss : torch.Tensor, scalar quantization loss
        indices : torch.Tensor of shape (batch_size, num_quantizers)

    Examples
    --------
    >>> model = RQVAEModel(in_dim=768, num_emb_list=[512,512], e_dim=64, layers=[256,128])
    >>> x = torch.randn(32, 768)
    >>> out, rq_loss, indices = model(x)
    >>> out.shape
    torch.Size([32, 768])
    """

    def __init__(
        self,
        in_dim=768,
        num_emb_list=None,
        e_dim=64,
        layers=None,
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        beta=0.25,
        kmeans_init=False,
        kmeans_iters=100,
        sk_epsilons=None,
        sk_iters=100,
    ):
        super(RQVAEModel, self).__init__()

        self.in_dim = in_dim
        self.num_emb_list = num_emb_list
        self.e_dim = e_dim

        self.layers = layers
        self.dropout_prob = dropout_prob
        self.bn = bn
        self.loss_type = loss_type
        self.quant_loss_weight = quant_loss_weight
        self.beta = beta
        self.kmeans_init = kmeans_init
        self.kmeans_iters = kmeans_iters
        self.sk_epsilons = sk_epsilons
        self.sk_iters = sk_iters
        self.encode_layer_dims = [self.in_dim] + self.layers + [self.e_dim]
        self.encoder = MLP(input_dim=self.encode_layer_dims[0], dims=self.encode_layer_dims[1:], output_layer=False, dropout=self.dropout_prob, activation="relu", batchnorm=self.bn)
        self.rq = ResidualVectorQuantizer(
            num_emb_list,
            e_dim,
            beta=self.beta,
            kmeans_init=self.kmeans_init,
            kmeans_iters=self.kmeans_iters,
            sk_epsilons=self.sk_epsilons,
            sk_iters=self.sk_iters,
        )

        self.decode_layer_dims = self.encode_layer_dims[::-1]
        self.decoder = MLP(input_dim=self.decode_layer_dims[0], dims=self.decode_layer_dims[1:], output_layer=False, dropout=self.dropout_prob, activation="relu", batchnorm=self.bn)

    def forward(self, x, use_sk=True):
        """Forward pass.
        Parameters
        ----------
            x (torch.Tensor): Input feature tensor of shape
                ``(batch_size, in_dim)``.
            use_sk (bool, optional): Whether to use Sinkhorn-based soft
                assignment in the residual vector quantizer. Default: ``True``.

        Returns
        -------
            out (torch.Tensor): Reconstructed output tensor of shape
                ``(batch_size, in_dim)``.
            rq_loss (torch.Tensor): Scalar residual vector quantization loss.
            indices (torch.Tensor): Codebook indices from all quantization
                stages, shape ``(batch_size, num_quantizers)``.
        """
        x = self.encoder(x)
        x_q, rq_loss, indices = self.rq(x, use_sk=use_sk)
        out = self.decoder(x_q)

        return out, rq_loss, indices

    @torch.no_grad()
    def get_indices(self, xs, use_sk=False):
        """
        Obtain residual quantizer codebook indices for input features.

        Parameters
        ----------
        xs : torch.Tensor
            Input tensor of shape (batch_size, in_dim)
        use_sk : bool, default=False
            Whether to use Sinkhorn-based soft assignment.

        Returns
        -------
        sids : torch.Tensor
        Codebook indices of shape (batch_size, num_quantizers)
        """
        x_e = self.encoder(xs)
        _, _, sids = self.rq(x_e, use_sk=use_sk)
        return sids

    def compute_loss(self, out, quant_loss, xs=None):
        """
        Compute total loss combining reconstruction and quantization losses.

        Parameters
        ----------
        out : torch.Tensor
            Reconstructed output tensor, shape (batch_size, in_dim)
        quant_loss : torch.Tensor
            Vector quantization loss scalar
        xs : torch.Tensor
            Ground-truth input tensor, shape (batch_size, in_dim)

        Returns
        -------
        loss_total : torch.Tensor
            Combined reconstruction and quantization loss
        loss_recon : torch.Tensor
            Reconstruction loss only
        """
        if self.loss_type == 'mse':
            loss_recon = F.mse_loss(out, xs, reduction='mean')
        elif self.loss_type == 'l1':
            loss_recon = F.l1_loss(out, xs, reduction='mean')
        else:
            raise ValueError('incompatible loss type')

        loss_total = loss_recon + self.quant_loss_weight * quant_loss

        return loss_total, loss_recon

    def _check_collision(self, all_sids_str):
        """
        Check whether there are duplicate semantic IDs in the dataset.
        """
        tot_item = len(all_sids_str)
        tot_indice = len(set(all_sids_str.tolist()))
        return tot_item == tot_indice

    def _get_sids_count(self, all_indices_str):
        """
        Count occurrences of each semantic ID.
        """
        indices_count = collections.defaultdict(int)
        for index in all_indices_str:
            indices_count[index] += 1
        return indices_count

    def _get_collision_item(self, all_indices_str):
        """
        Identify groups of items sharing the same semantic ID (collisions).
        """
        index2id = {}
        for i, index in enumerate(all_indices_str):
            if index not in index2id:
                index2id[index] = []
            index2id[index].append(i)

        collision_item_groups = []

        for index in index2id:
            if len(index2id[index]) > 1:
                collision_item_groups.append(index2id[index])

        return collision_item_groups

    @torch.no_grad()
    def generate_semantic_ids(self, data, data_loader, prefix=None, use_sk=False, device='cuda', item_ids=None):
        """Export row-aligned, unique semantic IDs using a collision suffix.

        The loader must visit every row exactly once in dataset order. Optional
        item_ids maps those rows to business IDs; otherwise keys are row indices.
        Re-export the complete catalog after changing RQ-VAE or catalog order.
        """
        import warnings

        from torch.utils.data import SequentialSampler

        if not isinstance(data_loader.sampler, SequentialSampler) or data_loader.drop_last:
            raise ValueError("Semantic ID export requires shuffle=False and drop_last=False")
        if len(data) != len(data_loader.dataset) or len(data) == 0:
            raise ValueError("data and loader must describe the same non-empty catalog")
        prefix = prefix or ["<a_{}>", "<b_{}>", "<c_{}>", "<d_{}>", "<e_{}>"]
        stages = len(self.num_emb_list)
        if len(prefix) < stages:
            raise ValueError("prefix must cover every quantization stage")
        item_ids = list(range(len(data))) if item_ids is None else list(item_ids)
        if len(item_ids) != len(data) or len({str(item) for item in item_ids}) != len(item_ids):
            raise ValueError("item_ids must be unique and aligned with catalog rows")
        if use_sk:
            warnings.warn("Export uses nearest codes plus a collision suffix; use_sk is retained for compatibility")
        codes = []
        was_training = self.training
        self.eval()
        try:
            for batch in tqdm(data_loader, desc="export semantic IDs"):
                indices = self.get_indices(batch.to(device), use_sk=False).cpu().tolist()
                codes.extend([tuple(index) for index in indices])
        finally:
            self.train(was_training)
        if len(codes) != len(item_ids):
            raise ValueError("Loader did not visit each catalog row exactly once")

        # Stable within a fixed catalog, independent of training batch order.
        # No Sinkhorn retries and no mutation of quantizer sk_epsilon at export.
        suffix_template = prefix[stages] if len(prefix) > stages else "<collision_{}>"
        counts = collections.defaultdict(int)
        result = {}
        for row in sorted(range(len(item_ids)), key=lambda row: str(item_ids[row])):
            code = codes[row]
            suffix = counts[code]
            counts[code] += 1
            result[item_ids[row]] = [prefix[stage].format(value) for stage, value in enumerate(code)] + [suffix_template.format(suffix)]
        if len({tuple(code) for code in result.values()}) != len(result):
            raise ValueError("prefix templates do not distinguish quantization stages/collisions")
        print(f"Exported {len(result)} unique semantic IDs; raw collision rate: {(len(codes) - len(counts)) / len(codes):.4f}")
        return result
