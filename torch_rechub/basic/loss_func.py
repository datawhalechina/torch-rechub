import torch
import torch.functional as F
import torch.nn as nn


class RegularizationLoss(nn.Module):
    """Unified L1/L2 regularization for embedding and dense parameters.

    Parameters
    ----------
    embedding_l1 : float, default=0.0
        L1 coefficient for embedding parameters.
    embedding_l2 : float, default=0.0
        L2 coefficient for embedding parameters.
    dense_l1 : float, default=0.0
        L1 coefficient for dense (non-embedding) parameters.
    dense_l2 : float, default=0.0
        L2 coefficient for dense (non-embedding) parameters.

    Examples
    --------
    >>> reg_loss_fn = RegularizationLoss(embedding_l2=1e-5, dense_l2=1e-5)
    >>> reg_loss = reg_loss_fn(model)
    >>> total_loss = task_loss + reg_loss
    """

    def __init__(self, embedding_l1=0.0, embedding_l2=0.0, dense_l1=0.0, dense_l2=0.0):
        super(RegularizationLoss, self).__init__()
        self.embedding_l1 = embedding_l1
        self.embedding_l2 = embedding_l2
        self.dense_l1 = dense_l1
        self.dense_l2 = dense_l2

    def forward(self, model):
        reg_loss = 0.0

        # Register normalization layers
        norm_params = set()
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                for param in module.parameters():
                    norm_params.add(id(param))

        # Register embedding layers
        embedding_params = set()
        for module in model.modules():
            if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                for param in module.parameters():
                    embedding_params.add(id(param))

        for param in model.parameters():
            if param.requires_grad:
                # Skip normalization layer parameters
                if id(param) in norm_params:
                    continue

                if id(param) in embedding_params:
                    if self.embedding_l1 > 0:
                        reg_loss += self.embedding_l1 * torch.sum(torch.abs(param))
                    if self.embedding_l2 > 0:
                        reg_loss += self.embedding_l2 * torch.sum(param**2)
                else:
                    if self.dense_l1 > 0:
                        reg_loss += self.dense_l1 * torch.sum(torch.abs(param))
                    if self.dense_l2 > 0:
                        reg_loss += self.dense_l2 * torch.sum(param**2)

        return reg_loss


class HingeLoss(torch.nn.Module):
    """Hinge loss for pairwise learning.

    Notes
    -----
    Reference: https://github.com/ustcml/RecStudio/blob/main/recstudio/model/loss_func.py
    """

    def __init__(self, margin=2, num_items=None):
        super().__init__()
        self.margin = margin
        self.n_items = num_items

    def forward(self, pos_score, neg_score):
        loss = torch.maximum(torch.max(neg_score, dim=-1).values - pos_score + self.margin, torch.tensor([0]).type_as(pos_score))
        if self.n_items is not None:
            impostors = neg_score - pos_score.view(-1, 1) + self.margin > 0
            rank = torch.mean(impostors, -1) * self.n_items
            return torch.mean(loss * torch.log(rank + 1))
        else:
            return torch.mean(loss)


class BPRLoss(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pos_score, neg_score):
        loss = torch.mean(-(pos_score - neg_score).sigmoid().log(), dim=-1)
        return loss


class NCELoss(torch.nn.Module):
    """Noise Contrastive Estimation (NCE) loss for recommender systems.

    Parameters
    ----------
    temperature : float, default=1.0
        Temperature for scaling logits.
    ignore_index : int, default=0
        Target index to ignore.
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Reduction applied to the output.

    Notes
    -----
    - Gutmann & Hyvärinen (2010), Noise-contrastive estimation.
    - HLLM: Hierarchical Large Language Model for Recommendation.

    Examples
    --------
    >>> nce_loss = NCELoss(temperature=0.1)
    >>> logits = torch.randn(32, 1000)
    >>> targets = torch.randint(0, 1000, (32,))
    >>> loss = nce_loss(logits, targets)
    """

    def __init__(self, temperature=1.0, ignore_index=0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets):
        """Compute NCE loss.

        Args:
            logits (torch.Tensor): Model output logits of shape (batch_size, vocab_size)
            targets (torch.Tensor): Target indices of shape (batch_size,)

        Returns:
            torch.Tensor: NCE loss value
        """
        # Scale logits by temperature
        logits = logits / self.temperature

        # Compute log softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get log probability of target class
        batch_size = targets.shape[0]
        target_log_probs = log_probs[torch.arange(batch_size), targets]

        # Create mask for ignore_index
        mask = targets != self.ignore_index

        # Compute loss
        loss = -target_log_probs

        # Apply mask
        if mask.any():
            loss = loss[mask]

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class InBatchNCELoss(torch.nn.Module):
    """In-batch NCE loss with explicit negatives.

    Parameters
    ----------
    temperature : float, default=0.1
        Temperature for scaling logits.
    ignore_index : int, default=0
        Target index to ignore.
    reduction : {'mean', 'sum', 'none'}, default='mean'
        Reduction applied to the output.

    Examples
    --------
    >>> loss_fn = InBatchNCELoss(temperature=0.1)
    >>> embeddings = torch.randn(32, 256)
    >>> item_embeddings = torch.randn(1000, 256)
    >>> targets = torch.randint(0, 1000, (32,))
    >>> loss = loss_fn(embeddings, item_embeddings, targets)
    """

    def __init__(self, temperature=0.1, ignore_index=0, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, embeddings, item_embeddings, targets):
        """Compute in-batch NCE loss.

        Args:
            embeddings (torch.Tensor): User/query embeddings of shape (batch_size, embedding_dim)
            item_embeddings (torch.Tensor): Item embeddings of shape (vocab_size, embedding_dim)
            targets (torch.Tensor): Target item indices of shape (batch_size,)

        Returns:
            torch.Tensor: In-batch NCE loss value
        """
        # Compute logits: (batch_size, vocab_size)
        logits = torch.matmul(embeddings, item_embeddings.t()) / self.temperature

        # Compute log softmax
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # Get log probability of target class
        batch_size = targets.shape[0]
        target_log_probs = log_probs[torch.arange(batch_size), targets]

        # Create mask for ignore_index
        mask = targets != self.ignore_index

        # Compute loss
        loss = -target_log_probs

        # Apply mask
        if mask.any():
            loss = loss[mask]

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
