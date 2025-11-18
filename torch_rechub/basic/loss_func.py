import torch
import torch.functional as F
import torch.nn as nn


class RegularizationLoss(nn.Module):
    """Unified L1/L2 Regularization Loss for embedding and dense parameters.
    
    Example:
        >>> reg_loss_fn = RegularizationLoss(embedding_l2=1e-5, dense_l2=1e-5)
        >>> # In model's forward or trainer
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
    """Hinge Loss for pairwise learning.
    reference: https://github.com/ustcml/RecStudio/blob/main/recstudio/model/loss_func.py

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


# loss = -torch.mean(F.logsigmoid(pos_score - torch.max(neg_score,
# dim=-1))) need v1.10
