"""Trainer for RPG, the parallel semantic-ID generative recommender."""

import math
import os

import torch
from tqdm import tqdm

from ..basic.callback import EarlyStopper


class RPGTrainer(object):
    """Train and evaluate :class:`~torch_rechub.models.generative.rpg.RPGModel`.

    Training minimizes the mean cross-entropy over the digits of the next
    item's semantic ID. Evaluation is a full ranking task: the model produces
    an ordered top-k list and the single held-out item is looked up in it, so
    Recall@k coincides with hit rate@k and NDCG@k reduces to the discounted
    gain of the hit.

    Parameters
    ----------
    model : RPGModel
        Model to train.
    optimizer_fn : callable, default=torch.optim.AdamW
        Optimizer constructor.
    optimizer_params : dict, optional
        Parameters passed to the optimizer. Defaults to
        ``{'lr': 3e-4, 'weight_decay': 0.0}``.
    n_epoch : int, default=150
        Maximum number of epochs.
    earlystop_patience : int, default=20
        Stop after this many epochs without improving ``val_metric``.
    warmup_steps : int, default=10000
        Linear warmup length, after which the learning rate decays on a cosine
        schedule for the rest of training.
    max_grad_norm : float, default=1.0
        Gradient-norm clipping threshold. Set to ``0`` to disable.
    topk : tuple of int, default=(5, 10)
        Cutoffs reported by :meth:`evaluate`.
    val_metric : str, default='ndcg@10'
        Metric driving checkpointing and early stopping.
    device : str, default='cpu'
        Device used for training.
    model_path : str, default='./'
        Directory the best checkpoint is written to.

    Examples
    --------
    >>> trainer = RPGTrainer(model, optimizer_params={'lr': 0.01}, device='cuda')  # doctest: +SKIP
    >>> trainer.fit(train_loader, val_loader)  # doctest: +SKIP
    >>> trainer.evaluate(test_loader, use_graph=True)  # doctest: +SKIP
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=None,
        n_epoch=150,
        earlystop_patience=20,
        warmup_steps=10000,
        max_grad_norm=1.0,
        topk=(5,
              10),
        val_metric='ndcg@10',
        device='cpu',
        model_path='./',
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 3e-4, "weight_decay": 0.0}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        self.n_epoch = n_epoch
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.topk = tuple(topk)
        self.val_metric = val_metric
        self.model_path = model_path
        self.early_stopper = EarlyStopper(patience=earlystop_patience)

    def _build_scheduler(self, total_steps):
        """Linear warmup followed by cosine decay to zero."""

        def lr_scale(step):
            if step < self.warmup_steps:
                return step / max(1, self.warmup_steps)
            progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_scale)

    def train_one_epoch(self, data_loader, scheduler):
        """Run one training epoch and return the mean batch loss."""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(data_loader, total=len(data_loader), ncols=100, desc="train"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            _, loss = self.model(batch["input_ids"], batch["attention_mask"], batch["labels"])
            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader, use_graph=False, num_beams=50, propagation_steps=3):
        """Compute Recall@k and NDCG@k over a dataloader.

        Parameters
        ----------
        data_loader : torch.utils.data.DataLoader
            Loader built with
            :meth:`~torch_rechub.utils.data.RPGSeqDataset.collate_fn`.
        use_graph : bool, default=False
            Retrieve through the decoding graph instead of scoring the whole
            catalog. The graph must already be built.
        num_beams, propagation_steps : int
            Forwarded to the model when ``use_graph`` is set.

        Returns
        -------
        dict
            Maps ``'recall@k'`` / ``'ndcg@k'`` to their mean value.
        """
        self.model.eval()
        max_k = max(self.topk)
        totals = {}
        n_samples = 0
        for batch in tqdm(data_loader, total=len(data_loader), ncols=100, desc="eval"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            preds = self.model.generate(batch["input_ids"], batch["attention_mask"], batch["seq_lens"], topk=max_k, use_graph=use_graph, num_beams=num_beams, propagation_steps=propagation_steps)
            n_samples += len(preds)
            for name, value in self._rank_metrics(preds, batch["target"]).items():
                totals[name] = totals.get(name, 0.0) + value
        return {name: value / max(n_samples, 1) for name, value in totals.items()}

    def _rank_metrics(self, preds, targets):
        """Sum Recall@k and NDCG@k over a batch of ranked lists.

        With a single ground-truth item per user the ideal DCG is 1, so NDCG
        is just the discounted gain at the hit's rank.
        """
        hits = preds == targets.view(-1, 1)
        found = hits.any(dim=1)
        ranks = hits.float().argmax(dim=1)
        gains = 1.0 / torch.log2(ranks.float() + 2)
        results = {}
        for k in self.topk:
            hit_at_k = found & (ranks < k)
            results[f'recall@{k}'] = hit_at_k.float().sum().item()
            results[f'ndcg@{k}'] = (gains * hit_at_k).sum().item()
        return results

    def fit(self, train_dataloader, val_dataloader):
        """Train with early stopping on ``val_metric``.

        The best weights are restored into ``self.model`` and written to
        ``<model_path>/model.pth`` before returning.

        Returns
        -------
        dict
            Validation metrics of the best epoch.
        """
        scheduler = self._build_scheduler(self.n_epoch * len(train_dataloader))
        best_metrics = {}
        for epoch in range(self.n_epoch):
            train_loss = self.train_one_epoch(train_dataloader, scheduler)
            metrics = self.evaluate(val_dataloader)
            print(f"epoch {epoch} | train loss {train_loss:.4f} | " + " ".join(f"{k} {v:.4f}" for k, v in metrics.items()))

            score = metrics[self.val_metric]
            if score > self.early_stopper.best_auc:
                best_metrics = metrics
            if self.early_stopper.stop_training(score, self.model.state_dict()):
                print(f"early stopping at epoch {epoch}, best {self.val_metric} {self.early_stopper.best_auc:.4f}")
                break

        # best_weights stays None only if no epoch ever scored above zero.
        if self.early_stopper.best_weights is not None:
            self.model.load_state_dict(self.early_stopper.best_weights)
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))
        return best_metrics
