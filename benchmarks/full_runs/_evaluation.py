"""Evaluate full test splits with one NumPy array per batch to bound Python overhead."""

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from torch_rechub.trainers import CTRTrainer, MTLTrainer


@torch.no_grad()
def _predictions(model, batches, device, n_tasks):
    model.eval()
    labels, predictions = [], []
    for features, target in batches:
        features = {name: values.to(device) for name, values in features.items()}
        labels.append(target.reshape(-1, n_tasks).cpu().numpy().astype(np.uint8, copy=False))
        predictions.append(model(features).reshape(-1, n_tasks).cpu().numpy())
    if not labels:
        raise ValueError("Cannot evaluate an empty split")
    return np.concatenate(labels), np.concatenate(predictions)


class FullCTRTrainer(CTRTrainer):

    def evaluate(self, model, data_loader):
        labels, scores = _predictions(model, data_loader, self.device, 1)
        return float(roc_auc_score(labels.ravel(), scores.ravel()))

    def test_metrics(self, model, data_loader):
        """Compute CTR test AUC and LogLoss from the same probabilities."""
        labels, scores = _predictions(model, data_loader, self.device, 1)
        labels, scores = labels.ravel(), scores.ravel()
        return {"AUC": float(roc_auc_score(labels, scores)), "LogLoss": float(log_loss(labels, scores, labels=[0, 1]))}


class FullMTLTrainer(MTLTrainer):

    def evaluate(self, model, data_loader):
        labels, scores = _predictions(model, data_loader, self.device, self.n_task)
        return [float(roc_auc_score(labels[:, i], scores[:, i])) for i in range(self.n_task)]
