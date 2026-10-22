"""Benchmark metric helpers."""

from __future__ import annotations

import collections
import math
import re
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from torch_rechub.basic.metric import topk_metrics
from torch_rechub.utils.match import Annoy


def evaluate_matching_topk(data, user_embedding, item_embedding, topk: int) -> dict[str, float]:
    """Evaluate matching embeddings with ANN top-k retrieval."""
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)
    recall_count = min(topk, len(data.all_item[data.item_col]))

    match_res = collections.defaultdict(dict)
    for user_id, user_emb in zip(data.x_test[data.user_col], user_embedding):
        if len(user_emb.shape) == 2:
            item_indices = []
            item_scores = []
            for interest_idx in range(user_emb.shape[0]):
                current_indices, current_scores = annoy.query(v=user_emb[interest_idx], n=recall_count)
                item_indices += current_indices
                item_scores += current_scores
            result_df = pd.DataFrame({"item": item_indices, "score": item_scores})
            result_df = result_df.sort_values(by="score", ascending=True)
            result_df = result_df.drop_duplicates(subset=["item"], keep="first", inplace=False)
            recall_item_list = result_df["item"][:recall_count].values
            match_res[data.user_map[user_id]] = np.vectorize(data.item_map.get)(data.all_item[data.item_col][recall_item_list])
        else:
            item_indices, _ = annoy.query(v=user_emb, n=recall_count)
            match_res[data.user_map[user_id]] = np.vectorize(data.item_map.get)(data.all_item[data.item_col][item_indices])

    ground_truth_df = pd.DataFrame({data.user_col: data.x_test[data.user_col], data.item_col: data.x_test[data.item_col]})
    ground_truth_df[data.user_col] = ground_truth_df[data.user_col].map(data.user_map)
    ground_truth_df[data.item_col] = ground_truth_df[data.item_col].map(data.item_map)
    user_pos_item = ground_truth_df.groupby(data.user_col).agg(list).reset_index()
    ground_truth = dict(zip(user_pos_item[data.user_col], user_pos_item[data.item_col]))

    raw_metrics = topk_metrics(y_true=ground_truth, y_pred=match_res, topKs=[recall_count])
    return _parse_topk_metrics(raw_metrics)


def _parse_topk_metrics(raw_metrics: dict[str, list[str]]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    pattern = re.compile(r"^(?P<name>[^@]+)@(?P<topk>\d+): (?P<value>[-+]?\d*\.?\d+)$")
    for values in raw_metrics.values():
        for item in values:
            match = pattern.match(item)
            if not match:
                continue
            parsed[f"{match.group('name')}@{match.group('topk')}"] = float(match.group("value"))
    return parsed


def shape_as_list(value: Any) -> list[int]:
    """Return tensor-like shape as a serialization-friendly list."""
    return [int(dim) for dim in value.shape]


def _metric_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_pred))


def _metric_logloss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(log_loss(y_true, y_pred))


def _metric_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_pred >= 0.5).astype(int) == y_true.astype(int)))


def _metric_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(mean_squared_error(y_true, y_pred))


#: Registry of ranking/CTR metric name -> function.  Extend here to expose new metrics
#: through ``metrics: [...]`` in YAML.
BINARY_RANKING_METRICS: dict[str,
                             Callable[[np.ndarray,
                                       np.ndarray],
                                      float]] = {
                                          "AUC": _metric_auc,
                                          "LogLoss": _metric_logloss,
                                          "Accuracy": _metric_accuracy,
                                          "MSE": _metric_mse,
                                      }

DEFAULT_BINARY_RANKING_METRICS = ("AUC", "LogLoss")


def evaluate_binary_ranking(y_true, y_pred, metric_names: list[str] | None = None) -> dict[str, float]:
    """Evaluate binary ranking/CTR predictions against the requested metric names.

    ``metric_names`` selects a subset from :data:`BINARY_RANKING_METRICS`; unknown names
    raise ``ValueError`` so typos are caught instead of silently skipped.
    """
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)
    y_pred_clipped = np.clip(y_pred_array, 1e-7, 1 - 1e-7)

    requested = list(metric_names) if metric_names else list(DEFAULT_BINARY_RANKING_METRICS)
    unknown = [name for name in requested if name not in BINARY_RANKING_METRICS]
    if unknown:
        raise ValueError(f"Unknown binary-ranking metrics {unknown}; available: {sorted(BINARY_RANKING_METRICS)}")

    results: dict[str, float] = {}
    for name in requested:
        # LogLoss needs clipped probabilities; AUC / Accuracy / MSE tolerate raw scores.
        array = y_pred_clipped if name == "LogLoss" else y_pred_array
        results[name] = BINARY_RANKING_METRICS[name](y_true_array, array)
    return results


def evaluate_multitask_scores(scores, task_names: list[str], task_types: list[str]) -> dict[str, float]:
    """Turn MTLTrainer.evaluate() per-task scores into a flat, named metric dict.

    NaN scores (e.g. degenerate AUC when a test split has only one class) are
    recorded as-is per task but excluded from the ``<metric>_mean`` aggregate so
    that one broken task does not poison the summary.
    """
    if len(scores) != len(task_names) or len(scores) != len(task_types):
        raise ValueError("scores, task_names, and task_types must have the same length")
    metrics: dict[str, float] = {}
    total_by_metric: dict[str, int] = {}
    valid_by_metric: dict[str, list[float]] = {}
    for score, name, task_type in zip(scores, task_names, task_types):
        metric_name = "AUC" if task_type == "classification" else "MSE"
        value = float(score)
        metrics[f"{metric_name}[{name}]"] = value
        total_by_metric[metric_name] = total_by_metric.get(metric_name, 0) + 1
        if not math.isnan(value):
            valid_by_metric.setdefault(metric_name, []).append(value)
    for metric_name, values in valid_by_metric.items():
        if total_by_metric.get(metric_name, 0) > 1:
            metrics[f"{metric_name}_mean"] = float(sum(values) / len(values))
    return metrics
