"""Benchmark metric helpers."""

from __future__ import annotations

import collections
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

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


def evaluate_binary_ranking(y_true, y_pred) -> dict[str, float]:
    """Evaluate binary ranking/CTR predictions."""
    y_true_array = np.asarray(y_true).reshape(-1)
    y_pred_array = np.asarray(y_pred).reshape(-1)
    y_pred_array = np.clip(y_pred_array, 1e-7, 1 - 1e-7)
    return {
        "AUC": float(roc_auc_score(y_true_array,
                                   y_pred_array)),
        "LogLoss": float(log_loss(y_true_array,
                                  y_pred_array)),
    }
