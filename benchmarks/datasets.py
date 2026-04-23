"""Dataset builders for benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SequenceFeature, SparseFeature
from torch_rechub.utils.data import df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match


@dataclass
class MatchingDatasetBundle:
    """Prepared inputs and features for matching benchmarks."""

    x_train: dict[str, np.ndarray]
    y_train: np.ndarray
    x_test: dict[str, np.ndarray]
    all_item: dict[str, np.ndarray]
    user_features: list[Any]
    history_features: list[Any]
    youtube_user_features: list[Any]
    item_features: list[Any]
    neg_item_feature: list[Any]
    user_map: dict[int, Any]
    item_map: dict[int, Any]
    user_col: str
    item_col: str


@dataclass
class RankingDatasetBundle:
    """Prepared inputs and features for ranking benchmarks."""

    x: pd.DataFrame
    y: pd.Series
    dense_features: list[Any]
    sparse_features: list[Any]


def build_movielens_matching_dataset(config: dict[str, Any]) -> MatchingDatasetBundle:
    """Build the MovieLens list-wise matching benchmark dataset."""
    data_path = Path(config["path"])
    seq_max_len = int(config.get("seq_max_len", 50))
    neg_ratio = int(config.get("neg_ratio", 3))
    sample_method = int(config.get("sample_method", 1))
    padding = config.get("padding", "post")
    truncating = config.get("truncating", "post")

    data = pd.read_csv(data_path)
    data["cate_id"] = data["genres"].apply(lambda value: value.split("|")[0])
    sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]
    user_col, item_col = "user_id", "movie_id"

    feature_max_idx = {}
    user_map = {}
    item_map = {}
    for feature in sparse_features:
        label_encoder = LabelEncoder()
        data[feature] = label_encoder.fit_transform(data[feature]) + 1
        feature_max_idx[feature] = data[feature].max() + 1
        if feature == user_col:
            user_map = {encoded_id + 1: raw_id for encoded_id, raw_id in enumerate(label_encoder.classes_)}
        if feature == item_col:
            item_map = {encoded_id + 1: raw_id for encoded_id, raw_id in enumerate(label_encoder.classes_)}

    user_profile = data[[user_col, "gender", "age", "occupation", "zip"]].drop_duplicates(user_col)
    item_profile = data[[item_col, "cate_id"]].drop_duplicates(item_col)

    train_df, test_df = generate_seq_feature_match(
        data,
        user_col,
        item_col,
        time_col="timestamp",
        item_attribute_cols=[],
        sample_method=sample_method,
        mode=2,
        neg_ratio=neg_ratio,
        min_item=0,
    )
    x_train = gen_model_input(train_df, user_profile, user_col, item_profile, item_col, seq_max_len=seq_max_len, padding=padding, truncating=truncating)
    y_train = np.zeros(train_df.shape[0], dtype=np.int64)
    x_test = gen_model_input(test_df, user_profile, user_col, item_profile, item_col, seq_max_len=seq_max_len, padding=padding, truncating=truncating)

    embed_dim = int(config.get("embed_dim", 16))
    user_cols = ["user_id", "gender", "age", "occupation", "zip"]
    user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=embed_dim) for name in user_cols]
    history_features = [SequenceFeature("hist_movie_id", vocab_size=feature_max_idx[item_col], embed_dim=embed_dim, pooling="concat", shared_with=item_col)]
    youtube_user_features = user_features + [SequenceFeature("hist_movie_id", vocab_size=feature_max_idx[item_col], embed_dim=embed_dim, pooling="mean", shared_with=item_col)]
    item_features = [SparseFeature(item_col, vocab_size=feature_max_idx[item_col], embed_dim=embed_dim)]
    neg_item_feature = [SequenceFeature("neg_items", vocab_size=feature_max_idx[item_col], embed_dim=embed_dim, pooling="concat", shared_with=item_col)]

    return MatchingDatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        all_item=df_to_dict(item_profile),
        user_features=user_features,
        history_features=history_features,
        youtube_user_features=youtube_user_features,
        item_features=item_features,
        neg_item_feature=neg_item_feature,
        user_map=user_map,
        item_map=item_map,
        user_col=user_col,
        item_col=item_col,
    )


def build_matching_dataset(config: dict[str, Any]) -> MatchingDatasetBundle:
    """Dispatch matching dataset construction by dataset name."""
    name = config.get("name")
    if name == "ml-1m-sample":
        return build_movielens_matching_dataset(config)
    raise ValueError(f"Unsupported matching dataset: {name}")


def build_criteo_ranking_dataset(config: dict[str, Any]) -> RankingDatasetBundle:
    """Build the Criteo sample CTR benchmark dataset."""
    data_path = Path(config["path"])
    embed_dim = int(config.get("embed_dim", 16))
    data = pd.read_csv(data_path, compression="gzip" if str(data_path).endswith(".gz") else None)
    dense_feature_names = [name for name in data.columns.tolist() if name.startswith("I")]
    sparse_feature_names = [name for name in data.columns.tolist() if name.startswith("C")]

    data[sparse_feature_names] = data[sparse_feature_names].fillna("0")
    data[dense_feature_names] = data[dense_feature_names].fillna(0)

    for feature_name in list(dense_feature_names):
        sparse_feature_names.append(f"{feature_name}_cat")
        data[f"{feature_name}_cat"] = data[feature_name].apply(_convert_numeric_feature)

    scaler = MinMaxScaler()
    data[dense_feature_names] = scaler.fit_transform(data[dense_feature_names])

    for feature_name in sparse_feature_names:
        label_encoder = LabelEncoder()
        data[feature_name] = label_encoder.fit_transform(data[feature_name])

    dense_features = [DenseFeature(feature_name) for feature_name in dense_feature_names]
    sparse_features = [SparseFeature(feature_name, vocab_size=data[feature_name].nunique(), embed_dim=embed_dim) for feature_name in sparse_feature_names]
    y = data["label"]
    x = data.drop(columns=["label"])
    return RankingDatasetBundle(x=x, y=y, dense_features=dense_features, sparse_features=sparse_features)


def build_ranking_dataset(config: dict[str, Any]) -> RankingDatasetBundle:
    """Dispatch ranking dataset construction by dataset name."""
    name = config.get("name")
    if name == "criteo-sample":
        return build_criteo_ranking_dataset(config)
    raise ValueError(f"Unsupported ranking dataset: {name}")


def _convert_numeric_feature(value):
    value = int(value)
    if value > 2:
        return int(np.log(value)**2)
    return value - 2
