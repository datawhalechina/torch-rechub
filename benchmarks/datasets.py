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


@dataclass
class MultiTaskDatasetBundle:
    """Prepared inputs and features for multi-task benchmarks."""

    x_train: dict[str, np.ndarray]
    y_train: np.ndarray
    x_val: dict[str, np.ndarray]
    y_val: np.ndarray
    x_test: dict[str, np.ndarray]
    y_test: np.ndarray
    features: list[Any]
    user_features: list[Any]
    item_features: list[Any]
    task_names: list[str]
    task_types: list[str]


def build_movielens_matching_dataset(config: dict[str, Any], embed_dim: int, mode: int) -> MatchingDatasetBundle:
    """Build the MovieLens list-wise matching benchmark dataset."""
    if mode != 2:
        raise ValueError(f"Phase-1 matching benchmark only supports list-wise mode (mode=2); got mode={mode}.")

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
        mode=mode,
        neg_ratio=neg_ratio,
        min_item=0,
    )
    x_train = gen_model_input(train_df, user_profile, user_col, item_profile, item_col, seq_max_len=seq_max_len, padding=padding, truncating=truncating)
    y_train = np.zeros(train_df.shape[0], dtype=np.int64)
    x_test = gen_model_input(test_df, user_profile, user_col, item_profile, item_col, seq_max_len=seq_max_len, padding=padding, truncating=truncating)

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


def build_matching_dataset(config: dict[str, Any], embed_dim: int, mode: int) -> MatchingDatasetBundle:
    """Dispatch matching dataset construction by dataset name."""
    name = config.get("name")
    if name == "ml-1m-sample":
        return build_movielens_matching_dataset(config, embed_dim=embed_dim, mode=mode)
    raise ValueError(f"Unsupported matching dataset: {name}")


def build_criteo_ranking_dataset(config: dict[str, Any], embed_dim: int) -> RankingDatasetBundle:
    """Build the Criteo sample CTR benchmark dataset."""
    data_path = Path(config["path"])
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


def build_ranking_dataset(config: dict[str, Any], embed_dim: int) -> RankingDatasetBundle:
    """Dispatch ranking dataset construction by dataset name."""
    name = config.get("name")
    if name == "criteo-sample":
        return build_criteo_ranking_dataset(config, embed_dim=embed_dim)
    raise ValueError(f"Unsupported ranking dataset: {name}")


def _convert_numeric_feature(value):
    value = int(value)
    if value > 2:
        return int(np.log(value)**2)
    return value - 2


_CENSUS_DENSE_COLS = [
    "age",
    "wage per hour",
    "capital gains",
    "capital losses",
    "divdends from stocks",
    "num persons worked for employer",
    "weeks worked in year",
]
_CENSUS_ESMM_USER_COLS = ["industry code", "occupation code", "race", "education", "sex"]


def build_census_multitask_dataset(config: dict[str, Any], embed_dim: int, model_name: str) -> MultiTaskDatasetBundle:
    """Build the Census-Income multi-task benchmark dataset.

    Two tasks drawn from Census-Income: income (cvr_label) and marital-status (ctr_label).
    When ``model_name == 'ESMM'`` we additionally emit the derived ``ctcvr_label`` and split
    the sparse columns into user/item groups, matching the original paper setup.
    """
    data_dir = Path(config["path"])
    df_train = pd.read_csv(data_dir / "census_income_train_sample.csv")
    df_val = pd.read_csv(data_dir / "census_income_val_sample.csv")
    df_test = pd.read_csv(data_dir / "census_income_test_sample.csv")
    train_idx = df_train.shape[0]
    val_idx = train_idx + df_val.shape[0]

    data = pd.concat([df_train, df_val, df_test], axis=0, ignore_index=True).fillna(0)
    data = data.rename(columns={"income": "cvr_label", "marital status": "ctr_label"})
    data["ctcvr_label"] = data["cvr_label"] * data["ctr_label"]

    non_label_cols = [col for col in data.columns if col not in ("cvr_label", "ctr_label", "ctcvr_label")]
    dense_cols = [col for col in _CENSUS_DENSE_COLS if col in non_label_cols]
    sparse_cols = [col for col in non_label_cols if col not in dense_cols]

    if model_name == "ESMM":
        label_cols = ["cvr_label", "ctr_label", "ctcvr_label"]
        used_cols = sparse_cols
        user_cols = [col for col in _CENSUS_ESMM_USER_COLS if col in sparse_cols]
        item_cols = [col for col in sparse_cols if col not in user_cols]
        user_features = [SparseFeature(col, int(data[col].max()) + 1, embed_dim=embed_dim) for col in user_cols]
        item_features = [SparseFeature(col, int(data[col].max()) + 1, embed_dim=embed_dim) for col in item_cols]
        features: list[Any] = []
        task_types = ["classification", "classification", "classification"]
    else:
        label_cols = ["cvr_label", "ctr_label"]
        used_cols = sparse_cols + dense_cols
        user_features, item_features = [], []
        features = [SparseFeature(col, int(data[col].max()) + 1, embed_dim=embed_dim) for col in sparse_cols] + [DenseFeature(col) for col in dense_cols]
        task_types = ["classification", "classification"]

    x_train = {name: data[name].values[:train_idx] for name in used_cols}
    x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
    x_test = {name: data[name].values[val_idx:] for name in used_cols}
    y_train = data[label_cols].values[:train_idx]
    y_val = data[label_cols].values[train_idx:val_idx]
    y_test = data[label_cols].values[val_idx:]

    return MultiTaskDatasetBundle(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
        features=features,
        user_features=user_features,
        item_features=item_features,
        task_names=label_cols,
        task_types=task_types,
    )


def build_multitask_dataset(config: dict[str, Any], embed_dim: int, model_name: str) -> MultiTaskDatasetBundle:
    """Dispatch multi-task dataset construction by dataset name."""
    name = config.get("name")
    if name == "census-income-sample":
        return build_census_multitask_dataset(config, embed_dim=embed_dim, model_name=model_name)
    raise ValueError(f"Unsupported multi-task dataset: {name}")
