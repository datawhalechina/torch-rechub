"""Model registry for benchmark runs."""

from __future__ import annotations

from typing import Any

from benchmarks.datasets import MatchingDatasetBundle, MultiTaskDatasetBundle, RankingDatasetBundle
from torch_rechub.models.matching import MIND, ComirecDR, ComirecSA, YoutubeDNN
from torch_rechub.models.multi_task import ESMM, MMOE, PLE, SharedBottom
from torch_rechub.models.ranking import DCN, DeepFM, WideDeep


def build_matching_model(model_config: dict[str, Any], data: MatchingDatasetBundle, seq_max_len: int):
    """Create a matching model from a benchmark config."""
    name = model_config["name"]
    params = model_config.get("params", {})
    temperature = float(params.get("temperature", 1.0))

    if name == "YoutubeDNN":
        return YoutubeDNN(
            user_features=data.youtube_user_features,
            item_features=data.item_features,
            neg_item_feature=data.neg_item_feature,
            user_params={"dims": params.get("user_mlp_dims",
                                            [128,
                                             64,
                                             16])},
            temperature=temperature,
        )
    if name == "MIND":
        return MIND(
            user_features=data.user_features,
            history_features=data.history_features,
            item_features=data.item_features,
            neg_item_feature=data.neg_item_feature,
            max_length=seq_max_len,
            temperature=temperature,
            interest_num=int(params.get("interest_num",
                                        4)),
        )
    if name == "ComirecDR":
        return ComirecDR(
            user_features=data.user_features,
            history_features=data.history_features,
            item_features=data.item_features,
            neg_item_feature=data.neg_item_feature,
            max_length=seq_max_len,
            temperature=temperature,
            interest_num=int(params.get("interest_num",
                                        4)),
        )
    if name == "ComirecSA":
        return ComirecSA(
            user_features=data.user_features,
            history_features=data.history_features,
            item_features=data.item_features,
            neg_item_feature=data.neg_item_feature,
            temperature=temperature,
            interest_num=int(params.get("interest_num",
                                        4)),
        )
    raise ValueError(f"Unsupported matching model: {name}")


def count_parameters(model) -> int:
    """Count trainable and frozen model parameters."""
    return sum(parameter.numel() for parameter in model.parameters())


def build_ranking_model(model_config: dict[str, Any], data: RankingDatasetBundle):
    """Create a ranking model from a benchmark config."""
    name = model_config["name"]
    params = model_config.get("params", {})
    mlp_params = params.get("mlp_params", {"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

    if name == "WideDeep":
        return WideDeep(wide_features=data.dense_features, deep_features=data.sparse_features, mlp_params=mlp_params)
    if name == "DeepFM":
        return DeepFM(deep_features=data.dense_features, fm_features=data.sparse_features, mlp_params=mlp_params)
    if name == "DCN":
        return DCN(features=data.dense_features + data.sparse_features, n_cross_layers=int(params.get("n_cross_layers", 3)), mlp_params=mlp_params)
    raise ValueError(f"Unsupported ranking model: {name}")


def build_multitask_model(model_config: dict[str, Any], data: MultiTaskDatasetBundle):
    """Create a multi-task model from a benchmark config."""
    name = model_config["name"]
    params = model_config.get("params", {})
    tower_dims = params.get("tower_dims", [8])
    tower_params_list = [{"dims": list(tower_dims)} for _ in data.task_names]

    if name == "ESMM":
        cvr_dims = params.get("cvr_mlp_dims", [16, 8])
        ctr_dims = params.get("ctr_mlp_dims", [16, 8])
        return ESMM(user_features=data.user_features, item_features=data.item_features, cvr_params={"dims": list(cvr_dims)}, ctr_params={"dims": list(ctr_dims)})
    if name == "MMOE":
        return MMOE(
            features=data.features,
            task_types=data.task_types,
            n_expert=int(params.get("n_expert",
                                    4)),
            expert_params={"dims": list(params.get("expert_dims",
                                                   [16]))},
            tower_params_list=tower_params_list,
        )
    if name == "PLE":
        return PLE(
            features=data.features,
            task_types=data.task_types,
            n_level=int(params.get("n_level",
                                   1)),
            n_expert_specific=int(params.get("n_expert_specific",
                                             2)),
            n_expert_shared=int(params.get("n_expert_shared",
                                           1)),
            expert_params={"dims": list(params.get("expert_dims",
                                                   [16]))},
            tower_params_list=tower_params_list,
        )
    if name == "SharedBottom":
        return SharedBottom(
            features=data.features,
            task_types=data.task_types,
            bottom_params={"dims": list(params.get("bottom_dims",
                                                   [32,
                                                    16]))},
            tower_params_list=tower_params_list,
        )
    raise ValueError(f"Unsupported multi-task model: {name}")
