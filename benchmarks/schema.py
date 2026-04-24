"""Lightweight schema validation for benchmark configs.

Uses a small whitelist of known keys per section rather than pulling in pydantic.
The goal is to catch typos like `batchsize` that would otherwise fall through
to a default and silently skew benchmark numbers.
"""

from __future__ import annotations

from typing import Iterable

ALLOWED_TASKS = {"matching", "ranking", "multitask"}

# Known keys per section.  Anything else raises ValueError during validation.
MATCHING_DATASET_KEYS = {"name", "path", "seq_max_len", "neg_ratio", "sample_method", "padding", "truncating"}
RANKING_DATASET_KEYS = {"name", "path"}
MULTITASK_DATASET_KEYS = {"name", "path"}
MODEL_KEYS = {"name", "params"}
TRAINER_KEYS = {"mode", "epochs", "batch_size", "learning_rate", "weight_decay", "earlystop_patience", "split_ratio", "device", "seed", "gpus", "num_workers"}
METRICS_KEYS = {"topk"}
TOP_LEVEL_KEYS = {"task", "dataset", "model", "trainer", "metrics", "output_dir"}

_DATASET_KEYS_BY_TASK = {
    "matching": MATCHING_DATASET_KEYS,
    "ranking": RANKING_DATASET_KEYS,
    "multitask": MULTITASK_DATASET_KEYS,
}


def validate_config(config: dict) -> None:
    """Raise ValueError if the config has missing or misspelled keys."""
    if not isinstance(config, dict):
        raise ValueError("benchmark config must be a mapping")
    _check_unknown(config, TOP_LEVEL_KEYS, "top-level")
    _check_required(config, {"task", "dataset", "model", "trainer", "output_dir"}, "top-level")

    task = config.get("task")
    if task not in ALLOWED_TASKS:
        raise ValueError(f"task must be one of {sorted(ALLOWED_TASKS)}, got {task!r}")

    dataset = config["dataset"]
    _check_required(dataset, {"name", "path"}, "dataset")
    _check_unknown(dataset, _DATASET_KEYS_BY_TASK[task], "dataset")

    model = config["model"]
    _check_required(model, {"name"}, "model")
    _check_unknown(model, MODEL_KEYS, "model")

    trainer = config["trainer"]
    _check_unknown(trainer, TRAINER_KEYS, "trainer")

    metrics = config.get("metrics")
    if task == "matching" and isinstance(metrics, dict):
        _check_unknown(metrics, METRICS_KEYS, "metrics")
    if task == "ranking" and metrics is not None:
        # Ranking metrics is a list of metric names; validate against the registry.
        from benchmarks.metrics import BINARY_RANKING_METRICS
        if not isinstance(metrics, list):
            raise ValueError("ranking 'metrics' must be a list of metric names")
        unknown = sorted(set(metrics) - set(BINARY_RANKING_METRICS))
        if unknown:
            raise ValueError(f"metrics has unknown entries {unknown}; available: {sorted(BINARY_RANKING_METRICS)}")


def _check_required(section: dict, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(section.keys()))
    if missing:
        raise ValueError(f"{label} config missing required keys: {missing}")


def _check_unknown(section: dict, allowed: Iterable[str], label: str) -> None:
    unknown = sorted(set(section.keys()) - set(allowed))
    if unknown:
        raise ValueError(f"{label} config has unknown keys {unknown} (allowed: {sorted(allowed)})")
