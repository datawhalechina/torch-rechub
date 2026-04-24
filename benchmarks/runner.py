"""Run Torch-RecHub benchmark configs."""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.baselines import any_regressed, compare_to_baseline, format_diff_report
from benchmarks.datasets import build_matching_dataset, build_multitask_dataset, build_ranking_dataset
from benchmarks.metrics import evaluate_binary_ranking, evaluate_matching_topk, evaluate_multitask_scores, shape_as_list
from benchmarks.models import build_matching_model, build_multitask_model, build_ranking_model, count_parameters
from benchmarks.report import write_result
from benchmarks.schema import validate_config
from torch_rechub.trainers import CTRTrainer, MatchTrainer, MTLTrainer
from torch_rechub.utils.data import DataGenerator, MatchDataGenerator


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML benchmark config verbatim (paths are resolved lazily at use-time)."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_project_path(value: str) -> Path:
    """Resolve a config-relative path against PROJECT_ROOT when it is not absolute."""
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def set_seed(seed: int) -> None:
    """Set common random seeds and enforce deterministic algorithms for reproducible benchmark runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_run_metadata() -> dict[str, Any]:
    """Capture git commit, timestamp, and runtime versions for reproducibility."""
    return {
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _read_git_commit(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "platform": platform.platform(),
    }


def _read_git_commit() -> str | None:
    """Return the current git commit SHA, or None if unavailable."""
    try:
        output = subprocess.check_output(
            ["git",
             "rev-parse",
             "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return output.decode("utf-8", errors="ignore").strip() or None


def run_matching_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run a phase-1 matching benchmark."""
    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(trainer_config.get("seed", 2022))
    set_seed(seed)

    model_params = model_config.get("params", {})
    embed_dim = int(model_params.get("embed_dim", 16))
    mode = int(trainer_config.get("mode", 2))

    dataset_runtime_config = dict(dataset_config)
    dataset_runtime_config["path"] = str(resolve_project_path(dataset_config["path"]))
    data_start = time.perf_counter()
    data = build_matching_dataset(dataset_runtime_config, embed_dim=embed_dim, mode=mode)
    data_seconds = time.perf_counter() - data_start

    model = build_matching_model(model_config, data, seq_max_len=int(dataset_runtime_config.get("seq_max_len", 50)))
    parameter_count = count_parameters(model)

    train_generator = MatchDataGenerator(x=data.x_train, y=data.y_train)
    train_dl, test_dl, item_dl = train_generator.generate_dataloader(data.x_test, data.all_item, batch_size=int(trainer_config.get("batch_size", 256)), num_workers=int(trainer_config.get("num_workers", 0)))

    trainer = MatchTrainer(
        model,
        mode=mode,
        optimizer_params={
            "lr": float(trainer_config.get("learning_rate",
                                           1e-3)),
            "weight_decay": float(trainer_config.get("weight_decay",
                                                     1e-6)),
        },
        n_epoch=int(trainer_config.get("epochs",
                                       1)),
        device=trainer_config.get("device",
                                  "cpu"),
        model_path=str(output_dir),
        gpus=trainer_config.get("gpus",
                                []),
    )

    train_start = time.perf_counter()
    trainer.fit(train_dl)
    train_seconds = time.perf_counter() - train_start

    infer_start = time.perf_counter()
    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path=str(output_dir))
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path=str(output_dir))
    infer_seconds = time.perf_counter() - infer_start

    eval_start = time.perf_counter()
    metrics = evaluate_matching_topk(data, user_embedding.detach().cpu().numpy(), item_embedding.detach().cpu().numpy(), topk=int(config.get("metrics", {}).get("topk", 10)))
    eval_seconds = time.perf_counter() - eval_start

    return {
        "task": config["task"],
        "run": collect_run_metadata(),
        "dataset": dataset_config,
        "model": {
            "name": model_config["name"],
            "params": model_params,
            "parameter_count": parameter_count,
        },
        "trainer": trainer_config,
        "metrics": metrics,
        "runtime": {
            "data_seconds": data_seconds,
            "train_seconds": train_seconds,
            "infer_seconds": infer_seconds,
            "eval_seconds": eval_seconds,
        },
        "embeddings": {
            "user_shape": shape_as_list(user_embedding),
            "item_shape": shape_as_list(item_embedding),
        },
    }


def run_ranking_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run a phase-2 ranking benchmark."""
    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(trainer_config.get("seed", 2022))
    set_seed(seed)

    model_params = model_config.get("params", {})
    embed_dim = int(model_params.get("embed_dim", 16))

    dataset_runtime_config = dict(dataset_config)
    dataset_runtime_config["path"] = str(resolve_project_path(dataset_config["path"]))
    data_start = time.perf_counter()
    data = build_ranking_dataset(dataset_runtime_config, embed_dim=embed_dim)
    data_seconds = time.perf_counter() - data_start

    data_generator = DataGenerator(data.x, data.y)
    train_dl, val_dl, test_dl = data_generator.generate_dataloader(split_ratio=trainer_config.get("split_ratio", [0.7, 0.1]), batch_size=int(trainer_config.get("batch_size", 256)), num_workers=int(trainer_config.get("num_workers", 0)))

    model = build_ranking_model(model_config, data)
    parameter_count = count_parameters(model)
    trainer = CTRTrainer(
        model,
        optimizer_params={
            "lr": float(trainer_config.get("learning_rate",
                                           1e-3)),
            "weight_decay": float(trainer_config.get("weight_decay",
                                                     1e-6)),
        },
        n_epoch=int(trainer_config.get("epochs",
                                       1)),
        earlystop_patience=int(trainer_config.get("earlystop_patience",
                                                  10)),
        device=trainer_config.get("device",
                                  "cpu"),
        model_path=str(output_dir),
        gpus=trainer_config.get("gpus",
                                []),
    )

    train_start = time.perf_counter()
    trainer.fit(train_dl, val_dl)
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    predictions = trainer.predict(trainer.model, test_dl)
    targets = _collect_targets(test_dl)
    metrics = evaluate_binary_ranking(targets, predictions, metric_names=config.get("metrics"))
    eval_seconds = time.perf_counter() - eval_start

    return {
        "task": config["task"],
        "run": collect_run_metadata(),
        "dataset": dataset_config,
        "model": {
            "name": model_config["name"],
            "params": model_params,
            "parameter_count": parameter_count,
        },
        "trainer": trainer_config,
        "metrics": metrics,
        "runtime": {
            "data_seconds": data_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        },
    }


def _collect_targets(data_loader) -> list[float]:
    targets = []
    for _, y in data_loader:
        targets.extend(np.asarray(y).reshape(-1).astype(float).tolist())
    return targets


def run_multitask_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run a phase-3 multi-task benchmark."""
    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    output_dir = resolve_project_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(trainer_config.get("seed", 2022))
    set_seed(seed)

    model_params = model_config.get("params", {})
    embed_dim = int(model_params.get("embed_dim", 16))

    dataset_runtime_config = dict(dataset_config)
    dataset_runtime_config["path"] = str(resolve_project_path(dataset_config["path"]))
    data_start = time.perf_counter()
    data = build_multitask_dataset(dataset_runtime_config, embed_dim=embed_dim, model_name=model_config["name"])
    data_seconds = time.perf_counter() - data_start

    data_generator = DataGenerator(data.x_train, data.y_train)
    train_dl, val_dl, test_dl = data_generator.generate_dataloader(
        x_val=data.x_val,
        y_val=data.y_val,
        x_test=data.x_test,
        y_test=data.y_test,
        batch_size=int(trainer_config.get("batch_size", 256)),
        num_workers=int(trainer_config.get("num_workers", 0)),
    )

    model = build_multitask_model(model_config, data)
    parameter_count = count_parameters(model)
    trainer = MTLTrainer(
        model,
        task_types=data.task_types,
        optimizer_params={
            "lr": float(trainer_config.get("learning_rate",
                                           1e-3)),
            "weight_decay": float(trainer_config.get("weight_decay",
                                                     1e-4)),
        },
        n_epoch=int(trainer_config.get("epochs",
                                       1)),
        earlystop_patience=int(trainer_config.get("earlystop_patience",
                                                  10)),
        device=trainer_config.get("device",
                                  "cpu"),
        model_path=str(output_dir),
        gpus=trainer_config.get("gpus",
                                []),
    )

    train_start = time.perf_counter()
    trainer.fit(train_dl, val_dl)
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    scores = trainer.evaluate(trainer.model, test_dl)
    metrics = evaluate_multitask_scores(scores, data.task_names, data.task_types)
    eval_seconds = time.perf_counter() - eval_start

    return {
        "task": config["task"],
        "run": collect_run_metadata(),
        "dataset": dataset_config,
        "model": {
            "name": model_config["name"],
            "params": model_params,
            "parameter_count": parameter_count,
        },
        "trainer": trainer_config,
        "metrics": metrics,
        "runtime": {
            "data_seconds": data_seconds,
            "train_seconds": train_seconds,
            "eval_seconds": eval_seconds,
        },
        "tasks": {
            "names": data.task_names,
            "types": data.task_types,
        },
    }


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch benchmark execution by task."""
    validate_config(config)
    if config.get("task") == "matching":
        return run_matching_benchmark(config)
    if config.get("task") == "ranking":
        return run_ranking_benchmark(config)
    if config.get("task") == "multitask":
        return run_multitask_benchmark(config)
    raise ValueError(f"Unsupported benchmark task: {config.get('task')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Torch-RecHub benchmark configs.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    parser.add_argument(
        "--check-baseline",
        action="store_true",
        help="Compare the run's metrics against benchmarks/baselines/<task>.yaml and exit non-zero on regression.",
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    result = run_benchmark(config)
    output_dir = resolve_project_path(config["output_dir"])
    write_result(output_dir, result)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    print(f"Benchmark result written to: {output_dir}")

    if args.check_baseline:
        diffs = compare_to_baseline(args.config, config["task"], result["metrics"])
        if not diffs:
            print(f"[baseline] FAIL: no entry for {args.config} in benchmarks/baselines/{config['task']}.yaml")
            raise SystemExit(1)
        print("[baseline] comparison:")
        print(format_diff_report(diffs))
        if any_regressed(diffs):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
