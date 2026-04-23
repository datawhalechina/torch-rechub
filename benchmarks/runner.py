"""Run Torch-RecHub benchmark configs."""

from __future__ import annotations

import argparse
import random
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

from benchmarks.datasets import build_matching_dataset, build_ranking_dataset
from benchmarks.metrics import evaluate_binary_ranking, evaluate_matching_topk, shape_as_list
from benchmarks.models import build_matching_model, build_ranking_model, count_parameters
from benchmarks.report import write_result
from torch_rechub.trainers import CTRTrainer, MatchTrainer
from torch_rechub.utils.data import DataGenerator, MatchDataGenerator


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML benchmark config."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    """Set common random seeds for reproducible benchmark runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_matching_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Run a phase-1 matching benchmark."""
    dataset_config = config["dataset"]
    model_config = config["model"]
    trainer_config = config["trainer"]
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(trainer_config.get("seed", 2022))
    set_seed(seed)

    data_start = time.perf_counter()
    model_params = model_config.get("params", {})
    dataset_config = dict(dataset_config)
    dataset_config["embed_dim"] = int(model_params.get("embed_dim", 16))
    data = build_matching_dataset(dataset_config)
    data_seconds = time.perf_counter() - data_start

    model = build_matching_model(model_config, data, seq_max_len=int(dataset_config.get("seq_max_len", 50)))
    parameter_count = count_parameters(model)

    train_generator = MatchDataGenerator(x=data.x_train, y=data.y_train)
    train_dl, test_dl, item_dl = train_generator.generate_dataloader(data.x_test, data.all_item, batch_size=int(trainer_config.get("batch_size", 256)), num_workers=int(trainer_config.get("num_workers", 0)))

    trainer = MatchTrainer(
        model,
        mode=int(trainer_config.get("mode",
                                    2)),
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
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = int(trainer_config.get("seed", 2022))
    set_seed(seed)

    data_start = time.perf_counter()
    model_params = model_config.get("params", {})
    dataset_config = dict(dataset_config)
    dataset_config["embed_dim"] = int(model_params.get("embed_dim", 16))
    data = build_ranking_dataset(dataset_config)
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
    metrics = evaluate_binary_ranking(targets, predictions)
    eval_seconds = time.perf_counter() - eval_start

    return {
        "task": config["task"],
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


def run_benchmark(config: dict[str, Any]) -> dict[str, Any]:
    """Dispatch benchmark execution by task."""
    if config.get("task") == "matching":
        return run_matching_benchmark(config)
    if config.get("task") == "ranking":
        return run_ranking_benchmark(config)
    raise ValueError(f"Unsupported benchmark task: {config.get('task')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Torch-RecHub benchmark configs.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    result = run_benchmark(config)
    output_dir = Path(config["output_dir"])
    write_result(output_dir, result)
    print(yaml.safe_dump(result, allow_unicode=True, sort_keys=False))
    print(f"Benchmark result written to: {output_dir}")


if __name__ == "__main__":
    main()
