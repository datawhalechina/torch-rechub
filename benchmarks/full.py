"""Compare recommendation models on a full dataset described in benchmarks/configs/full.yaml."""

import argparse
import importlib
import math
from pathlib import Path

import yaml

CONFIG = Path(__file__).parent / "configs" / "full.yaml"
# yapf: disable
SUPPORTED_MODELS = {
    "movielens": ("YoutubeDNN", "MIND", "ComirecDR", "ComirecSA", "DSSM", "GRU4Rec", "NARM", "STAMP", "SASRec", "SINE"),
    "criteo": ("WideDeep", "DeepFM", "DCN"),
    "avazu": ("WideDeep", "DeepFM", "DCN"),
    "amazon": ("DIN",),
    "census": ("SharedBottom", "ESMM", "MMOE", "PLE", "PLE-2", "AITM"),
    "ali_ccp": ("DeepFM", "MMOE"),
    "aliexpress": ("SharedBottom", "AITM"),
}
# yapf: enable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SUPPORTED_MODELS)
    parser.add_argument("--config", type=Path, default=CONFIG, help="YAML file with data paths and model lists.")
    parser.add_argument("--data", type=Path, help="Override the selected dataset's path.")
    parser.add_argument("--models", nargs="+", help="Run only the named models, in this order.")
    parser.add_argument("--epochs", type=int, help="Train for this many epochs and restore the best validation checkpoint.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, help="Directory for this comparison and its checkpoints.")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)[args.dataset]
    data = (args.data or Path(config["path"])).resolve()
    if not data.exists():
        parser.error(f"Data not found: {data}. Set the path in {args.config} or pass --data.")
    models = args.models if args.models is not None else config["models"]
    if not isinstance(models, list) or not models:
        parser.error("Select at least one model in the models list.")
    supported = SUPPORTED_MODELS[args.dataset]
    for name in models:
        if name not in supported:
            parser.error(f"Unknown {args.dataset} model {name!r}. Choose from {', '.join(supported)}.")
    if len(set(models)) != len(models):
        parser.error("Each model should appear once in a comparison.")

    epochs = args.epochs if args.epochs is not None else config.get("epochs")
    if epochs is not None and (type(epochs) is not int or epochs < 1):
        parser.error("epochs must be a positive integer")

    output = args.output or Path("benchmark_results") / "full" / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    try:
        experiment = importlib.import_module(f"benchmarks.full_runs.{args.dataset}")
    except ModuleNotFoundError as exc:
        if exc.name == "pyarrow":
            parser.error(f"{args.dataset} requires PyArrow. Install it with uv sync --extra benchmark --extra bigdata.")
        raise
    from benchmarks.runner import collect_run_metadata

    metadata = collect_run_metadata()
    metrics = experiment.run(data, output.resolve(), models, args.device, epochs=epochs)
    if epochs is not None:
        metadata["epochs"] = epochs
    result = {"dataset": args.dataset, "path": str(data), "device": args.device, "run": metadata, "metrics": metrics}
    _write_report(output, result)


def _write_report(output, result):
    """Save raw test scores and a Markdown table with one row per model."""
    run = result["run"]
    scores = result["metrics"]
    columns = list(dict.fromkeys(metric for row in scores.values() for metric in row))
    lines = [
        f"# {result['dataset']} benchmark",
        "",
        f"- Data: `{result['path']}`",
        f"- Device: `{result['device']}`",
    ]
    if "epochs" in run:
        lines.append(f"- Epochs: {run['epochs']}")
    lines.extend([
        f"- Git: `{run['git_commit']}`",
        f"- Started (UTC): {run['timestamp']}",
        "",
        "| Model | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ])
    for name, row in scores.items():
        values = []
        for metric in columns:
            value = row.get(metric, math.nan)
            values.append(f"{value:.6f}" if math.isfinite(value) else "—")
        lines.append("| " + " | ".join([name, *values]) + " |")

    (output / "suite.yaml").write_text(yaml.safe_dump(result, allow_unicode=True, sort_keys=False), encoding="utf-8")
    report = output / "suite.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Test results: {report.resolve()}")


if __name__ == "__main__":
    main()
