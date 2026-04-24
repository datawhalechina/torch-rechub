"""Run a set of benchmark configs and emit a side-by-side comparison report.

Usage:
    python benchmarks/suite.py --configs benchmarks/configs/matching/*.yaml --output benchmark_results/suites/matching
    python benchmarks/suite.py --configs benchmarks/configs/ranking/*.yaml  --output benchmark_results/suites/ranking

The suite does not re-implement per-benchmark logic: it simply calls
`run_benchmark` for each config, writes each run's artifacts via `write_result`,
and then produces a top-level `suite.yaml` and `suite.md` table so multiple
models can be compared at a glance.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.baselines import BaselineDiff, any_regressed, compare_to_baseline, format_diff_report
from benchmarks.report import write_result
from benchmarks.runner import load_config, resolve_project_path, run_benchmark


def _missing_baseline_sentinel(config: str) -> BaselineDiff:
    """Return a synthetic failing diff for a config that has no baseline entry."""
    return BaselineDiff(config=config, metric="<no baseline>", expected=float("nan"), actual=float("nan"), tolerance=0.0)


def run_suite(config_paths: list[Path], output_dir: Path, check_baseline: bool = False) -> dict[str, Any]:
    """Execute every config and return an aggregated suite result.

    When ``check_baseline`` is true, each run's metrics are compared against
    ``benchmarks/baselines/<task>.yaml`` and a flat list of ``BaselineDiff`` rows
    is attached to the returned suite under ``baseline_diffs``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    all_diffs = []
    for config_path in config_paths:
        config = load_config(config_path)
        result = run_benchmark(config)
        per_run_dir = resolve_project_path(config["output_dir"])
        write_result(per_run_dir, result)
        runs.append(
            {
                "config": str(config_path),
                "task": result["task"],
                "dataset": result["dataset"]["name"],
                "model": result["model"]["name"],
                "parameter_count": result["model"]["parameter_count"],
                "metrics": dict(result["metrics"]),
                "runtime": dict(result["runtime"]),
            }
        )
        if check_baseline:
            diffs = compare_to_baseline(config_path, result["task"], result["metrics"])
            if not diffs:
                all_diffs.append(_missing_baseline_sentinel(str(config_path)))
            else:
                all_diffs.extend(diffs)

    suite: dict[str, Any] = {"runs": runs}
    if check_baseline:
        suite["baseline_diffs"] = [{
            "config": diff.config,
            "metric": diff.metric,
            "expected": diff.expected,
            "actual": diff.actual,
            "tolerance": diff.tolerance,
            "regressed": diff.regressed,
        } for diff in all_diffs]
    (output_dir / "suite.yaml").write_text(yaml.safe_dump(suite, allow_unicode=True, sort_keys=False), encoding="utf-8")
    (output_dir / "suite.md").write_text(render_suite_markdown(runs), encoding="utf-8")
    if check_baseline:
        (output_dir / "baseline.md").write_text(_render_baseline_markdown(all_diffs), encoding="utf-8")
    suite["_diffs"] = all_diffs
    return suite


def _render_baseline_markdown(diffs) -> str:
    """Render baseline diffs as a simple Markdown table."""
    if not diffs:
        return "# Baseline Check\n\nNo baselines matched any of the run configs.\n"
    lines = ["# Baseline Check", "", "| status | config | metric | expected | actual | delta | tolerance |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for diff in diffs:
        status = "FAIL" if diff.regressed else "ok"
        lines.append(f"| {status} | {diff.config} | {diff.metric} | {diff.expected:.4f} | {diff.actual:.4f} | {diff.delta:+.4f} | {diff.tolerance:.4f} |")
    lines.append("")
    return "\n".join(lines)


def render_suite_markdown(runs: list[dict[str, Any]]) -> str:
    """Render a compact side-by-side Markdown table across all runs."""
    if not runs:
        return "# Suite\n\nNo runs.\n"

    metric_keys: list[str] = []
    for run in runs:
        for key in run["metrics"]:
            if key not in metric_keys:
                metric_keys.append(key)

    header = ["model", "dataset", "params", *metric_keys, "train_s"]
    lines = [
        "# Benchmark Suite",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for run in runs:
        row = [
            run["model"],
            run["dataset"],
            f"{run['parameter_count']}",
            *[f"{run['metrics'].get(key, float('nan')):.4f}" if key in run["metrics"] else "-" for key in metric_keys],
            f"{run['runtime'].get('train_seconds', float('nan')):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def _expand_globs(patterns: list[str]) -> list[Path]:
    """Expand shell-style globs and deduplicate while keeping order."""
    seen: set[str] = set()
    ordered: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern)) or [pattern]
        for match in matches:
            if match in seen:
                continue
            seen.add(match)
            ordered.append(Path(match))
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark suite across multiple configs.")
    parser.add_argument("--configs", nargs="+", required=True, help="YAML config paths (supports globs).")
    parser.add_argument("--output", required=True, help="Directory where suite.yaml and suite.md are written.")
    parser.add_argument("--check-baseline", action="store_true", help="Compare each run against benchmarks/baselines/<task>.yaml and exit non-zero on regression.")
    args = parser.parse_args()

    config_paths = _expand_globs(args.configs)
    if not config_paths:
        raise SystemExit("No configs matched the given patterns.")

    output_dir = resolve_project_path(args.output)
    suite = run_suite(config_paths, output_dir, check_baseline=args.check_baseline)
    print(f"Suite ran {len(suite['runs'])} configs; report at {output_dir}")

    if args.check_baseline:
        diffs = suite["_diffs"]
        if not diffs:
            print("[baseline] no runs matched any baseline entry")
            return
        print("[baseline] comparison:")
        print(format_diff_report(diffs))
        if any_regressed(diffs):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
