"""Baseline comparison helpers for benchmark regression checking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

BASELINES_DIR = Path(__file__).resolve().parent / "baselines"

# Metrics where lower is better: regression = actual > expected + tolerance.
# All other metrics are treated as higher-is-better.
_LOWER_IS_BETTER = {"LogLoss", "MSE"}


def _is_regression(metric: str, delta: float, tolerance: float) -> bool:
    """Return True when the metric has moved in the wrong direction beyond tolerance."""
    if metric in _LOWER_IS_BETTER:
        return delta > tolerance
    return delta < -tolerance


@dataclass
class BaselineDiff:
    """Difference between a produced metric and its baseline entry."""

    config: str
    metric: str
    expected: float
    actual: float
    tolerance: float

    @property
    def delta(self) -> float:
        return self.actual - self.expected

    @property
    def regressed(self) -> bool:
        if math.isnan(self.actual) or math.isnan(self.expected):
            return True
        return _is_regression(self.metric, self.delta, self.tolerance)

    def format_line(self) -> str:
        marker = "FAIL" if self.regressed else "ok"
        return f"[{marker}] {self.config}::{self.metric}  expected={self.expected:.4f} actual={self.actual:.4f} delta={self.delta:+.4f} tol={self.tolerance:.4f}"


def load_baselines(task: str) -> dict[str, Any]:
    """Load the baseline YAML for a given task (``matching``/``ranking``/``multitask``)."""
    path = BASELINES_DIR / f"{task}.yaml"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    return data.get("configs", {}) or {}


def normalize_config_key(config_path: str | Path) -> str:
    """Normalize config paths to the posix-style relative key used in baseline YAML."""
    return Path(config_path).as_posix().replace("\\", "/")


def compare_to_baseline(config_path: str | Path, task: str, metrics: dict[str, float]) -> list[BaselineDiff]:
    """Compare produced metrics against the baseline for ``config_path`` under ``task``.

    Returns a list of ``BaselineDiff`` rows (including passing ones).  An empty list
    means no baseline entry exists for this config.
    """
    baselines = load_baselines(task)
    key = _match_baseline_key(baselines, config_path)
    if key is None:
        return []

    entry = baselines[key].get("metrics", {}) or {}
    diffs: list[BaselineDiff] = []
    for metric_name, spec in entry.items():
        expected = float(spec["expected"])
        tolerance = float(spec.get("tolerance", 0.0))
        if metric_name not in metrics:
            diffs.append(BaselineDiff(config=key, metric=metric_name, expected=expected, actual=float("nan"), tolerance=tolerance))
            continue
        diffs.append(BaselineDiff(config=key, metric=metric_name, expected=expected, actual=float(metrics[metric_name]), tolerance=tolerance))
    return diffs


def _match_baseline_key(baselines: dict[str, Any], config_path: str | Path) -> str | None:
    """Find the baseline dict key that matches the given config path."""
    wanted = normalize_config_key(config_path)
    if wanted in baselines:
        return wanted
    # Allow callers to pass absolute paths; fall back to suffix matching.
    for key in baselines:
        if wanted.endswith(key):
            return key
    return None


def format_diff_report(diffs: list[BaselineDiff]) -> str:
    lines = [diff.format_line() for diff in diffs]
    return "\n".join(lines)


def any_regressed(diffs: list[BaselineDiff]) -> bool:
    return any(diff.regressed for diff in diffs)
