"""Benchmark result writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_result(output_dir: Path, result: dict[str, Any]) -> None:
    """Write YAML and Markdown benchmark summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "result.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(result, file, allow_unicode=True, sort_keys=False)
    (output_dir / "summary.md").write_text(render_markdown_summary(result), encoding="utf-8")


def render_markdown_summary(result: dict[str, Any]) -> str:
    """Render a compact Markdown benchmark summary."""
    lines = [
        f"# Benchmark: {result['model']['name']} on {result['dataset']['name']}",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for name, value in sorted(result["metrics"].items()):
        lines.append(f"| {name} | {value:.4f} |")

    lines.extend([
        "",
        "## Runtime",
        "",
        "| Stage | Seconds |",
        "| --- | ---: |",
    ])
    for name, value in sorted(result["runtime"].items()):
        lines.append(f"| {name} | {value:.4f} |")

    lines.extend([
        "",
        "## Model Info",
        "",
        f"- Parameters: `{result['model']['parameter_count']}`",
    ])
    if "embeddings" in result:
        lines.extend([
            f"- User embedding shape: `{result['embeddings']['user_shape']}`",
            f"- Item embedding shape: `{result['embeddings']['item_shape']}`",
        ])
    lines.append("")
    return "\n".join(lines)
