---
title: Benchmark (Experimental)
description: Reproducible model comparison benchmarks for Torch-RecHub
---

# Benchmark (Experimental)

`benchmarks/` provides reproducible model comparisons under a fixed experimental protocol, separate from `examples/` (which focuses on tutorials).

## Quick Start

```bash
# Single config
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml

# Run a group and produce a comparison table
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching
```

Each run writes `result.yaml` (metrics + run metadata), `summary.md`, and `model.pth`. The suite additionally writes `suite.md` — a side-by-side comparison table.

## Supported Tasks

### Matching / Retrieval

Dataset: MovieLens sample. Models: `MIND`, `YoutubeDNN`, `ComirecDR`, `ComirecSA`. Metrics: `Hit@K`, `Recall@K`, `NDCG@K`, `MRR@K`, `Precision@K`.

### Ranking / CTR

Dataset: Criteo sample. Models: `WideDeep`, `DeepFM`, `DCN`. Metrics configurable via `metrics: [AUC, LogLoss, ...]` (available: `AUC`, `LogLoss`, `Accuracy`, `MSE`).

### Multi-Task

Dataset: Census-Income sample, two tasks (income = cvr, marital status = ctr). Models: `ESMM`, `MMOE`, `PLE`. Metrics: `AUC[<task>]` per task + `AUC_mean` (NaN tasks excluded).

## Baseline & Regression Check

```bash
python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_dcn.yaml --check-baseline
python benchmarks/suite.py --configs "benchmarks/configs/ranking/*.yaml" --output benchmark_results/suites/ranking --check-baseline
```

Baselines live in `benchmarks/baselines/<task>.yaml`. Direction-aware: higher-is-better metrics (AUC, Hit, …) only fail when they drop; lower-is-better metrics (LogLoss, MSE) only fail when they rise.
