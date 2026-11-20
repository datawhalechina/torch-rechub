---
title: Model benchmarks (experimental)
description: Compare recommendation models with reproducible benchmark configurations
---

# Model benchmarks (experimental)

To compare MIND and YoutubeDNN on the same users, run their configurations in `benchmarks/`. Each configuration records the data file, random seed, training epochs and metrics, so you can see what changed between runs.

Start in the repository root. The benchmark scripts read YAML, and the retrieval examples below use Annoy to look up candidates. Install both extras for these examples. The `benchmark` extra is enough for ranking or multi-task experiments.

```bash
uv sync --extra benchmark --extra annoy
```

## Run a benchmark

Activate the `.venv` you just created and run the commands from the repository root. The first runs MIND alone. The second runs each retrieval configuration in turn.

```bash
# Run a single configuration.
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml

# Run a whole suite and write a comparison table.
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching
```

Each run writes its results to the configured `output_dir`. Open `summary.md` to read the metrics or use `result.yaml` for programmatic access. The YAML also records the Git commit, Python and PyTorch versions, and timings. Retrieval and ranking runs save weights in `model.pth`. The multi-task example writes `model_base_0.pth`.

A suite also writes `suite.yaml` and `suite.md` to the directory passed to `--output`. The table below illustrates the format using the bundled sample data. When comparing models on a full dataset, check that their configurations use the same split.

| model | dataset | params | Hit@10 | NDCG@10 | train_s |
| --- | --- | --- | --- | --- | --- |
| MIND | ml-1m-sample | 3792 | 0.5000 | 0.5000 | 0.026 |
| YoutubeDNN | ml-1m-sample | 23872 | 0.0000 | 0.0000 | 0.021 |
| ComirecDR | ml-1m-sample | 54736 | 0.0000 | 0.0000 | 0.023 |
| ComirecSA | ml-1m-sample | 4816 | 0.0000 | 0.0000 | 0.018 |

## Available tasks

### Retrieval

MovieLens sample at `examples/matching/data/ml-1m/ml-1m_sample.csv`.

| Configuration | Model |
| --- | --- |
| `configs/matching/ml_1m_mind.yaml` | MIND |
| `configs/matching/ml_1m_youtube_dnn.yaml` | YoutubeDNN |
| `configs/matching/ml_1m_comirec_dr.yaml` | ComirecDR |
| `configs/matching/ml_1m_comirec_sa.yaml` | ComirecSA |

Metrics include `Hit@K`, `Recall@K`, `NDCG@K`, `MRR@K` and `Precision@K`.

### Ranking and click-through rate prediction

Criteo sample at `examples/ranking/data/criteo/criteo_sample.csv`.

| Configuration | Model |
| --- | --- |
| `configs/ranking/criteo_widedeep.yaml` | WideDeep |
| `configs/ranking/criteo_deepfm.yaml` | DeepFM |
| `configs/ranking/criteo_dcn.yaml` | DCN |

The example reports `AUC` and `LogLoss`. The YAML `metrics` field accepts `AUC`, `LogLoss`, `Accuracy` and `MSE`.

### Multi-task learning

Census-Income sample at `examples/ranking/data/census-income/`. The example uses income as the CVR label and marital status as the CTR label.

| Configuration | Model |
| --- | --- |
| `configs/multitask/census_esmm.yaml` | ESMM |
| `configs/multitask/census_mmoe.yaml` | MMOE |
| `configs/multitask/census_ple.yaml` | PLE |

The report includes `AUC[<task>]` for each task and `AUC_mean` across tasks with a defined AUC. Tasks with NaN AUC are excluded from the mean.

## Configuration format

```yaml
task: matching          # matching | ranking | multitask

dataset:
  name: ml-1m-sample
  path: examples/matching/data/ml-1m/ml-1m_sample.csv
  seq_max_len: 50
  neg_ratio: 3

model:
  name: MIND
  params:
    embed_dim: 16
    interest_num: 4
    temperature: 0.02

trainer:
  mode: 2
  epochs: 1
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.000001
  device: cpu
  seed: 2022

metrics:
  topk: 10

output_dir: benchmark_results/matching/ml_1m_mind
```

The runner checks configuration keys before training. For example, if you type `batchsize` instead of `batch_size`, the error identifies the unknown field so you can correct it.

## Check a run against its baseline

After changing a model or trainer, add `--check-baseline` to check for regressions. The script reads the reference value (`expected`) and allowed deviation (`tolerance`) from `benchmarks/baselines/<task>.yaml`. It exits with a nonzero status if performance deteriorates beyond that tolerance.

```bash
python benchmarks/runner.py \
  --config benchmarks/configs/ranking/criteo_dcn.yaml \
  --check-baseline

python benchmarks/suite.py \
  --configs "benchmarks/configs/ranking/*.yaml" \
  --output  benchmark_results/suites/ranking \
  --check-baseline
```

For example, an expected AUC of 0.75 with a tolerance of 0.01 fails only below 0.74. Lower values are better for LogLoss and MSE, so those metrics fail only above `expected + tolerance`.

To update a baseline, rerun the suite with the same configuration. Review its metrics in `benchmark_results/suites/<task>/suite.yaml` before copying them into the corresponding baseline file.
