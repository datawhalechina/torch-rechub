---
title: Benchmark（实验性功能）
description: Torch-RecHub 可复现模型对比 benchmark
---

# Benchmark（实验性功能）

`benchmarks/` 提供固定实验协议下的可复现模型对比，与 `examples/` 分开维护：`examples/` 侧重教学，`benchmarks/` 侧重横向比较。

## 快速运行

在仓库根目录执行：

```bash
# 单个 config
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml

# 一次跑一组并输出对比表
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching
```

每次运行写出：

- `result.yaml`：结构化指标 + 运行元数据（git commit、Python/Torch 版本、时间戳）
- `summary.md`：人类可读摘要
- `model.pth`：模型权重

Suite 额外写出 `suite.md` 对比表，例如：

| model | dataset | params | Hit@10 | NDCG@10 | train_s |
| --- | --- | --- | --- | --- | --- |
| MIND | ml-1m-sample | 3792 | 0.5000 | 0.5000 | 0.026 |
| YoutubeDNN | ml-1m-sample | 23872 | 0.0000 | 0.0000 | 0.021 |
| ComirecDR | ml-1m-sample | 54736 | 0.0000 | 0.0000 | 0.023 |
| ComirecSA | ml-1m-sample | 4816 | 0.0000 | 0.0000 | 0.018 |

## 支持的任务

### Matching（召回）

数据集：MovieLens sample（`examples/matching/data/ml-1m/ml-1m_sample.csv`）

| Config | 模型 |
| --- | --- |
| `configs/matching/ml_1m_mind.yaml` | MIND |
| `configs/matching/ml_1m_youtube_dnn.yaml` | YoutubeDNN |
| `configs/matching/ml_1m_comirec_dr.yaml` | ComirecDR |
| `configs/matching/ml_1m_comirec_sa.yaml` | ComirecSA |

指标：`Hit@K`、`Recall@K`、`NDCG@K`、`MRR@K`、`Precision@K`

### Ranking（排序 / CTR）

数据集：Criteo sample（`examples/ranking/data/criteo/criteo_sample.csv`）

| Config | 模型 |
| --- | --- |
| `configs/ranking/criteo_widedeep.yaml` | WideDeep |
| `configs/ranking/criteo_deepfm.yaml` | DeepFM |
| `configs/ranking/criteo_dcn.yaml` | DCN |

指标：`AUC`、`LogLoss`（可在 YAML 的 `metrics:` 字段自定义，支持 `AUC`、`LogLoss`、`Accuracy`、`MSE`）

### Multi-Task（多任务）

数据集：Census-Income sample（`examples/ranking/data/census-income/`），双任务：income（cvr）和 marital status（ctr）

| Config | 模型 |
| --- | --- |
| `configs/multitask/census_esmm.yaml` | ESMM |
| `configs/multitask/census_mmoe.yaml` | MMOE |
| `configs/multitask/census_ple.yaml` | PLE |

指标：`AUC[<task>]`（每任务独立）+ `AUC_mean`（NaN 任务自动排除）

## Config 格式

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

加载时会做 schema 校验：拼错的 key（如 `batchsize`）直接报错，不会静默 fallback 到默认值。

## 基线与回归检测

`benchmarks/baselines/<task>.yaml` 记录每个 config 的 `expected + tolerance`。加 `--check-baseline` 后，指标低于预期超出容忍范围时非零退出：

```bash
python benchmarks/runner.py \
  --config benchmarks/configs/ranking/criteo_dcn.yaml \
  --check-baseline

python benchmarks/suite.py \
  --configs "benchmarks/configs/ranking/*.yaml" \
  --output  benchmark_results/suites/ranking \
  --check-baseline
```

**更新基线**：跑一次 suite → 把 `benchmark_results/suites/<task>/suite.yaml` 里的指标复制到 `benchmarks/baselines/<task>.yaml`，设置合适的 `tolerance`。

回归方向语义：`AUC`、`Hit`、`Recall` 等高指标只在**低于** `expected - tolerance` 时失败；`LogLoss`、`MSE` 等低指标只在**高于** `expected + tolerance` 时失败。
