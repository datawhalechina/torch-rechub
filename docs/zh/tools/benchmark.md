---
title: Benchmark（实验性功能）
description: Torch-RecHub 可复现模型对比 benchmark
---

# Benchmark（实验性功能）

如果想比较 MIND 和 YoutubeDNN 在同一批用户上的召回效果，可以用 `benchmarks/` 下的配置运行两个模型。每个配置都写明了数据文件、随机种子、训练轮数和指标，便于确认两次实验改了什么。

先进入仓库根目录。基准脚本用 YAML 读取配置，下面的召回实验还要用 Annoy 查候选，因此一起安装这两组依赖。只运行排序或多任务实验时，选 `benchmark` 即可。

```bash
uv sync --extra benchmark --extra annoy
```

## 快速运行

激活刚才创建的 `.venv` 环境，从仓库根目录运行下面的命令。第一条只跑 MIND，第二条依次跑完所有召回配置。

```bash
# 单个 config
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml

# 一次跑一组并输出对比表
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching
```

单次运行把结果写到配置里的 `output_dir`。直接打开 `summary.md` 就能看指标，要用脚本处理数据则读 `result.yaml`，里面还记录了代码提交号、Python 和 PyTorch 版本、运行时间。召回和排序权重保存在 `model.pth`，多任务示例保存在 `model_base_0.pth`。

批量运行还会在 `--output` 指定的目录生成 `suite.yaml` 和 `suite.md`。下面是自带小样本的结果格式。在全量数据上比较模型时，先确认各配置的数据划分一致。

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

运行时会先检查字段名。例如把 `batch_size` 拼成 `batchsize`，脚本会指出这个字段有误，改好后再开始训练。

## 基线与回归检测

改完模型或训练器后，想知道指标有没有退步，可以加 `--check-baseline`。脚本会从 `benchmarks/baselines/<task>.yaml` 读取参考值 `expected` 和允许的偏差 `tolerance`，超过允许的退步幅度就以非零状态退出。

```bash
python benchmarks/runner.py \
  --config benchmarks/configs/ranking/criteo_dcn.yaml \
  --check-baseline

python benchmarks/suite.py \
  --configs "benchmarks/configs/ranking/*.yaml" \
  --output  benchmark_results/suites/ranking \
  --check-baseline
```

例如 AUC 的参考值为 0.75，允许偏差为 0.01，低于 0.74 才判为退步。LogLoss 和 MSE 则是越低越好，高于 `expected + tolerance` 才会失败。

调整参照结果时，先按相同配置重新运行整组实验，再将 `benchmark_results/suites/<task>/suite.yaml` 中确认过的指标写入对应基线文件。
