# Torch-RecHub Benchmark 设计

本文档描述 Torch-RecHub 的 benchmark 设计方案。目标是在固定数据预处理、切分策略、训练预算、评价指标和结果格式的前提下，让不同模型的实验结果可以复现、可以横向比较。

## 目标

- 在相同数据集、切分方式、负采样设置、评价指标和训练预算下比较模型。
- 将 benchmark 与 `examples/` 分离：`examples/` 侧重教学，benchmark 侧重可复现比较。
- 每次运行同时保存机器可读结果 `result.yaml` 和人工可读摘要 `summary.md`。
- 先从轻量级 MovieLens 召回 benchmark 做起，再扩展到排序和多任务。

## 非目标

- 当前阶段不是排行榜。第一阶段只验证 benchmark 基础设施和小样本可复现流程。
- 不替代 `examples/` 中的教程脚本。
- 默认不在 CI 中运行大规模完整 benchmark，避免拖慢贡献流程。

## 目录结构

```text
benchmarks/
├── BENCHMARK_DESIGN.md
├── README.md
├── configs/
│   ├── matching/
│   │   ├── ml_1m_mind.yaml
│   │   ├── ml_1m_youtube_dnn.yaml
│   │   └── ml_1m_comirec_dr.yaml
│   └── ranking/
│       ├── criteo_widedeep.yaml
│       ├── criteo_deepfm.yaml
│       └── criteo_dcn.yaml
├── datasets.py
├── metrics.py
├── models.py
├── report.py
└── runner.py
```

## 第一阶段：Matching Benchmark

第一阶段支持 MovieLens sample 数据上的 list-wise 召回模型 benchmark。

支持模型：

- `MIND`
- `YoutubeDNN`
- `ComirecDR`
- `ComirecSA`

数据集：

- `examples/matching/data/ml-1m/ml-1m_sample.csv`
- 通过 `generate_seq_feature_match(..., mode=2)` 构造 leave-one-out 风格的序列召回数据
- 每条训练样本包含 1 个正样本物品和若干负样本物品

指标：

- `Hit@K`
- `Recall@K`
- `NDCG@K`
- `MRR@K`
- `Precision@K`

记录信息：

- 模型名称和模型参数
- 数据路径和预处理参数
- 训练超参数
- 随机种子和设备
- 模型参数量
- 数据处理、训练、推理、评估耗时
- 用户向量和物品向量形状

## 第二阶段：Ranking Benchmark

第二阶段支持 Criteo sample 数据上的 CTR/ranking benchmark，使用固定随机种子和固定 `split_ratio` 生成训练集、验证集、测试集，并记录：

- `AUC`
- `LogLoss`
- 数据处理、训练、评估耗时
- 参数量

支持模型：

- `WideDeep`
- `DeepFM`
- `DCN`

数据集：

- `examples/ranking/data/criteo/criteo_sample.csv`
- 稠密特征 `I*` 使用 `MinMaxScaler` 归一化
- 离散特征 `C*` 使用 `LabelEncoder` 编码
- 稠密特征额外离散化为 `I*_cat`，用于 FM / embedding 交叉特征

当前暂不把 `DIN` 放入第二阶段基础 CTR benchmark。`DIN` 依赖历史行为序列和 target item 特征，更适合后续单独加入 ranking-sequence benchmark。

## 第三阶段：Multi-Task Benchmark

增加多任务数据集和按任务拆分的指标：

- 每个任务的 `AUC` / `LogLoss`
- 跨任务平均指标
- 参数量和运行耗时

初始模型：

- `ESMM`
- `MMOE`
- `PLE`

## CI 策略

- CI 只运行极小样本、单 epoch 的 smoke benchmark。
- 完整 benchmark suite 通过手动 workflow 或定时 workflow 运行。
- benchmark 输出作为 artifact 保存，不自动提交到仓库。

## 配置格式

benchmark 配置使用 YAML，便于后续扩展多模型、多数据集和 suite 级别配置。

示例：

```yaml
task: matching

dataset:
  name: ml-1m-sample
  path: examples/matching/data/ml-1m/ml-1m_sample.csv
  seq_max_len: 50
  neg_ratio: 3
  sample_method: 1
  padding: post
  truncating: post

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

Ranking 配置示例：

```yaml
task: ranking

dataset:
  name: criteo-sample
  path: examples/ranking/data/criteo/criteo_sample.csv

model:
  name: DeepFM
  params:
    embed_dim: 16
    mlp_params:
      dims:
        - 256
        - 128
      dropout: 0.2
      activation: relu

trainer:
  epochs: 1
  batch_size: 128
  learning_rate: 0.001
  weight_decay: 0.001
  split_ratio:
    - 0.7
    - 0.1
  device: cpu
  seed: 2022

metrics:
  - AUC
  - LogLoss

output_dir: benchmark_results/ranking/criteo_deepfm
```
