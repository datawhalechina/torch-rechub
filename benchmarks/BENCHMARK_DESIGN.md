# Torch-RecHub Benchmark 设计

同一个模型换一种负采样方式，Recall 就可能变化。为了判断差异来自模型还是实验设置，每次运行都记录数据文件、切分、采样、随机种子和训练预算。这里说明召回、排序和多任务实验各自比较什么。具体命令见[运行说明](./README.md)。

仓库自带的小样本用于检查结果是否能复现。若要据此选线上模型，请换成业务数据，在相同条件下重新比较。

## 目录结构

```text
benchmarks/
├── BENCHMARK_DESIGN.md
├── README.md
├── configs/
│   ├── matching/
│   │   ├── ml_1m_mind.yaml
│   │   ├── ml_1m_youtube_dnn.yaml
│   │   ├── ml_1m_comirec_dr.yaml
│   │   └── ml_1m_comirec_sa.yaml
│   ├── ranking/
│   │   ├── criteo_widedeep.yaml
│   │   ├── criteo_deepfm.yaml
│   │   └── criteo_dcn.yaml
│   └── multitask/
│       ├── census_esmm.yaml
│       ├── census_mmoe.yaml
│       └── census_ple.yaml
├── datasets.py
├── metrics.py
├── models.py
├── report.py
├── runner.py
├── schema.py
└── suite.py
```

## MovieLens 召回实验

每条训练样本包含一部用户实际交互过的电影，以及从电影库中采到的负例。评估时为每位用户留出最后一次交互，检查这部电影排在推荐列表中的什么位置。

支持模型：

- `MIND`
- `YoutubeDNN`
- `ComirecDR`
- `ComirecSA`

数据集：

- `examples/matching/data/ml-1m/ml-1m_sample.csv`
- `generate_seq_feature_match(..., mode=2)` 构造候选列表，第一列放正例

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

## Criteo 点击率实验

三种模型使用同一份曝光样本和固定的随机种子。`split_ratio` 指定训练集与验证集占比，剩下的记录用作测试集。输出包括：

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

Criteo 的每行是一次曝光。训练 DIN 还需要按时间排列的交互历史和当前候选商品，可用[排序示例](../examples/ranking/README.md)中的 Amazon Electronics 数据。

## Census-Income 多任务实验

给定同一条人口普查记录，模型分别预测收入档位和是否从未结婚。报告中的 `cvr_label` 对应收入，`ctr_label` 对应婚姻状态。ESMM 另外使用两列标签的乘积 `ctcvr_label`。各个任务分别记录 AUC，`AUC_mean` 汇总有效任务的 AUC。报告还包括参数量和运行耗时。

支持模型：

- `ESMM`
- `MMOE`
- `PLE`

## 运行方式

Benchmark 当前从仓库根目录手动执行。单个配置使用 `benchmarks/runner.py`，一组配置使用 `benchmarks/suite.py`，输出位于 `benchmark_results/`。[可运行的配置与命令](./README.md)

## 配置格式

下面的两份 YAML 分别对应 MovieLens 召回和 Criteo 排序实验。复查结果时，要连同 `dataset` 和 `trainer` 一起比较，仅有模型名称不足以重现实验。

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
