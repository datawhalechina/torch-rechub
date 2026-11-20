# Benchmarks

想比较两个模型时，可以让它们使用同一份数据、同一种切分和相同的训练轮数。`benchmarks/configs/` 里已有三组小样本配置。第一次运行可以从下面的 MovieLens 示例开始。

## 运行 Matching Benchmark

从仓库根目录安装实验依赖，再运行 MIND 配置。召回评估会用 Annoy 在电影向量里查候选。

```bash
uv sync --extra benchmark --extra annoy
uv run python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml
```

在同一份电影数据上，还可以运行 YoutubeDNN 和两个 ComiRec 模型。

```bash
uv run python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_youtube_dnn.yaml
uv run python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_comirec_dr.yaml
uv run python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_comirec_sa.yaml
```

这条 MIND 命令把文件写到 `benchmark_results/matching/ml_1m_mind/`。打开 `summary.md` 看指标和耗时，`result.yaml` 里还有数据路径、模型参数和向量形状。`model.pth` 是本次训练保存的权重。其他实验写到各自配置里的 `output_dir`。

## 当前范围

仓库提供以下三类小样本配置。

### Matching / Retrieval

MovieLens sample 数据：

- `MIND`
- `YoutubeDNN`
- `ComirecDR`
- `ComirecSA`

### Ranking / CTR

Criteo sample 数据：

- `WideDeep`
- `DeepFM`
- `DCN`

运行示例：

```bash
uv run python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_widedeep.yaml
uv run python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_deepfm.yaml
uv run python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_dcn.yaml
```

### Multi-Task

Census-Income 有两个分类任务。`cvr_label` 表示收入是否达到数据中的 50,000 美元档，`ctr_label` 表示婚姻状态是否为“从未结婚”。这两个列名沿用了推荐示例的命名。

- `ESMM`（输出额外的 `ctcvr_label`）
- `MMOE`
- `PLE`

运行示例：

```bash
uv run python benchmarks/runner.py --config benchmarks/configs/multitask/census_esmm.yaml
uv run python benchmarks/runner.py --config benchmarks/configs/multitask/census_mmoe.yaml
uv run python benchmarks/runner.py --config benchmarks/configs/multitask/census_ple.yaml
```

例如，`AUC[cvr_label]` 对应收入预测，`AUC[ctr_label]` 对应婚姻状态。`AUC_mean` 是各个可计算 AUC 的任务的平均值。

## 一次跑一组 (Suite)

如果想一次跑多个 config 并得到一张对比表：

```bash
uv run python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching

uv run python benchmarks/suite.py \
  --configs "benchmarks/configs/ranking/*.yaml" \
  --output  benchmark_results/suites/ranking
```

每个配置仍把 `summary.md` 和 `result.yaml` 写到自己的目录。要并排比较所有模型，打开 `--output` 下的 `suite.md`。后续脚本需要读取整组结果时，用同目录的 `suite.yaml`。

## Config 校验

如果把 `batch_size` 写成 `batchsize`，程序会在训练开始前指出配置里有未知字段。缺少必填项也会在这里报错。

## 指标注册

排序实验按 YAML 的 `metrics` 列表计算指标，可选 `AUC`、`LogLoss`、`Accuracy`、`MSE`。要添加指标，在 `benchmarks/metrics.py` 的 `BINARY_RANKING_METRICS` 中登记名称和函数。函数接收 `y_true` 与 `y_pred`，返回一个分数。

## 基线与回归检测

基线文件 `benchmarks/baselines/<task>.yaml` 为每个指标记录参考值 `expected` 和允许退步的幅度 `tolerance`。加上 `--check-baseline` 后，若指标向较差的方向变化且超出这个幅度，命令会以非零状态退出。例如 AUC 下降和 LogLoss 上升都可能触发检查。

```bash
# 单次运行
uv run python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_dcn.yaml --check-baseline

# 整组运行
uv run python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching \
  --check-baseline
```

确认新的一组结果可以重复后，从 `benchmark_results/suites/<task>/suite.yaml` 取出相应指标，更新基线中的 `expected`，并按多次运行的波动设定 `tolerance`。

各任务的数据切分和指标见[实验设计](./BENCHMARK_DESIGN.md)。
