# Benchmarks

benchmark 脚本用于做可复现的模型比较。它们和 `examples/` 分开维护：`examples/` 侧重教学和演示，`benchmarks/` 侧重固定实验协议和结果输出。

## 运行 Matching Benchmark

在仓库根目录执行：

```bash
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_mind.yaml
```

第一阶段还提供以下配置：

```bash
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_youtube_dnn.yaml
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_comirec_dr.yaml
python benchmarks/runner.py --config benchmarks/configs/matching/ml_1m_comirec_sa.yaml
```

每次运行会写出：

- `result.yaml`：结构化 benchmark 元数据和指标
- `summary.md`：简洁的人类可读摘要
- `model.pth`：`MatchTrainer` 保存的模型权重

默认输出目录是：

```text
benchmark_results/
```

该目录已加入 `.gitignore`，不会提交到仓库。

## 当前范围

当前支持两类轻量 benchmark。

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
python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_widedeep.yaml
python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_deepfm.yaml
python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_dcn.yaml
```

### Multi-Task

Census-Income sample 数据，双任务（income = cvr_label，marital status = ctr_label）：

- `ESMM`（输出额外的 `ctcvr_label`）
- `MMOE`
- `PLE`

运行示例：

```bash
python benchmarks/runner.py --config benchmarks/configs/multitask/census_esmm.yaml
python benchmarks/runner.py --config benchmarks/configs/multitask/census_mmoe.yaml
python benchmarks/runner.py --config benchmarks/configs/multitask/census_ple.yaml
```

指标按任务命名（例如 `AUC[cvr_label]`、`AUC[ctr_label]`），并在任务超过 1 个时额外输出 `AUC_mean`。

## 一次跑一组 (Suite)

如果想一次跑多个 config 并得到一张对比表：

```bash
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching

python benchmarks/suite.py \
  --configs "benchmarks/configs/ranking/*.yaml" \
  --output  benchmark_results/suites/ranking
```

Suite 会：

- 依次跑每个 config，并调用 `runner.py` 相同的落地逻辑写各自的 `result.yaml` / `summary.md`
- 在 `--output` 目录下额外写一份 `suite.yaml`（结构化）和 `suite.md`（对比表）

## Config 校验

加载 config 时会做轻量 schema 校验（必填字段 + 白名单 key），拼错的 key（如 `batchsize` → `batch_size`）会直接报错，不再静默 fallback 到默认值。

## 指标注册

Ranking 的 `metrics: [...]` 真正驱动指标选择；可选项由 `benchmarks/metrics.py::BINARY_RANKING_METRICS` 定义（当前支持 `AUC`、`LogLoss`、`Accuracy`、`MSE`）。要新增指标只需在字典里注册一个 `(y_true, y_pred) -> float`。拼写错误会在 `validate_config` 阶段就抛出。

## 基线与回归检测

`benchmarks/baselines/<task>.yaml` 按 config 相对路径记录每个指标的 `expected + tolerance`。跑 runner 或 suite 时加 `--check-baseline` 会比较新结果：任何指标偏离 tolerance 会非零退出。

```bash
# 单次运行
python benchmarks/runner.py --config benchmarks/configs/ranking/criteo_dcn.yaml --check-baseline

# 整组运行
python benchmarks/suite.py \
  --configs "benchmarks/configs/matching/*.yaml" \
  --output  benchmark_results/suites/matching \
  --check-baseline
```

更新基线的工作流程：跑一次 suite 得到稳定结果 → 把 `benchmark_results/suites/<task>/suite.yaml` 里对应的 `metrics` 复制到 `benchmarks/baselines/<task>.yaml`，并给出合适的 `tolerance`。

更完整的路线图见 `BENCHMARK_DESIGN.md`。
