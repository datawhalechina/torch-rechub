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

更完整的路线图见 `BENCHMARK_DESIGN.md`。
