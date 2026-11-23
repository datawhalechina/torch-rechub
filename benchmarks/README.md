# Benchmarks

在同一份数据上比较多个模型，运行结束后打开 `suite.md` 查看指标。仓库提供全量数据模板，也附带可以直接运行的小样本配置。

## 同一数据集上的模型对比

### MovieLens-1M

每位用户留出最后一部电影作测试。各模型训练 10 轮，按验证集 NDCG@10 选权重，在同一电影库中召回。GAUC 是按用户计算的平均 AUC。每人只有一个测试目标，Recall@K 与 Hit@K 相等。

| 模型 | GAUC | Recall@5 / Hit@5 | Recall@10 / Hit@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SINE | 0.837437 | 0.106623 | 0.168212 | 0.068642 | 0.088378 |
| STAMP | 0.839963 | 0.095199 | 0.149834 | 0.059889 | 0.077447 |
| GRU4Rec | 0.858918 | 0.090563 | 0.147351 | 0.056654 | 0.074937 |
| ComirecDR | 0.854196 | 0.082781 | 0.131126 | 0.052228 | 0.067701 |
| SASRec | 0.832628 | 0.080629 | 0.126159 | 0.051157 | 0.065777 |
| YoutubeDNN | 0.818883 | 0.058609 | 0.103477 | 0.035582 | 0.049994 |
| MIND | 0.827980 | 0.048510 | 0.084272 | 0.029637 | 0.041174 |
| DSSM | 0.805208 | 0.046854 | 0.083444 | 0.029233 | 0.040895 |
| ComirecSA | 0.834305 | 0.045861 | 0.081291 | 0.027935 | 0.039161 |
| NARM | 0.806611 | 0.039073 | 0.074503 | 0.023360 | 0.034827 |

### Avazu_x1

WideDeep、DeepFM 和 DCN 使用同一训练、验证与测试切分，均训练 2 轮。在全部 8,085,794 条测试曝光上计算点击 AUC 和 LogLoss。AUC 越高越好，LogLoss 越低越好。

| 模型 | AUC ↑ | LogLoss ↓ |
| --- | ---: | ---: |
| WideDeep | 0.758915 | 0.369409 |
| DeepFM | 0.758300 | 0.369325 |
| DCN | 0.759873 | 0.369670 |

其他数据集的模型对照、指标定义和复现命令见[中文 Benchmark 文档](../docs/zh/tools/benchmark.md)和[全量结果表](./FULL_RESULTS.md)。

## 用自己的数据比较模型

从仓库根目录运行。先按设备安装 PyTorch，再安装实验依赖。Criteo 和 AliExpress 的全量模板使用 PyArrow，需要加上 `--extra bigdata`。

```bash
uv sync --extra benchmark --extra bigdata
```

以 Criteo 为例，CSV 首行包含点击标签 `label`、数值字段 `I1` 至 `I13` 和类别字段 `C1` 至 `C26`。在 `benchmarks/configs/full.yaml` 中编辑对应段落：

```yaml
criteo:
  path: data/criteo/criteo.csv
  models: [WideDeep, DeepFM]
  epochs: 2
```

激活项目虚拟环境后运行：

```bash
python -m benchmarks.full criteo --device cuda:0
```

默认报告位于 `benchmark_results/full/criteo/suite.md`。`suite.yaml` 保留原始指标和运行环境。此例按配置顺序训练两个排序模型，各自恢复验证点击 AUC 最高的权重，再评估同一测试集。省略 `epochs` 会沿用模板原有的轮数。临时换模型、轮数或输出目录也可以直接传 `--models`、`--epochs`、`--output`。[全量配置](./configs/full.yaml) · [数据下载与格式](../docs/zh/tools/benchmark.md)

## 用自带样本跑通流程

小样本涵盖 MovieLens 召回、Criteo 排序和 Census-Income 多任务。单次运行使用 `runner`，一次比较整组模型使用 `suite`。以下两条命令分别训练一个 DeepFM 和全部三个排序模型：

```bash
python -m benchmarks.runner --config benchmarks/configs/ranking/criteo_deepfm.yaml
python -m benchmarks.suite --configs "benchmarks/configs/ranking/*.yaml" --output benchmark_results/suites/ranking
```

单个配置将 `summary.md` 和 `result.yaml` 写入它的 `output_dir`。整组模型的对比表在 `--output` 指定目录的 `suite.md` 中。运行召回配置前，用 `uv sync --extra benchmark --extra annoy` 安装 Annoy。[小样本配置](./configs/) · [指标与基线检查](./BENCHMARK_DESIGN.md)
