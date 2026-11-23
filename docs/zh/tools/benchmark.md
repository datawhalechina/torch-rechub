---
title: Benchmark
description: 在同一份数据上比较推荐模型，复现全量 GPU 实验并生成 Markdown 指标报告。
---

# Benchmark

Torch-RecHub 模型均使用一张 RTX 5080（16 GB）训练。每个数据集上的模型使用相同的划分与训练轮数。召回表看推荐列表有没有命中用户的下一部电影，排序和多任务表看模型能否预测点击、转化或相应标签。表中只列最终测试分数。

## 按数据集比较模型

### MovieLens-1M

将 1,000,209 条评分都视为交互。对每位用户，按时间留出最后一部电影作测试、倒数第二部作验证。10 个模型各训练 10 轮，按验证 NDCG@10 选权重，再为 6,040 位用户从 3,706 部电影中做全库召回。

| 模型 | GAUC | Recall@5 | Recall@10 | Hit@5 | Hit@10 | NDCG@5 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SINE | 0.837437 | 0.106623 | 0.168212 | 0.106623 | 0.168212 | 0.068642 | 0.088378 |
| STAMP | 0.839963 | 0.095199 | 0.149834 | 0.095199 | 0.149834 | 0.059889 | 0.077447 |
| GRU4Rec | 0.858918 | 0.090563 | 0.147351 | 0.090563 | 0.147351 | 0.056654 | 0.074937 |
| ComirecDR | 0.854196 | 0.082781 | 0.131126 | 0.082781 | 0.131126 | 0.052228 | 0.067701 |
| SASRec | 0.832628 | 0.080629 | 0.126159 | 0.080629 | 0.126159 | 0.051157 | 0.065777 |
| YoutubeDNN | 0.818883 | 0.058609 | 0.103477 | 0.058609 | 0.103477 | 0.035582 | 0.049994 |
| MIND | 0.827980 | 0.048510 | 0.084272 | 0.048510 | 0.084272 | 0.029637 | 0.041174 |
| DSSM | 0.805208 | 0.046854 | 0.083444 | 0.046854 | 0.083444 | 0.029233 | 0.040895 |
| ComirecSA | 0.834305 | 0.045861 | 0.081291 | 0.045861 | 0.081291 | 0.027935 | 0.039161 |
| NARM | 0.806611 | 0.039073 | 0.074503 | 0.039073 | 0.074503 | 0.023360 | 0.034827 |

每位用户只有一个测试目标，因此相同 K 下 Recall 和 Hit 相等。NDCG 还考虑目标在前 K 个位置中的名次。GAUC 按用户分别计算 AUC 后取平均，同分的正负样本对得 0.5 分。[RecBole 指标定义](https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py)

各模型固定种子 2022，使用最近 20 次交互和批内负样本训练。NARM、STAMP、SASRec 和 SINE 仅使用电影历史，其他模型还使用用户 ID。测试时过滤用户此前交互过的电影，保留本次目标。MIND 和两种 ComiRec 对每部电影取四个兴趣向量中的最高分。

STAMP 在本实验中使用 `emb_std=0.1`。原示例的 0.002 在本次权重衰减和批内采样设置下使所有电影的分数收敛到零。

### Avazu_x1

三个模型使用 Avazu_x1 公开切分，训练 2 轮，以验证点击 AUC 选择权重，在 CPU 上用同一组权重对全部 8,085,794 条测试曝光打分。ROC AUC 越高，说明模型将正例排在负例前的能力越好。LogLoss 越低，说明预测概率更接近实际标签。[FuxiCTR 指标实现](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/metrics.py)

| 模型 | Torch-RecHub AUC ↑ | Torch-RecHub LogLoss ↓ | BARS AUC ↑ | BARS LogLoss ↓ |
| --- | ---: | ---: | ---: | ---: |
| WideDeep | 0.758915 | 0.369409 | 0.7649 | 0.3665 |
| DeepFM | 0.758300 | 0.369325 | 0.7648 | 0.3667 |
| DCN | 0.759873 | 0.369670 | 0.7652 | 0.3665 |

BARS 的同一切分使用 10 维 embedding，本次使用 16 维。这里沿用项目示例，在三个切分上拟合词表和缩放器，前三个字段也将归一化值转换为类别索引。跨框架复现分数时，还需对齐这些输入和训练设置。[BARS 评估设置](https://openbenchmark.github.io/BARS/CTR/leaderboard/avazu_x1.html) · [BARS 原始结果](https://raw.githubusercontent.com/reczoo/RecBench/main/docs/CTR/leaderboard/avazu_x1.csv)

### Criteo

两个模型均训练 2 轮，固定种子 2022，按曝光随机分为 70% 训练、10% 验证和 20% 测试。AUC 和 LogLoss 使用同一组权重在 CPU 上对 9,168,125 条测试曝光打分。数值缩放和类别词表仅用训练行拟合。DeepFM 另外将训练中出现少于 10 次的类别映射为 0，深层网络和 FM 共享 26 个类别 embedding。

| 模型 | AUC ↑ | LogLoss ↓ |
| --- | ---: | ---: |
| WideDeep | 0.790378 | 0.459636 |
| DeepFM | 0.804685 | 0.446696 |

### Census-Income

使用发布的三个切分，训练 10 轮，在 49,881 条测试记录上分别预测收入是否属于 `50000+` 档、婚姻状态是否为「从未结婚」。所有任务共用按收入验证 AUC 选出的权重。ESMM 还预测一个联合标签，即同时满足这两个条件。PLE 的两行是同一模型的不同层数。

| 模型 | 收入 AUC ↑ | 婚姻状态 AUC ↑ | 联合标签 AUC ↑ |
| --- | ---: | ---: | ---: |
| SharedBottom | 0.951358 | 0.994704 | — |
| ESMM | 0.841657 | 0.991467 | 0.970112 |
| MMOE | 0.951119 | 0.994779 | — |
| PLE（1 层） | 0.948204 | 0.994738 | — |
| PLE（2 层） | 0.950635 | 0.994746 | — |
| AITM | 0.949838 | 0.994583 | — |

### AliExpress-US

两个模型均训练 2 轮，从训练文件按 `search_id` 留出约 10% 的搜索作验证，以转化 AUC 选择权重。点击和转化的测试 AUC 都在全部 7,460,564 条曝光上计算。美国分区使用公开镜像，20 条仓库测试样本中有两条转化标签与镜像不同。此表使用镜像标签。[镜像文件](https://drive.google.com/file/d/1oM9IcBYsaI7xJ2i6l-BnjUadU2YKquk0/view)

| 模型 | 点击 AUC ↑ | 转化 AUC ↑ |
| --- | ---: | ---: |
| SharedBottom | 0.698530 | 0.795083 |
| AITM | 0.692870 | 0.770054 |

## 用自己的数据比较模型

从仓库根目录运行。先准备与设备匹配的 PyTorch 环境，再安装实验依赖。Criteo 和 AliExpress 用 PyArrow 分块读取数据，需要 `bigdata`。其他全量数据集只需 `uv sync --extra benchmark`。[安装指南](../guide/install.md)

```bash
uv sync --extra benchmark --extra bigdata
```

在 `benchmarks/configs/full.yaml` 中找到对应的数据集。`path` 填文件或目录，`models` 填要依次训练的模型名称。如果使用自己的 Criteo 格式 CSV，首行需要包含二值点击标签 `label`、13 个数值列 `I1` 至 `I13` 和 26 个类别列 `C1` 至 `C26`。

```yaml
criteo:
  path: data/criteo/criteo.csv
  models: [WideDeep, DeepFM]
  epochs: 2
```

激活项目虚拟环境后运行下面的命令。这里两个模型分别训练 2 轮，各自恢复验证 AUC 最高的权重，再评估同一测试集。省略 `epochs` 会使用各模型原有的训练轮数。用自己的数据试跑时，主任务验证集需要同时包含正例和负例，才能按 AUC 选权重。

```bash
python -m benchmarks.full criteo --device cuda:0
```

打开 `benchmark_results/full/criteo/suite.md` 比较两个模型的测试 AUC 和 LogLoss。表中 `—` 表示该模型没有这项指标，或测试标签只有一类、无法计算 AUC。同目录的 `suite.yaml` 保留数值、数据路径、设备和代码版本。临时换数据、模型或轮数时，可以用命令行覆盖配置。`--output` 同时指定报告、权重和临时编码文件的位置，跑全量数据时应选择空间充足的数据盘。

```bash
python -m benchmarks.full census --data data/census-income --models MMOE PLE --epochs 10 --output benchmark_results/census_10
```

### 数据文件和可选模型

| 配置名 | `path` 指向的文件或目录 | 可选模型 |
| --- | --- | --- |
| `movielens` | 由原始评分、电影和用户文件生成的 `ml-1m.csv`。[MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) | YoutubeDNN、MIND、ComirecDR、ComirecSA、DSSM、GRU4Rec、NARM、STAMP、SASRec、SINE |
| `criteo` | 把 7 个分卷放在同一目录，从 `.001` 开始解压后使用 `criteo.csv`。[Criteo](https://aistudio.baidu.com/dataset/detail/389642) | WideDeep、DeepFM、DCN |
| `avazu` | 直接使用含 `train.csv`、`valid.csv` 和 `test.csv` 的压缩包。[Avazu_x1.zip](https://zenodo.org/records/5700987) | WideDeep、DeepFM、DCN |
| `amazon` | 含 `user_id`、`item_id`、`cate_id`、`time` 的完整 CSV。[Amazon Electronics](https://aistudio.baidu.com/dataset/detail/355993) | DIN |
| `census` | `census_income_{train,val,test}.csv` 所在目录。[Census-Income](https://aistudio.baidu.com/dataset/detail/390912) | SharedBottom、ESMM、MMOE、PLE、PLE-2、AITM |
| `ali_ccp` | `ali_ccp_{train,val,test}.csv` 所在目录。[Ali-CCP](https://aistudio.baidu.com/datasetdetail/281072) | DeepFM、MMOE |
| `aliexpress` | 美国分区的 `AliExpress_US.zip`，内含 `train.csv` 和 `test.csv`。[实测所用镜像](https://drive.google.com/file/d/1oM9IcBYsaI7xJ2i6l-BnjUadU2YKquk0/view) | SharedBottom、AITM |

### 准备完整 MovieLens-1M

从 [GroupLens](https://grouplens.org/datasets/movielens/1m/) 下载 `ml-1m.zip`。将包中的 `ratings.dat`、`movies.dat` 和 `users.dat` 放入 `examples/matching/data/ml-1m/`。在已激活的项目环境中，从仓库根目录执行以下命令生成完整 CSV，并按配置运行召回模型。

```bash
cd examples/matching
python data/ml-1m/preprocess_ml.py
cd ../..
python -m benchmarks.full movielens --device cuda:0
```

## 用仓库样本试跑

仓库带有下面三组小样本和配置。只想看流程时，可以先在 CPU 上运行。配置放在 `benchmarks/configs/` 对应的子目录。

| 子目录 | 数据与模型 | 指标 |
| --- | --- | --- |
| `matching/` | MovieLens · MIND、YoutubeDNN、ComirecDR、ComirecSA | Hit@K、Recall@K、NDCG@K、MRR@K、Precision@K |
| `ranking/` | Criteo · WideDeep、DeepFM、DCN | 可选 AUC、LogLoss、Accuracy、MSE |
| `multitask/` | Census-Income · ESMM、MMOE、PLE | 各任务的 AUC，以及可计算任务的平均 AUC |

以下命令先为召回实验安装 Annoy。`runner.py` 运行一个配置，在该配置的 `output_dir` 生成 `summary.md` 和 `result.yaml`。`suite.py` 依次运行匹配的配置，在 `--output` 下生成 `suite.md` 和 `suite.yaml`。

```bash
uv sync --extra benchmark --extra annoy
python -m benchmarks.runner --config benchmarks/configs/matching/ml_1m_mind.yaml
python -m benchmarks.suite --configs "benchmarks/configs/matching/*.yaml" --output benchmark_results/suites/matching
```

修改模型或训练器后，可在任一小样本运行命令中加 `--check-baseline`。脚本按 `benchmarks/baselines/<task>.yaml` 中的参考值与容差检查回归，AUC 下降或 LogLoss 上升超过容差时会以非零状态退出。
