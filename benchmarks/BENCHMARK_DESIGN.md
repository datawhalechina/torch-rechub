# Benchmark 实验设计

把模型放在同一张指标表前，先固定数据版本、训练与测试划分、目标标签、评估样本和训练轮数。下面列出各入口采用的口径。[运行命令](./README.md)

## 全量实验怎样比较模型

同一张表使用相同的数据划分、验证指标、训练预算和测试候选集。每个模型按验证集选权重，最后对同一测试集评估。多任务模型使用一组权重报告各任务 AUC。种子、批大小、模型特有超参数和数据处理可在对应数据集的训练脚本中查看。[训练脚本](./full_runs/) · [实测结果](./FULL_RESULTS.md)

| 数据集 | 测试对象 | 主要用途 |
| --- | --- | --- |
| MovieLens-1M | 每位用户最后一次交互的电影，从其历史之外的电影库中检索 | 按验证 NDCG@10 选权重，测试时报告用户平均 AUC（GAUC）以及 K=5、10 的 Recall、Hit 和 NDCG |
| Criteo、Avazu_x1 | 每条广告曝光的点击标签 | 比较 CTR 模型 |
| Amazon Electronics | 每位用户最后一次交互搭配一个采样负例 | 评估 DIN 区分真实交互与采样物品的能力 |
| Census-Income | 每条人口普查记录的收入与婚姻状态 | 比较多任务模型 |
| Ali-CCP、AliExpress-US | 全部测试曝光的点击与购买或转化标签 | 比较点击与转化预测 |

MovieLens 每位用户留出一个测试正例，因此相同 K 下的 Recall 与 Hit 数值相同。CTR 任务按全部测试曝光计算 ROC AUC 和适用的 LogLoss。比较不同框架时，还需核对是否使用全库候选、是否过滤已交互物品，以及数据划分、标签阈值和预处理方式。[RecBole 的 Top-K 指标](https://github.com/RUCAIBox/RecBole/blob/master/recbole/evaluator/metrics.py) · [FuxiCTR 的 CTR 指标](https://github.com/reczoo/FuxiCTR/blob/main/fuxictr/metrics.py)

## 仓库小样本如何计算指标

| 任务 | 配置和评估方式 |
| --- | --- |
| [MovieLens 召回](./configs/matching/) | 训练时按用户历史构造候选列表，第一列是正例。评估时留出最后一次交互，用 Annoy 在电影库中检索，计算 Hit、Recall、NDCG、MRR 和 Precision 的 Top-K 指标。报告也记录用户和物品向量的形状。 |
| [Criteo 排序](./configs/ranking/) | `split_ratio` 指定训练集和验证集的占比，其余曝光作为测试集。默认报告 AUC 和 LogLoss，也可以在 `metrics` 中选择 Accuracy 或 MSE。 |
| [Census-Income 多任务](./configs/multitask/) | `cvr_label` 表示收入档位，`ctr_label` 表示是否从未结婚。ESMM 还预测两项标签的乘积 `ctcvr_label`。`AUC_mean` 是可计算的任务 AUC 的算术平均值。 |

## 检查小样本是否退步

单个配置或整组运行时都可以加 `--check-baseline`。每项指标在基线文件中有参考值 `expected` 和容差 `tolerance`。AUC 下降或 LogLoss 上升超过容差时，命令以非零状态退出。[基线文件](./baselines/)

修改模型或预处理后，先在相同设置下重复运行，再依据稳定的指标和波动幅度更新基线。若要为排序实验增加指标，可在 `BINARY_RANKING_METRICS` 中登记一个接收真实标签和预测分数的函数。[指标实现](./metrics.py)
