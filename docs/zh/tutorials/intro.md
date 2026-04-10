---
title: 场景教程导览
description: Torch-RecHub 场景教程概览与示例入口
---

# 场景教程导览

本章节聚焦 Torch-RecHub 在不同推荐场景下的实战用法。文档中的代码默认基于仓库内置样本数据，建议在**仓库根目录**执行。

> **代码资源**：
> - 完整 Python 示例脚本：`examples/`
> - 文档内按步骤拆解的教程：`docs/zh/tutorials/`

## 教程列表

| 教程 | 适用场景 | 链接 |
| --- | --- | --- |
| CTR 预测 | 排序 / 点击率预估 | [CTR 预测教程](/zh/tutorials/ctr) |
| 召回模型 | 双塔召回 / 向量检索 | [召回模型教程](/zh/tutorials/retrieval) |
| 多任务学习 | CTR/CVR 联合建模 | [多任务教程](/zh/tutorials/pipeline) |

## 快速导航

### CTR 预测（精排）

适合想快速跑通 `WideDeep / DeepFM / DCN` 的用户。

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128]})
trainer = CTRTrainer(model, device="cuda:0")
trainer.fit(train_dl, val_dl)
```

[查看完整教程 →](/zh/tutorials/ctr)

### 召回模型

适合想跑通 `DSSM / YoutubeDNN / MIND` 的双塔或多兴趣召回链路。

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

model = DSSM(user_features, item_features)
trainer = MatchTrainer(model, mode=0, device="cuda:0")
trainer.fit(train_dl)
```

[查看完整教程 →](/zh/tutorials/retrieval)

### 多任务学习

适合想了解 `MMOE / PLE / ESMM` 在 Ali-CCP 样本数据上的训练流程。

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer

model = MMOE(features, task_types=["classification", "classification"], n_expert=8,
             expert_params={"dims": [16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])
trainer = MTLTrainer(model, task_types=["classification", "classification"], device="cuda:0")
trainer.fit(train_dl, val_dl)
```

[查看完整教程 →](/zh/tutorials/pipeline)

## 模型使用示例

按模型分类的详细使用教程，每篇包含数据准备、模型配置、训练、评估和调优建议。

### 排序模型

| 模型 | 说明 | 链接 |
| --- | --- | --- |
| DeepFM | FM + Deep 联合模型 | [DeepFM 教程](/zh/tutorials/models/ranking/deepfm) |
| Wide&Deep | 记忆 + 泛化联合模型 | [Wide&Deep 教程](/zh/tutorials/models/ranking/widedeep) |
| DCN / DCNv2 | 显式特征交叉网络 | [DCN 教程](/zh/tutorials/models/ranking/dcn) |
| DIN | 目标注意力序列模型 | [DIN 教程](/zh/tutorials/models/ranking/din) |
| DIEN | 兴趣演化序列模型 | [DIEN 教程](/zh/tutorials/models/ranking/dien) |
| BST | Transformer 序列模型 | [BST 教程](/zh/tutorials/models/ranking/bst) |

### 召回模型

| 模型 | 说明 | 链接 |
| --- | --- | --- |
| DSSM | 双塔语义匹配模型 | [DSSM 教程](/zh/tutorials/models/matching/dssm) |
| YoutubeDNN | YouTube 深度召回模型 | [YoutubeDNN 教程](/zh/tutorials/models/matching/youtube_dnn) |
| MIND | 多兴趣胶囊网络召回 | [MIND 教程](/zh/tutorials/models/matching/mind) |

### 多任务模型

| 模型 | 说明 | 链接 |
| --- | --- | --- |
| MMOE | 多门控专家混合模型 | [MMOE 教程](/zh/tutorials/models/multi_task/mmoe) |
| PLE | 渐进分层提取多任务模型 | [PLE 教程](/zh/tutorials/models/multi_task/ple) |

## 推荐的验证顺序

1. 先跑 [快速开始](/zh/guide/quick_start)，确认环境、训练器、样本数据都可用。
2. 再看 [CTR 教程](/zh/tutorials/ctr) 或 [召回教程](/zh/tutorials/retrieval)，理解完整数据流。
3. 最后深入具体模型页，查看参数解释、调优建议和 ONNX / 可视化用法。
