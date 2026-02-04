---
title: 场景教程导览
description: Torch-RecHub 场景教程概述
---

# 场景教程导览

本章节提供 Torch-RecHub 在不同推荐场景下的实战教程，帮助开发者快速上手。

> **代码资源**：项目提供了交互式 Jupyter Notebook 教程（位于 `tutorials/` 目录）和完整的 Python 示例脚本（位于 `examples/` 目录），可配合本文档学习使用。

## 教程列表

| 教程 | 描述 | 链接 |
| --- | --- | --- |
| **CTR 预测** | 点击率预测模型训练 | [CTR 预测教程](/zh/tutorials/ctr) |
| **召回模型** | 双塔召回模型训练 | [召回模型教程](/zh/tutorials/retrieval) |
| **完整流程** | 端到端推荐系统 | [完整流程教程](/zh/tutorials/pipeline) |

## 快速导航

### CTR 预测（精排）

学习如何使用 DeepFM、DCN 等模型进行点击率预测。

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

model = DeepFM(deep_features, fm_features, mlp_params)
trainer = CTRTrainer(model)
trainer.fit(train_dl, val_dl)
```

[查看完整教程 →](/zh/tutorials/ctr)

### 召回模型

学习如何使用 DSSM、MIND 等模型进行向量召回。

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

model = DSSM(user_features, item_features)
trainer = MatchTrainer(model)
trainer.fit(train_dl)
```

[查看完整教程 →](/zh/tutorials/retrieval)

