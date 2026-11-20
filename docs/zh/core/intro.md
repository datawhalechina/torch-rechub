---
title: 核心组件导览
description: Torch-RecHub 核心组件概述
---

# 核心组件导览

以电影推荐为例。我们知道用户看过哪些电影，也知道这次给他展示了哪部电影。要训练一个预测点击的模型，需要先把这条记录变成模型能读的数字，再把预测结果和实际点击标签放在一起计算损失。

## 一条记录怎样进入模型

1. **编码。** 假设词表里电影 `M42` 的编号是 `3`。用户只看过这部电影，历史长度设为 3，那么 `hist_movie_id` 就是 `[3, 0, 0]`。后两个位置是补齐用的。候选电影和历史电影要用同一份词表。[特征定义](/zh/core/features)
2. **组批。** 假设每批有 `B` 条记录，候选电影 ID 的形状是 `[B]`，历史是 `[B, 3]`。`DataGenerator` 把这些特征放在同一个字典里，连同 `B` 个点击标签交给训练器。[数据加载](/zh/core/data)
3. **训练。** 排序模型读取特征字典，给每条记录打分。`CTRTrainer.fit(train_dl, val_dl)` 用训练集更新参数，每轮在验证集上计算 AUC。[训练与评估](/zh/core/evaluation)

## 其他任务用哪个训练器

召回时要从物品库中选出候选。`MatchDataGenerator` 为用户和物品分别准备数据，`MatchTrainer` 可以在双塔模型训练结束后分别算出两侧向量。[召回教程](/zh/tutorials/retrieval)

一条样本有多个标签时，仍可用 `DataGenerator` 组批。按标签列的顺序配置 `MTLTrainer` 的任务，训练后可以单独查看每个任务的指标。[多任务教程](/zh/tutorials/pipeline)

HSTU 和 HLLM 用 `SequenceDataGenerator` 和 `SeqTrainer` 学习下一件物品。RQ-VAE 学习物品的语义 ID，TIGER 再学习如何生成这些 ID，两者各有对应的训练脚本。[生成式模型](/zh/models/generative)
