---
title: 研发工具导览
description: Torch-RecHub 研发工具概述
---

# 研发工具导览

在排序、召回和多任务示例里，给 `fit()` 传入 `log_interval=1`，进度条就会显示每个 batch 的损失和学习率。只想确认模型有没有开始学习，看这里就够了。

如果要回看训练结果，可以按问题找工具。

- **哪一轮的模型值得保存？** 用验证集监控 AUC 或损失。训练器会保留最佳权重，可以在指标连续几轮没有改善时停止。[早停与保存](/zh/tools/callbacks)
- **两次训练的曲线有什么区别？** 把损失和验证指标写到本地 TensorBoard 日志，或发到团队使用的 W&B、SwanLab 项目。[实验追踪](/zh/tools/tracking)
- **特征经过了哪些层，形状在哪里变了？** 给模型画一张前向计算图，可以直接在 Jupyter 查看，也可以保存为文件。[模型结构图](/zh/tools/visualization)
- **换模型后指标变了吗？** 用同一份数据和配置跑基准实验，查看两次结果的对比表。[Benchmark](/zh/tools/benchmark)

