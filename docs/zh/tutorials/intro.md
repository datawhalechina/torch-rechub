---
title: 场景教程导览
description: Torch-RecHub 场景教程概览与示例入口
---

# 场景教程导览

想先确认环境能训练模型，可以从[快速开始](/zh/guide/quick_start)跑一次 DeepFM。下面三篇教程按任务组织，使用仓库自带的数据。在仓库根目录打开 Notebook 或终端，示例里的相对路径就能找到数据文件。

## 教程列表

- **给候选广告排序**。用 Criteo 的一条曝光记录预测点击概率，再用 AUC 检查模型能否把点击过的广告排在前面。[排序教程](/zh/tutorials/ctr)
- **从电影库找候选**。用 MovieLens 训练 DSSM，分别算出用户向量和电影向量，然后为一个用户取回 10 部电影。[召回教程](/zh/tutorials/retrieval)
- **同时预测点击和购买**。用 Ali-CCP 的两列标签训练 MMOE 或 PLE，分别查看两个任务的 AUC。ESMM 的例子还会用到“点击且购买”这一列。[多任务教程](/zh/tutorials/pipeline)

## 模型使用示例

已经选好模型，可以直接打开对应的教程查看它需要哪些字段。

### 排序模型

| 模型 | 说明 | 链接 |
| --- | --- | --- |
| DeepFM | FM + Deep 联合模型 | [DeepFM 教程](/zh/tutorials/models/ranking/deepfm) |
| Wide&Deep | 线性层和多层网络一起打分 | [Wide&Deep 教程](/zh/tutorials/models/ranking/widedeep) |
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
