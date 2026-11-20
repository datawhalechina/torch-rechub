---
title: 模型库导览
description: Torch-RecHub 模型库概述
---

# 模型库导览

从日志里挑一条样本，就能判断该从哪类模型开始。假设你在做电影推荐：

| 想回答的问题 | 需要的数据 | 从这里开始 |
| --- | --- | --- |
| 从片库里先找出哪几百部电影？ | 用户看过的电影，以及片库中的候选电影 | [召回模型](/zh/models/matching)。DSSM 分别计算用户向量和电影向量，按相似度找候选。 |
| 候选电影里，哪一部该排在前面？ | 每次曝光的用户、电影、上下文和点击标签 | [排序模型](/zh/models/ranking)。DeepFM 给每条曝光打一个分，候选电影可以按这个分数排序。 |
| 用户会不会点开？点开后又会不会付费？ | 同一次曝光上的点击和转化标签 | [多任务模型](/zh/models/mtl)。SharedBottom 为每个目标输出一列预测，ESMM 还能利用点击与后续转化的关系。 |
| 看过这几部电影之后，下一部可能是什么？ | 每个用户按时间排列的观看记录 | [生成式推荐模型](/zh/models/generative)。HSTU 根据历史给电影打分，TIGER 逐步生成下一部电影的语义 ID。 |

## 模型选择指南

第一次跑排序可以从 DeepFM 开始，有了按时间排序的用户历史，再试 DIN 怎样用候选电影给历史行为分配权重。做召回时，先用 DSSM 跑通从训练到向量检索的过程。如果希望“悬疑片”和“动画片”两种兴趣都能召回候选，可以再试 MIND 的多兴趣向量。

多任务实验先看同一张表上能取到哪些标签。只有点击标签时就用单任务排序。如果还有转化标签，可以在同一份验证集上分别比较两个任务的 AUC。生成式模型的选择还取决于现有材料：有交互序列即可训练 HSTU，训练 HLLM 还要准备电影文本向量，训练 TIGER 则需要先为电影建立语义 ID。

## 使用示例

排序和召回的训练调用很像，区别在于怎样准备一批样本。下面展示两个独立的调用片段，各自的特征和 DataLoader 定义见[排序入门](/zh/guide/quick_start)和[召回教程](/zh/tutorials/retrieval)。

```python
# 排序模型使用示例
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

# 创建模型
model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2})

# 创建训练器
trainer = CTRTrainer(model, optimizer_params={"lr": 0.001}, device="cpu")

# 训练模型
trainer.fit(train_dataloader, val_dataloader)

# 召回模型使用示例
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

# 创建模型
model = DSSM(user_features=user_features, item_features=item_features, temperature=1.0,
             user_params={"dims": [256, 128, 64]}, item_params={"dims": [256, 128, 64]})

# 创建训练器
trainer = MatchTrainer(model, mode=0, device="cpu")

# 训练模型
trainer.fit(train_dataloader)
```

## 模型论文

* **排序**　[DeepFM](https://arxiv.org/abs/1703.04247)、[Wide&Deep](https://arxiv.org/abs/1606.07792)、[DCN](https://arxiv.org/abs/1708.05123)、[DCNv2](https://arxiv.org/abs/2008.13535)、[DIN](https://arxiv.org/abs/1706.06978)、[DIEN](https://arxiv.org/abs/1809.03672)、[BST](https://arxiv.org/abs/1905.06874)、[AFM](https://arxiv.org/abs/1708.04617)、[AutoInt](https://arxiv.org/abs/1810.11921)、[FiBiNET](https://doi.org/10.1145/3298689.3347043)、[FAT-DeepFFM](https://arxiv.org/abs/1905.06336)、[EDCN](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf)。
* **召回**　[DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)、[YouTubeDNN](https://dl.acm.org/doi/10.1145/2959100.2959190)、[YouTubeSBC](https://doi.org/10.1145/3298689.3346996)、[MIND](https://arxiv.org/abs/1904.08030)、[SINE](https://arxiv.org/abs/2102.09267)、[GRU4Rec](https://arxiv.org/abs/1511.06939)、[SASRec](https://arxiv.org/abs/1808.09781)、[NARM](https://arxiv.org/abs/1711.04725)、[STAMP](https://doi.org/10.1145/3219819.3219950)、[ComiRec](https://arxiv.org/abs/2005.09347)。
* **多任务学习**　[ESMM](https://arxiv.org/abs/1804.07931)、[MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007)、[PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)、[AITM](https://arxiv.org/abs/2105.08489)。
* **序列与生成式推荐**　[HSTU](https://arxiv.org/abs/2402.17152)、[HLLM](https://arxiv.org/abs/2409.12740)、[TIGER](https://arxiv.org/abs/2305.05065)。

DeepFFM 与 FAT-DeepFFM 共用一篇论文，前者为文中的基础模型。

## 贡献新模型

如果您想贡献新的模型，请参考 [贡献指南](/zh/community/contributing)，遵循项目的编码规范和文档要求。
