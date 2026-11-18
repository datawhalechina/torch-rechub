---
title: 模型库导览
description: Torch-RecHub 模型库概述
---

# 模型库导览

Torch-RecHub 提供了丰富的推荐模型库，涵盖了推荐系统的各个环节，包括排序、召回、多任务学习和生成式推荐。所有模型均基于 PyTorch 实现，易于使用和扩展。

## 模型库结构

模型库按照推荐系统的不同阶段和任务类型进行组织：

1. **排序模型 (Ranking)**：用于精排阶段，预测用户对物品的点击率或偏好分数
2. **召回模型 (Matching)**：用于粗排阶段，从海量物品中召回候选集
3. **多任务模型 (Multi-Task)**：同时优化多个相关任务，提高模型的泛化能力
4. **生成式推荐 (Generative)**：利用生成式模型生成个性化推荐

## 模型选择指南

### 排序模型选择

| 模型 | 适用场景 | 特点 |
| --- | --- | --- |
| WideDeep | 基础排序任务 | 结合线性模型和深度模型，兼顾记忆和泛化能力 |
| DeepFM | 特征交互重要的场景 | 同时捕获低阶和高阶特征交互 |
| DCN/DCNv2 | 显式特征交叉场景 | 显式学习高阶特征交叉，计算效率高 |
| DIN | 用户兴趣动态变化场景 | 基于注意力机制捕捉用户兴趣 |
| DIEN | 长序列兴趣建模 | 建模用户兴趣的动态演化过程 |
| BST | 序列特征重要的场景 | 使用 Transformer 建模序列特征 |
| AutoInt | 自动特征交互学习 | 自动学习特征交互模式 |

### 召回模型选择

| 模型 | 适用场景 | 特点 |
| --- | --- | --- |
| DSSM | 文本匹配场景 | 双塔结构，将用户和物品映射到同一向量空间 |
| YoutubeDNN | 大规模推荐场景 | 基于用户行为序列的深度召回 |
| MIND | 多兴趣推荐场景 | 为用户学习多个兴趣表示 |
| GRU4Rec/SASRec | 序列推荐场景 | 建模用户近期行为序列 |
| ComirecDR/ComirecSA | 可控多兴趣推荐 | 允许控制生成的兴趣数量 |

### 多任务模型选择

| 模型 | 适用场景 | 特点 |
| --- | --- | --- |
| SharedBottom | 任务相关性强的场景 | 所有任务共享底层网络 |
| MMOE | 任务冲突较大的场景 | 多门控专家混合，为不同任务学习不同专家组合 |
| PLE | 复杂多任务场景 | 渐进式分层提取，缓解负迁移问题 |
| ESMM | 样本选择偏差场景 | 全空间建模，解决样本选择偏差 |
| AITM | 任务间存在依赖关系 | 自适应信息迁移，学习任务间的依赖关系 |

### 生成式推荐选择

| 模型 | 适用场景 | 特点 |
| --- | --- | --- |
| HSTU | next-item 序列推荐 | 层级序列转换单元，支持位置/时间偏置 |
| HLLM | 预计算 LLM item embedding 的序列推荐 | 冻结 item 语义表，训练用户序列 Transformer |
| RQ-VAE | item 语义 ID 量化 | 把连续 item embedding 压缩成多级 codebook ID |
| TIGER | 语义 ID 生成式检索 | 用 T5 生成下一个 item 的合法语义 ID |

## 模型文档导航

### 排序模型

详细介绍各种排序模型的原理、使用方法和参数说明。

[查看排序模型文档](/zh/models/ranking)

### 召回模型

详细介绍各种召回模型的原理、使用方法和参数说明。

[查看召回模型文档](/zh/models/matching)

### 多任务模型

详细介绍各种多任务模型的原理、使用方法和参数说明。

[查看多任务模型文档](/zh/models/mtl)

### 生成式推荐模型

详细介绍各种生成式推荐模型的原理、使用方法和参数说明。

[查看生成式推荐模型文档](/zh/models/generative)

## 使用示例

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
