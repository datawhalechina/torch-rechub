---
title: 召回模型
description: Torch-RecHub 召回模型详细介绍
---

# 召回模型

做电影推荐时，召回这一步先从整个片库挑出一批候选电影。以 DSSM 为例，训练后可以提前算好每部电影的向量。来了一个用户，只需算一次用户向量，再去片库里找相似的电影。[MovieLens 召回教程](/zh/tutorials/retrieval)演示了怎样把检索结果还原成电影 ID。

## 1. DSSM

DSSM（Deep Structured Semantic Models）有两个塔。用户塔读取用户属性或观看历史，物品塔读取电影 ID 等属性，两边输出相同维度的向量。

### 论文引用

```
Huang, Po-Sen, et al. "Learning deep structured semantic models for web search using clickthrough data." Proceedings of the 22nd ACM international conference on Information & Knowledge Management. 2013.
```

本实现先对向量做 L2 归一化。默认训练时，同一行的用户和电影做点积，经过 sigmoid 后与这行的 0/1 标签计算损失。开启批内负采样时，训练器会用同一批的其他电影组成候选。

### 使用方法

```python
import os

from torch_rechub.models.matching import DSSM
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 定义特征
user_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

item_features = [
    SparseFeature(name="item_id", vocab_size=100000, embed_dim=32),
    SparseFeature(name="category", vocab_size=1000, embed_dim=16)
]

# 创建模型
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=1.0,  # 用于批内采样的相似度缩放
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "prelu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | 必填 |
| item_features | list | 物品特征列表 | 必填 |
| temperature | float | 保留参数；当前 DSSM `forward` 未将它应用到分数 | 1.0 |
| user_params | dict | 用户塔网络参数 | 必填 |
| item_params | dict | 物品塔网络参数 | 必填 |

## 2. FaceBookDSSM

FaceBookDSSM 用一条样本比较两部电影。一部是用户有过正反馈的电影，另一部是采样得到的负例。

两个电影共用物品塔。默认前向返回正例分和负例分，配合 `MatchTrainer(mode=1)` 用 BPR 损失拉大两者的差距。

### 使用方法

```python
from torch_rechub.models.matching import FaceBookDSSM

# 创建模型
model = FaceBookDSSM(
    user_features=user_features,
    pos_item_features=item_features,
    neg_item_features=neg_item_features,
    user_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
    item_params={"dims": [512, 256, 128], "dropout": 0.3, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | 必填 |
| pos_item_features | list | 正样本物品特征列表 | 必填 |
| neg_item_features | list | 与正样本对应的负样本特征列表 | 必填 |
| user_params | dict | 用户塔网络参数 | 必填 |
| item_params | dict | 正负物品共用的物品塔参数 | 必填 |

## 3. YoutubeDNN

YoutubeDNN 用用户特征和观看历史预测接下来会看的电影。用户侧用 MLP 得到一个向量，物品侧直接查电影 ID 的嵌入表。

### 论文引用

```
Covington, Paul, Jay Adams, and Emre Sargin. "Deep neural networks for youtube recommendations." Proceedings of the 10th ACM conference on recommender systems. 2016.
```

下面的例子先对历史电影向量取平均。训练时把目标电影放在候选列表第一个位置，再拼上采样的负电影。用户向量的维度必须与电影向量一致，两者才能做点积。

### 使用方法

```python
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import YoutubeDNN

user_features = [
    SparseFeature("user_id", vocab_size=10000, embed_dim=16),
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="mean", shared_with="item_id", padding_idx=0),
]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16, padding_idx=0)]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id"),
]
model = YoutubeDNN(
    user_features=user_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, 16], "dropout": 0.2},
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | 必填 |
| item_features | list | 物品特征列表 | 必填 |
| neg_item_feature | list | 负样本 ID 序列，需使用 `pooling="concat"` 并与正样本 ID 共享 embedding | 必填 |
| user_params | dict | 用户塔网络参数 | 必填 |
| temperature | float | 温度参数 | 1.0 |

## 4. YoutubeSBC

YoutubeSBC 在双塔打分时引入物品的采样概率。热门电影更容易出现在同一个批次里，同一批物品并不等于从片库中均匀抽样。

### 论文引用

```
Wu, Liang, et al. "RecSys 2019 tutorial: Deep learning for recommendations." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

模型先算用户和物品的余弦相似度，再从每列得分里减去该物品采样权重的对数。使用这个模型需要先准备 `sample_weight_feature`。

### 使用方法

```python
from torch_rechub.models.matching import YoutubeSBC

# 创建模型
model = YoutubeSBC(
    user_features=user_features,
    item_features=item_features,
    sample_weight_feature=sample_weight_feature,
    user_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    item_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    batch_size=256,
    n_neg=3,
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | 必填 |
| item_features | list | 物品特征列表 | 必填 |
| sample_weight_feature | list | 用于采样偏差校正的物品采样权重特征 | 必填 |
| user_params | dict | 用户塔网络参数 | 必填 |
| item_params | dict | 物品塔网络参数 | 必填 |
| batch_size | int | 必须与 DataLoader 的 batch size 一致 | 必填 |
| n_neg | int | 每个正样本的 batch 内负样本数，必须小于 batch size | 3 |
| temperature | float | 温度参数 | 1.0 |

## 5. MIND

MIND 为一个用户保留多个兴趣向量。例如，同一人常看动画也常看悬疑片，可以用多个向量分别检索候选，再合并电影列表。

### 论文引用

```
Chen, Jiaxi, et al. "Multi-interest network with dynamic routing for recommendation at Tmall." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

动态路由把观看历史聚合成多个兴趣向量。训练时目标物品参与挑选或聚合与它相关的兴趣，推理时保留用户的多个兴趣向量。

### 使用方法

```python
from torch_rechub.models.matching import MIND

user_features = [SparseFeature("user_id", vocab_size=10000, embed_dim=16)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id")
]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16)]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id")
]

# 创建模型
model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    interest_num=4,
    temperature=0.02
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 非序列用户特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史物品序列 | 必填 |
| item_features | list | 正样本物品 ID 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 ID 序列 | 必填 |
| max_length | int | 输入历史序列的固定长度 | 必填 |
| interest_num | int | 学习的兴趣数量 | 4 |
| temperature | float | 温度参数 | 1.0 |

## 6. GRU4Rec

GRU4Rec 按发生顺序读入一串观看记录。前几次看的是哪几部、最后看的是哪一部，都会影响最终的用户向量。

### 论文引用

```
Hidasi, Balázs, et al. "Session-based recommendations with recurrent neural networks." arXiv preprint arXiv:1511.06939 (2015).
```

本实现把 GRU 的最后一个隐状态与用户属性拼接后送入 MLP。MovieLens 示例将正电影放在第一列，训练器在正电影和预先采好的负电影之间计算交叉熵。

### 使用方法

```python
from torch_rechub.basic.features import SequenceFeature, SparseFeature
from torch_rechub.models.matching import GRU4Rec

user_features = [SparseFeature("user_id", vocab_size=10000, embed_dim=16)]
item_features = [SparseFeature("item_id", vocab_size=100000, embed_dim=16)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id"),
]
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=100000, embed_dim=16,
                    pooling="concat", shared_with="item_id"),
]
model = GRU4Rec(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    user_params={"dims": [128, 64, 16], "dropout": 0.2},
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | 必填 |
| history_features | list | `pooling="concat"` 的历史序列；当前实现只使用第一个序列 | 必填 |
| item_features | list | 正样本物品 ID 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 ID 序列 | 必填 |
| user_params | dict | 用户 MLP 参数，最后一层需与 item embedding 维度一致 | 必填 |
| temperature | float | 温度参数 | 1.0 |

GRU4Rec 沿嵌入维计算用户与每个候选的点积，每行输出正物品和负物品的分数。

## 7. NARM

NARM 用两种信息描述一次会话：GRU 的最后一个状态概括整段历史，注意力再从各步状态里挑出与这次会话更相关的部分。

### 论文引用

```
Li, Jing, et al. "Neural attentive session-based recommendation." Proceedings of the 2017 ACM on Conference on Information and Knowledge Management. 2017.
```

两部分拼接成会话向量，再投影到物品向量的维度，给候选物品打分。

### 使用方法

```python
from torch_rechub.models.matching import NARM

# 创建模型
model = NARM(
    item_history_feature=item_history_feature,
    hidden_dim=64,
    emb_dropout_p=0.2,
    session_rep_dropout_p=0.2,
    item_feature=item_feature,  # 仅在 batch 内负采样/物品塔推理时需要
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | 会话 item 序列，0 为 padding | 必填 |
| hidden_dim | int | GRU 隐藏维度 | 必填 |
| emb_dropout_p | float | item embedding dropout | 必填 |
| session_rep_dropout_p | float | 会话表示 dropout | 必填 |
| item_feature | SparseFeature or None | 可选的目标 item ID，用于独立 item tower/批内负采样 | None |

## 8. SASRec

SASRec 用自注意力读取按时间排列的物品 ID。每个位置只能看到当前位置及更早的行为，模型据此预测下一个物品。

### 论文引用

```
Kang, Wang-Cheng, and Julian McAuley. "Self-attentive sequential recommendation." 2018 IEEE International Conference on Data Mining (ICDM). IEEE, 2018.
```

位置嵌入区分行为的先后。训练时有历史序列和相应的正负目标序列，推理时从用户的历史得到一个向量。

### 使用方法

```python
from torch_rechub.models.matching import SASRec

# 创建模型
model = SASRec(
    features=[seq_feature, pos_feature, neg_feature],
    max_len=50,
    dropout_rate=0.2,
    num_blocks=2,
    num_heads=4,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 顺序固定为历史序列、正样本序列、负样本序列，同为 `SequenceFeature` | 必填 |
| max_len | int | 序列长度 | 50 |
| dropout_rate | float | Embedding/注意力/FFN dropout | 0.5 |
| num_blocks | int | Self-Attention block 数 | 2 |
| num_heads | int | 注意力头数，需整除 embedding 维度 | 1 |
| item_feature | SparseFeature or None | 实验性的独立 item tower 特征 | None |

SASRec 在序列所在设备上构造掩码与位置索引。配置 `item_feature` 后可以单独提取目标物品向量。

## 9. SINE

SINE 从所有用户共用的一组兴趣原型里，为当前用户选出少数几个。这里的“稀疏”指选中的原型少。

### 论文引用

```
Chen, Jiaxi, et al. "SINE: A sparse interest network for sequential recommendation." Proceedings of the 14th ACM Conference on Recommender Systems. 2021.
```

`num_concept` 是原型总数，`num_intention` 是每个用户选择的原型数。模型根据历史与原型的匹配程度选择原型，最后聚合成一个用户向量。

### 使用方法

```python
from torch_rechub.models.matching import SINE

# 创建模型
model = SINE(
    history_features=["hist_item_id"],
    item_features=["item_id"],
    neg_item_features=["neg_items"],
    num_items=100000,
    embedding_dim=32,
    hidden_dim=64,
    num_concept=100,
    num_intention=4,
    seq_max_len=50,
    num_heads=4,
    temperature=0.02
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| history_features | list[str] | 历史 item 序列字段名；当前只使用第一个 | 必填 |
| item_features | list[str] | 正样本 item 字段名 | 必填 |
| neg_item_features | list[str] | 负样本 item 序列字段名 | 必填 |
| num_items | int | 包含 padding 的 item 词表大小 | 必填 |
| embedding_dim | int | Embedding 维度 | 必填 |
| hidden_dim | int | 注意力隐藏维度 | 必填 |
| num_concept | int | 全局概念原型数 | 必填 |
| num_intention | int | 每个用户选取的意图数 | 必填 |
| seq_max_len | int | 输入固定序列长度 | 必填 |
| num_heads | int | 自注意力头数 | 1 |
| temperature | float | 温度参数 | 1.0 |

## 10. STAMP

STAMP 根据一次会话中的点击预测下一个物品。它单独取出用户最后点击的物品，让最近一次选择也参与打分。

### 论文引用

```
Liu, Qiao, et al. "STAMP: short-term attention/memory priority model for session-based recommendation." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

模型同时使用整段历史的汇总信息和最后一次点击，得到会话向量，再与物品向量计算得分。

### 使用方法

```python
from torch_rechub.models.matching import STAMP

# 创建模型
model = STAMP(
    item_history_feature=item_history_feature,
    weight_std=0.05,
    emb_std=0.05,
    item_feature=item_feature,  # 仅在 batch 内负采样/物品塔推理时需要
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| item_history_feature | SequenceFeature | 会话 item 序列，0 为 padding | 必填 |
| weight_std | float | 注意力参数初始化标准差 | 必填 |
| emb_std | float | Embedding/线性层初始化标准差 | 必填 |
| item_feature | SparseFeature or None | 可选的目标 item ID，用于 item tower/批内负采样 | None |

## 11. ComirecDR

ComirecDR 用动态路由把一次次交互分配给不同兴趣。一个用户可以保留多个向量，避免把几种偏好都压在同一个向量里。

### 论文引用

```
Chen, Jiaxi, et al. "Controllable multi-interest framework for recommendation." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.
```

同一组物品向量可以供多个兴趣检索。兴趣向量数越多，每个用户需要发起的检索次数也越多。

### 使用方法

```python
from torch_rechub.models.matching import ComirecDR

# 创建模型
model = ComirecDR(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,
    interest_num=4,
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 非序列用户特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史 item 序列 | 必填 |
| item_features | list | 正样本 item 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 item 序列 | 必填 |
| max_length | int | 固定历史序列长度 | 必填 |
| interest_num | int | 兴趣数量 | 4 |
| temperature | float | 温度参数 | 1.0 |

## 12. ComirecSA

ComirecSA 与 ComirecDR 都保留多个兴趣向量。SA 版本用多组注意力权重汇总历史，每组权重对应一个兴趣。

例如，两组权重可以分别关注观看记录中的不同电影。两组是否学出不同内容，取决于训练数据。

### 使用方法

```python
from torch_rechub.models.matching import ComirecSA

# 创建模型
model = ComirecSA(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    interest_num=4,
    temperature=0.02,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 非序列用户特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史 item 序列 | 必填 |
| item_features | list | 正样本 item 特征 | 必填 |
| neg_item_feature | list | `pooling="concat"` 的负样本 item 序列 | 必填 |
| interest_num | int | 兴趣数量 | 4 |
| temperature | float | 温度参数 | 1.0 |

## 13. 模型比较

| 训练样本里有什么 | 可以先看 | 读代码时关注什么 |
| --- | --- | --- |
| 一行用户、电影和 0/1 标签 | DSSM | 用户字段与物品字段怎样分到两个塔 |
| 同一用户的正电影和负电影 | FaceBookDSSM、YoutubeDNN | 前者比较两个分数，后者在候选列表中找正例 |
| 按时间排列的点击记录 | GRU4Rec、NARM、STAMP、SASRec | 模型怎样利用行为发生的顺序和最近一次点击 |
| 想从同一段历史检索多组候选 | MIND、ComirecDR、ComirecSA | 一个用户输出几个向量，以及如何合并各自的结果 |

## 14. 从训练到检索

在[召回训练教程](/zh/tutorials/retrieval)中，MovieLens 的用户与物品字段均用 16 维稀疏嵌入，两个塔的最后一层均输出 64 维向量。使用教程中的同一组数据，可以比较默认 DSSM 和开启批内采样时的 Top-K 结果。

## 15. 如何看 Top-K 结果

假设一个用户有两部留出的电影，前十个候选里找回一部。这名用户的 Recall@10 是 1/2，因为两部相关电影找回了一部。Precision@10 是 1/10，因为十个候选里只有一部相关。Hit@10 则是 1，只要命中任何一部就算命中。想区分“排第一”和“排第十”，再看 NDCG@10。

## 16. 保存物品索引

建立索引时把电影向量按行写入，同时按相同顺序保存原始电影 ID。检索得到行号后，用这份 ID 列表查回电影。更新模型权重后需重新计算电影向量，并用新向量重建索引。[向量索引](/zh/serving/vector_index)
