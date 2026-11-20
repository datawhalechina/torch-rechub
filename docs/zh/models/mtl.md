---
title: 多任务模型
description: Torch-RecHub 多任务模型详细介绍
---

# 多任务模型

同一次商品曝光可以带有多个标签，例如是否点击、是否购买。多任务模型读取这行样本，在不同的输出列预测不同目标。训练前先确定列的顺序，后面传给训练器的标签、任务类型和评估名称都按这个顺序填写。[Ali-CCP 多任务教程](/zh/tutorials/pipeline)

## 1. SharedBottom

SharedBottom 先把一行样本的特征交给一个共享的底层网络，再把同一份结果送给各任务自己的输出网络（任务塔）。这是比较多个目标时最简单的起点。

某个任务的损失会更新自己的任务塔和共享底层。`task_types` 的顺序决定各列的输出方式，分类任务输出概率，回归任务输出数值。

### 使用方法

```python
from torch_rechub.models.multi_task import SharedBottom
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 定义特征
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

# 创建模型
model = SharedBottom(
    features=common_features,
    task_types=["classification", "classification"],  # 两个分类任务
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 任务1的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 任务2的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | 必填 |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | 必填 |
| bottom_params | dict | 共享底层网络参数 | 必填 |
| tower_params_list | list | 任务特定顶层网络参数列表 | 必填 |

## 2. ESMM

ESMM 利用“先点击，再转化”的关系。两座塔分别预测点击概率（CTR）和点击后的转化概率（CVR），两者相乘就得到一次曝光既被点击又产生转化的概率（CTCVR）。

### 论文引用

```
Xiao, Jun, et al. "Entire space multi-task model: An effective approach for estimating post-click conversion rate." Proceedings of the 41st international ACM SIGIR conference on research & development in information retrieval. 2018.
```

两个塔共用输入特征的嵌入，各自学习不同的权重。框架的 ESMM 返回 `[cvr, ctr, ctcvr]` 三列，`MTLTrainer` 用第二列 CTR 和第三列 CTCVR 计算训练损失。

### 使用方法

```python
from torch_rechub.models.multi_task import ESMM

# 创建模型
model = ESMM(
    user_features=user_features,
    item_features=item_features,
    cvr_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    ctr_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户侧稀疏特征 | 必填 |
| item_features | list | 物品侧稀疏特征 | 必填 |
| cvr_params | dict | CVR tower 的 MLP 参数 | 必填 |
| ctr_params | dict | CTR tower 的 MLP 参数 | 必填 |

传给 `MTLTrainer` 的标签也按 `[cvr, ctr, ctcvr]` 排列，形状为 `[B, 3]`，`task_types` 填三个 `"classification"`。

## 3. MMOE

MMOE 让多个小网络（专家）各自处理同一份输入。每个任务都有自己的门控网络，决定当前样本应从各专家的输出里取多少。

### 论文引用

```
Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

例如预测点击和预测停留时长可以共用专家，但两组门控权重可以不同。`n_expert` 控制专家数，`tower_params_list` 按任务顺序配置各自的输出网络。

### 使用方法

```python
import os
import numpy as np

from torch_rechub.models.multi_task import MMOE

# 创建模型
model = MMOE(
    features=common_features,
    task_types=["classification", "regression"],  # 分类任务和回归任务
    n_expert=8,  # 专家网络数量
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 分类任务的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 回归任务的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | 必填 |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | 必填 |
| n_expert | int | 专家网络数量 | 必填 |
| expert_params | dict | 专家网络参数 | 必填 |
| tower_params_list | list | 任务特定顶层网络参数列表 | 必填 |

## 4. PLE

PLE 给每个任务保留专用专家，也保留一组共享专家。点击任务可以从自己的专家和共享专家中取信息，另一个任务有自己的选择。

### 论文引用

```
Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations." Fourteenth ACM Conference on Recommender Systems. 2020.
```

模型可以叠加多层专家与门控。比较 PLE 与 MMOE 时，分别看每个任务的验证指标，才能发现某个任务的提升是否伴随另一个任务的下降。

### 使用方法

```python
from torch_rechub.models.multi_task import PLE

# 创建模型
model = PLE(
    features=common_features,
    task_types=["classification", "regression"],
    n_level=2,  # PLE层数
    n_expert_specific=4,  # 每层任务特定专家数量
    n_expert_shared=4,  # 每层共享专家数量
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 分类任务的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 回归任务的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | 必填 |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | 必填 |
| n_level | int | PLE层数 | 必填 |
| n_expert_specific | int | 每层任务特定专家数量 | 必填 |
| n_expert_shared | int | 每层共享专家数量 | 必填 |
| expert_params | dict | 专家网络参数 | 必填 |
| tower_params_list | list | 任务特定顶层网络参数列表 | 必填 |

## 5. AITM

AITM 适合有先后顺序的二分类目标，例如点击、加购、购买。后一阶段可以使用前一阶段的表示，所以任务排列顺序也是模型配置的一部分。

### 论文引用

```
Tang, Jiaxi, et al. "Learning Task Relationships in Multi-task Learning with Adaptive Information Transfer." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

每个任务先由自己的底层网络计算表示，再通过注意力把前一任务传来的信息合进来。最后每个塔输出一个概率。

### 使用方法

```python
from torch_rechub.models.multi_task import AITM

# 创建模型
model = AITM(
    features=common_features,
    n_task=2,
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 任务1的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 任务2的顶层参数
    ],
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | 必填 |
| n_task | int | 有顺序依赖的二分类任务数 | 必填 |
| bottom_params | dict | 每个任务的 bottom MLP 参数 | 必填 |
| tower_params_list | list | 任务特定顶层网络参数列表 | 必填 |

AITM 当前所有任务都固定为二分类，并按任务列顺序从前一任务向后一任务传递信息；不接受 `task_types` 或 `attention_params`。

## 6. 模型比较

多个标签都针对同一次曝光时，可以先用 SharedBottom 建立基线，看看每个任务单独能学到什么程度。想让不同任务自己选择共享多少信息，再比较 MMOE 和 PLE。若要直接利用点击后才可能成交的关系，则看 ESMM。AITM 适合按明确的业务顺序传递信息。

## 7. 训练和检查标签

[Ali-CCP 多任务教程](/zh/tutorials/pipeline)给出了可运行的训练与评估示例。MMOE 和 PLE 使用 `[cvr, ctr]` 两列标签；ESMM 使用 `[cvr, ctr, ctcvr]` 三列，其中 `ctcvr = ctr * cvr`。按相同的顺序给 `evaluate` 返回的分数加上任务名，即可分别检查每个任务。

## 8. 怎样读多任务的验证结果

假设模型输出顺序是 `[cvr, ctr]`。如果打印时先标 CTR，就会把转化率的 AUC 当成点击率的 AUC。先检查验证标签、预测列、打印名称是否一一对应，再看两个任务各自是否改善。转化往往比点击少，还要检查验证集是否同时含有正例和负例。

两个任务的损失量级相差较大时，`MTLTrainer` 支持用 `adaptive_params={"method": "uwl"}` 学习损失权重。从默认权重开始，在同一份验证集上比较调整前后的各项指标。
