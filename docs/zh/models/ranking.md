---
title: 排序模型
description: Torch-RecHub 排序模型详细介绍
---

# 排序模型

排序模型是推荐系统中的核心组件，用于预测用户对物品的点击率或偏好分数，从而对召回的候选集进行精排序。Torch-RecHub 提供了多种先进的排序模型，涵盖了不同的特征处理和建模方式。

## 批内负采样：统一训练入口

所有 13 个公开排序模型都可使用同一个入口，无需修改模型构造参数、默认输出或推理代码：

```python
trainer = CTRTrainer(model, in_batch_neg=True,
                     item_features=["item_id", "cate_id"])
```

不开启时，仍按原来的标签和 BCE 训练。开启时，Trainer 只使用 batch 中 `y=1` 的正交互：保留本行用户、上下文和历史，将其他正交互的物品字段一起换入，生成标签为 0 的负配对，再沿用模型默认前向和 BCE。训练标签必须是二分类标签，不能像召回配对那样用全 0 占位。

`item_features` 接受列名或已有 Feature 对象。必须包含物品 ID 及所有相关物品属性，不能包含用户或历史字段；预计算的用户—物品交叉特征需要重新计算，不能直接用于该简单配对入口。匿名字段（例如 Criteo 的 C1/C2）不能自动推断归属，须由使用者根据真实数据定义指定。

默认 `in_batch_neg_ratio=1`，即每个正样本随机取一个其他行的物品，避免排序模型计算整个 `B × B` 组合的开销。可设置正整数调整数量，或设置 `None` 使用所有其他行；随机采样有放回，排除本行但不屏蔽重复物品或其他已点击物品，因此仍可能产生假负样本。`sampler_seed` 可控制采样随机性。batch 中不足两条正交互时会跳过并提示，整个 epoch 都无法配对时会报错。

DIN/BST 的候选物品注意力会通过原前向重新计算。DIEN 沿用 `loss_mode=False` 和原有辅助历史损失；本开关改变目标物品负采样，不取消 DIEN 辅助任务的负历史输入。

Amazon DIN/DIEN 示例只需添加 `--in_batch_neg`。其他单任务排序示例需同时指定 `--item_features`；其已有标签在训练时用于筛选正交互。使用 `generate_seq_feature(..., in_batch_neg=True)` 可以生成纯正样本训练集，验证和测试仍保留原有正负标签。

验证、预测、导出及原模型权重结构不变。批内生成的负例是训练假设，不是实际曝光未点击记录；评估仍应使用真实标签，训练输出也不能直接视为经过校准的业务点击率。多任务模型和 `MTLTrainer` 不属于此单任务排序入口。

## 1. WideDeep

### 功能描述

WideDeep 是一种结合了线性模型（Wide 部分）和深度神经网络（Deep 部分）的混合模型，旨在同时利用线性模型的记忆能力和深度模型的泛化能力。

### 论文引用

```
Cheng, Heng-Tze, et al. "Wide & deep learning for recommender systems." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.
```

### 核心原理

- **Wide 部分**：线性模型，使用交叉特征，擅长捕获记忆效应
- **Deep 部分**：深度神经网络，使用 Embedding 层和全连接层，擅长捕获泛化效应
- **联合训练**：Wide 部分和 Deep 部分同时训练，输出结果通过 sigmoid 函数结合

### 使用方法

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import WideDeep

# 定义特征
dense_features = [DenseFeature(name="age", embed_dim=1), DenseFeature(name="income", embed_dim=1)]
sparse_features = [SparseFeature(name="city", vocab_size=100, embed_dim=16), SparseFeature(name="gender", vocab_size=3, embed_dim=8)]

# 创建模型
model = WideDeep(
    wide_features=sparse_features,
    deep_features=sparse_features + dense_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| wide_features | list | Wide 部分使用的特征列表 | None |
| deep_features | list | Deep 部分使用的特征列表 | None |
| mlp_params | dict | 深度神经网络参数，包含 dims、dropout、activation 等 | None |

### 适用场景

- 基础排序任务
- 需要同时利用记忆和泛化能力的场景
- 特征工程资源有限的场景

## 2. DeepFM

### 功能描述

DeepFM 是一种结合了因子分解机（FM）和深度神经网络的模型，能够同时捕获低阶和高阶特征交互。

### 论文引用

```
Guo, Huifeng, et al. "DeepFM: a factorization-machine based neural network for CTR prediction." Proceedings of the 26th international joint conference on artificial intelligence. 2017.
```

### 核心原理

- **FM 部分**：捕获二阶特征交互，具有线性复杂度
- **Deep 部分**：通过神经网络捕获高阶特征交互
- **共享 Embedding**：FM 部分和 Deep 部分共享特征 Embedding，减少参数数量

### 使用方法

```python
from torch_rechub.models.ranking import DeepFM

# 创建模型
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| fm_features | list | FM 部分使用的特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |

### 适用场景

- 特征交互重要的场景
- 需要同时捕获低阶和高阶特征交互的场景
- 点击率预测任务

## 3. DCN

### 功能描述

DCN（Deep & Cross Network）是一种显式学习特征交叉的模型，通过交叉网络（Cross Network）显式捕获高阶特征交互，同时保持线性的计算复杂度。

### 论文引用

```
Wang, Ruoxi, et al. "Deep & cross network for ad click predictions." Proceedings of the ADKDD'17. 2017.
```

### 核心原理

- **Cross Network**：显式学习高阶特征交叉，每层输出为：
  $$x_{l+1} = x_0 x_l^T w_l + b_l + x_l$$
- **Deep Network**：深度神经网络，捕获非线性特征交互
- **联合训练**：Cross Network 和 Deep Network 并行计算，结果拼接后通过全连接层输出

### 使用方法

```python
from torch_rechub.models.ranking import DCN

# 创建模型
model = DCN(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | Cross 与 Deep 部分共用的特征列表 | 必填 |
| mlp_params | dict | 深度神经网络参数 | None |
| n_cross_layers | int | Cross Network 的层数 | 必填 |

### 适用场景

- 需要显式特征交叉的场景
- 计算资源有限的场景
- 点击率预测任务

## 4. DCNv2

### 功能描述

DCNv2 是 DCN 的增强版本，将 Cross Network 的标量/向量参数扩展为矩阵交互；本项目同时支持低秩专家混合，以降低完整矩阵交互的成本。

### 论文引用

```
Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021.
```

### 核心原理

- **矩阵式交叉**：比 DCN 的向量式交叉表达力更强
- **低秩专家混合**：`use_low_rank_mixture=True` 时使用多个低秩 Cross 专家
- **结构可选**：支持 `crossnet_only`、`stacked` 和 `parallel`

### 使用方法

```python
from torch_rechub.models.ranking import DCNv2

# 创建模型
model = DCNv2(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    model_structure="parallel",       # crossnet_only / stacked / parallel
    use_low_rank_mixture=True,
    low_rank=32,
    num_experts=4,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | Cross 与 Deep 部分共用的特征列表 | 必填 |
| mlp_params | dict | 深度神经网络参数 | None |
| n_cross_layers | int | Cross Network 的层数 | 必填 |
| model_structure | str | `crossnet_only` / `stacked` / `parallel` | `parallel` |
| use_low_rank_mixture | bool | 是否使用低秩专家混合 CrossNet | True |
| low_rank | int | 低秩维度 | 32 |
| num_experts | int | CrossNetMix 专家数 | 4 |

### 适用场景

- 需要更高效特征交叉的场景
- 大规模推荐系统
- 点击率预测任务

## 5. EDCN

### 功能描述

EDCN（Enhanced Deep & Cross Network）是一种增强型的交叉网络模型，引入了显式特征交叉和深度特征提取的结合，进一步提高了模型的表达能力。

### 论文引用

```
Ma, Xiao, et al. "Enhanced Deep & Cross Network for Feature Cross Learning in Click-Through Rate Prediction." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

### 核心原理

- **Cross Network**：显式学习高阶特征交叉
- **Deep Network**：深度神经网络，捕获非线性特征交互
- **特征重要性学习**：引入特征重要性权重，提高模型的解释性

### 使用方法

```python
from torch_rechub.models.ranking import EDCN

# 创建模型
model = EDCN(
    features=sparse_features + dense_features,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    bridge_type="hadamard_product",
    use_regulation_module=True,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 模型使用的全部特征 | 必填 |
| mlp_params | dict | 深度神经网络参数 | None |
| n_cross_layers | int | Cross Network 的层数 | 必填 |
| bridge_type | str | Cross/Deep 桥接方式 | `hadamard_product` |
| use_regulation_module | bool | 是否使用特征调节模块 | True |

### 适用场景

- 复杂特征交互场景
- 需要高表达能力的模型
- 点击率预测任务

## 6. AFM

### 功能描述

AFM（Attention Factorization Machine）是一种基于注意力机制的因子分解机，能够自适应地学习不同特征交互的重要性。

### 论文引用

```
Xiao, Jun, et al. "Attentional factorization machines: Learning the weight of feature interactions via attention networks." arXiv preprint arXiv:1708.04617 (2017).
```

### 核心原理

- **FM 基础**：基于因子分解机，捕获二阶特征交互
- **注意力机制**：引入注意力网络，为每个特征交互分配动态权重
- **注意力输出**：注意力权重与特征交互向量加权求和，得到最终的交互向量

### 使用方法

```python
from torch_rechub.models.ranking import AFM

# 创建模型
model = AFM(
    fm_features=sparse_features,
    embed_dim=16,  # 必须与 fm_features 的 embedding 维度一致
    t=64,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| fm_features | list | 参与 FM 交互的稀疏特征，嵌入维度需一致 | 必填 |
| embed_dim | int | 特征嵌入维度 | 必填 |
| t | int | 注意力隐藏层维度 | 64 |

### 适用场景

- 特征交互重要性差异较大的场景
- 需要解释性的场景
- 点击率预测任务

## 7. FiBiNET

### 功能描述

FiBiNET（Feature Importance and Bilinear feature Interaction NETwork）是一种结合了特征重要性学习和双线性特征交互的模型，能够更有效地捕获特征交互。

### 论文引用

```
Juan, Yuchin, et al. "FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction." Proceedings of the 13th ACM Conference on Recommender Systems. 2019.
```

### 核心原理

- **特征重要性网络**：通过 Squeeze-and-Excitation 机制学习特征重要性
- **双线性交互**：使用双线性函数捕获特征交互，支持不同的交互形式
- **特征增强**：对输入特征进行增强，提高模型的表达能力

### 使用方法

```python
from torch_rechub.models.ranking import FiBiNet

# 创建模型
model = FiBiNet(
    features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    reduction_ratio=3,
    bilinear_type="field_interaction",
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 参与 SENET 和双线性交互的特征 | 必填 |
| mlp_params | dict | 预测 MLP 参数 | 必填 |
| reduction_ratio | int | SENET 压缩比 | 3 |
| bilinear_type | str | `field_all` / `field_each` / `field_interaction` | `field_interaction` |

### 适用场景

- 特征重要性差异较大的场景
- 需要复杂特征交互的场景
- 点击率预测任务

## 8. DeepFFM

### 功能描述

DeepFFM（Deep Field-aware Factorization Machine）是一种结合了场感知因子分解机和深度神经网络的模型，能够捕获场感知的高阶特征交互。

### 论文引用

```
Xiao, Jun, et al. "Deep learning over multi-field categorical data." European conference on information retrieval. Springer, Cham, 2016.
```

### 核心原理

- **FFM 基础**：场感知因子分解机，为每个特征场对学习特定的交互向量
- **Deep Network**：深度神经网络，捕获高阶特征交互
- **联合训练**：FFM 部分和 Deep 部分联合训练，输出结果结合

### 使用方法

```python
from torch_rechub.models.ranking import DeepFFM, FatDeepFFM

# FFM 会为每个字段对使用不同 embedding，cross feature 的词表需乘以字段数
ffm_linear_features = [
    SparseFeature(f.name, vocab_size=f.vocab_size, embed_dim=1) for f in sparse_features
]
ffm_cross_features = [
    SparseFeature(f.name, vocab_size=f.vocab_size * len(sparse_features), embed_dim=10)
    for f in sparse_features
]

model = DeepFFM(
    linear_features=ffm_linear_features,
    cross_features=ffm_cross_features,
    embed_dim=10,
    mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"},
)

# 创建 FatDeepFFM 模型（增强版本）
fat_model = FatDeepFFM(
    linear_features=ffm_linear_features,
    cross_features=ffm_cross_features,
    embed_dim=10,
    reduction_ratio=1,
    mlp_params={"dims": [1600, 1600], "dropout": 0.5, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| linear_features | list | 一阶线性部分特征，通常 `embed_dim=1` | 必填 |
| cross_features | list | FFM 交互特征，词表需预留字段偏移空间 | 必填 |
| embed_dim | int | FFM embedding 维度 | 必填 |
| reduction_ratio | int | 仅 FatDeepFFM 需要的 CEN 压缩比 | 必填（FatDeepFFM） |
| mlp_params | dict | FFM 交互后的 MLP 参数 | 必填 |

### 适用场景

- 场感知特征交互重要的场景
- 复杂特征交互场景
- 点击率预测任务

## 9. BST

### 功能描述

BST（Behavior Sequence Transformer）是一种使用 Transformer 建模用户行为序列的模型，能够捕获用户行为序列中的长距离依赖关系。

### 论文引用

```
Chen, Qiwei, et al. "Behavior Sequence Transformer for E-commerce Recommendation in Alibaba." arXiv preprint arXiv:1905.06874 (2019).
```

### 核心原理

- **Transformer Encoder**：使用多头自注意力机制捕获序列中的依赖关系
- **位置编码**：添加位置信息，保留序列的顺序信息
- **特征融合**：将序列特征与其他特征融合，得到最终的预测结果

### 使用方法

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import BST

features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id"),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8,
                    pooling="concat", shared_with="target_cate_id"),
]

model = BST(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    nhead=8,
    dropout=0.2,
    num_layers=1,
    max_seq_len=51,
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 用户画像/上下文特征，不含历史和目标 | 必填 |
| history_features | list | `pooling="concat"` 的历史序列 | 必填 |
| target_features | list | 与历史特征对应的目标物品特征 | 必填 |
| mlp_params | dict | 预测 MLP 参数 | 必填 |
| nhead | int | 多头注意力头数，需整除历史特征维度总和 | 8 |
| dropout | float | Transformer dropout | 0.2 |
| num_layers | int | Transformer Encoder 层数 | 1 |
| max_seq_len | int | 历史长度 + 1 个 target 的上限 | 51 |

### 适用场景

- 用户行为序列重要的场景
- 长序列建模场景
- 推荐系统中的顺序推荐任务

## 10. DIN

### 功能描述

DIN（Deep Interest Network）是一种基于注意力机制的深度兴趣网络，能够根据目标物品动态捕获用户的兴趣。

### 论文引用

```
Zhou, Guorui, et al. "Deep interest network for click-through rate prediction." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

### 核心原理

- **兴趣提取**：从用户行为序列中提取兴趣表示
- **注意力机制**：根据目标物品计算每个历史行为的注意力权重
- **兴趣动态聚合**：根据注意力权重动态聚合用户兴趣，得到最终的兴趣表示

### 使用方法

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIN

features = [SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8)]
target_features = [SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8)]
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id")
]

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    attention_mlp_params={"dims": [64, 32], "use_softmax": False},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 用户画像/上下文特征 | 必填 |
| history_features | list | `pooling="concat"` 的历史序列 | 必填 |
| target_features | list | 与历史特征数量、顺序和维度对应的目标特征 | 必填 |
| mlp_params | dict | 预测 MLP 参数 | 必填 |
| attention_mlp_params | dict | Activation Unit 参数 | 必填 |

### 适用场景

- 用户兴趣动态变化的场景
- 目标物品相关的兴趣建模
- 点击率预测任务

## 11. DIEN

### 功能描述

DIEN（Deep Interest Evolution Network）是阿里妈妈在 AAAI'2019 提出的模型，在 DIN 基础上引入**兴趣抽取层**（GRU + 辅助损失）和**兴趣演化层**（AUGRU），建模用户兴趣随时间的动态演化过程。

### 论文引用

```
Zhou, Guorui, et al. "Deep interest evolution network for click-through rate prediction." Proceedings of the AAAI conference on artificial intelligence. 2019.
```

### 核心原理

- **兴趣抽取层**：GRU 对行为序列建模，辅助损失用正负样本对监督每步隐状态（论文 Eq.7）
- **兴趣演化层**：AUGRU 将注意力分数嵌入更新门，对全序列 softmax 归一化后逐步演化（论文 Eq.14-16）
- **辅助损失**：$L_{aux} = -\frac{1}{N}\sum[\log\sigma(h_t \cdot e^+_{t+1}) + \log(1-\sigma(h_t \cdot e^-_{t+1}))]$
- **Padding 处理**：padding 位（index=0）不参与 GRU/AUGRU 计算；空历史样本保持零隐状态

### 使用方法

```python
from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.ranking import DIEN

# target_features 必须设 padding_idx=0，因为 history/neg_history 共享其 embedding 表
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items+1, embed_dim=8, padding_idx=0),
]
# history/neg_history 通过 shared_with 指向 target feature（embed_dict 的 root key）
history_features = [
    SequenceFeature("hist_item_id", vocab_size=n_items+1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
]
neg_history_features = [
    SequenceFeature("neg_hist_item_id", vocab_size=n_items+1, embed_dim=8,
                    pooling="concat", shared_with="target_item_id", padding_idx=0),
]

model = DIEN(
    features=features,
    history_features=history_features,
    neg_history_features=neg_history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    alpha=0.2,
)
# CTRTrainer 需设 loss_mode=False，因为 forward 返回 (prediction, aux_loss)
```

### 参数说明

| 参数 | 类型 | 描述 |
| --- | --- | --- |
| features | list | 用户画像 / 上下文特征，输入顶层 MLP |
| history_features | list | 正样本行为序列，pooling="concat"，padding_idx=0，shared_with=target_feature |
| neg_history_features | list | 负采样行为序列，同上，shared_with 必须指向 target feature（非 history feature） |
| target_features | list | 目标物品特征，padding_idx=0，用于 AUGRU 注意力 |
| mlp_params | dict | 顶层 MLP 参数，activation 固定为 dice |
| alpha | float | 辅助损失权重，默认 0.2 |

### 适用场景

- 用户兴趣随时间动态演化的场景（电商、新闻推荐）
- 拥有带时序的用户行为序列数据
- 点击率预测任务

## 12. AutoInt

### 功能描述

AutoInt（Automatic Feature Interaction Learning via Self-Attentive Neural Networks）是一种使用自注意力机制自动学习特征交互的模型，能够灵活地捕获各种阶数的特征交互。

### 论文引用

```
Song, Weiping, et al. "AutoInt: Automatic Feature Interaction Learning via Self-Attentive Neural Networks." Proceedings of the 28th ACM International Conference on Information and Knowledge Management. 2019.
```

### 核心原理

- **Embedding 层**：将离散特征映射到低维向量空间
- **多头自注意力机制**：自动学习特征之间的交互关系
- **残差连接**：增强模型的训练稳定性
- **层归一化**：加速模型收敛

### 使用方法

```python
from torch_rechub.models.ranking import AutoInt

# 创建模型
model = AutoInt(
    sparse_features=sparse_features,
    dense_features=dense_features,
    num_layers=3,
    num_heads=2,
    dropout=0.2,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| sparse_features | list | 至少一个、且 `embed_dim` 相同的稀疏特征 | 必填 |
| dense_features | list | 连续特征，可为空列表 | 必填 |
| num_layers | int | Interacting Layer 层数 | 3 |
| num_heads | int | 注意力头数 | 2 |
| dropout | float | 注意力 dropout | 0.0 |
| mlp_params | dict or None | 可选 Deep 分支参数 | None |

### 适用场景

- 自动特征交互学习
- 复杂特征交互场景
- 点击率预测任务

## 13. 模型比较

| 模型 | 复杂度 | 表达能力 | 计算效率 | 解释性 |
| --- | --- | --- | --- | --- |
| WideDeep | 低 | 中 | 高 | 高 |
| DeepFM | 中 | 高 | 中 | 中 |
| DCN/DCNv2 | 中 | 高 | 高 | 中 |
| EDCN | 中 | 高 | 中 | 中 |
| AFM | 中 | 中 | 中 | 高 |
| FiBiNET | 中 | 高 | 中 | 中 |
| DeepFFM | 高 | 高 | 低 | 中 |
| BST | 高 | 高 | 低 | 中 |
| DIN | 中 | 高 | 中 | 中 |
| DIEN | 高 | 高 | 低 | 中 |
| AutoInt | 高 | 高 | 低 | 中 |

## 14. 使用建议

1. **根据数据规模选择模型**：小规模数据推荐使用简单模型（如 WideDeep、DeepFM），大规模数据可以尝试更复杂的模型
2. **根据特征类型选择模型**：序列特征重要时推荐使用 BST、DIN、DIEN；特征交互重要时推荐使用 DCN、DeepFM
3. **根据计算资源选择模型**：计算资源有限时推荐使用计算效率高的模型（如 DCN、WideDeep）
4. **尝试多种模型并进行融合**：不同模型可能捕获不同的特征交互模式，模型融合可以提高最终效果

## 15. 代码示例：完整的排序模型训练流程

```python
import os

from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import DenseFeature, SparseFeature

# 1. 定义特征
# 假设我们有以下特征
dense_features = [
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1)
]

sparse_features = [
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    SparseFeature(name="gender", vocab_size=3, embed_dim=16),
    SparseFeature(name="occupation", vocab_size=20, embed_dim=16)
]

# 2. 准备数据
# 假设 x 和 y 是已经处理好的特征和标签数据
x = {
    "age": age_data,
    "income": income_data,
    "city": city_data,
    "gender": gender_data,
    "occupation": occupation_data
}
y = label_data

# 3. 创建数据生成器
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

# 4. 创建模型
model = DeepFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# 5. 创建训练器
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cpu",
    model_path="saved/deepfm"
)

# 6. 训练模型
os.makedirs("saved/deepfm", exist_ok=True)
trainer.fit(train_dl, val_dl)

# 7. 评估模型
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc}")

# 8. 导出 ONNX 模型（先安装：pip install "torch-rechub[onnx]"）
trainer.export_onnx("deepfm.onnx")
```

## 16. 常见问题与解决方案

### Q: 如何选择合适的模型？
A: 根据数据规模、特征类型、计算资源和业务需求选择合适的模型。建议先从简单模型开始，逐步尝试更复杂的模型。

### Q: 模型训练过拟合怎么办？
A: 可以尝试以下方法：
- 增加正则化（L1/L2正则化）
- 增加 dropout 率
- 使用早停（Early Stopping）
- 增加训练数据
- 简化模型结构

### Q: 如何处理大规模特征？
A: 可以尝试以下方法：
- 特征选择：只保留重要特征
- 特征哈希：将高维特征映射到低维空间
- 分层 Embedding：对不同特征使用不同的 Embedding 维度

### Q: 如何加速模型训练？
A: 当前 `CTRTrainer` 可通过 `device` 使用单卡 GPU，也可通过 `gpus=[...]` 在单机上包装 `torch.nn.DataParallel`。训练器没有内置自动混合精度（AMP）或多机分布式训练。batch size 是否能增大取决于显存，建议先从较小值逐步测试。
