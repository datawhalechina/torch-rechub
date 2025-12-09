---
title: 排序模型
description: Torch-RecHub 排序模型详细介绍
---

# 排序模型

排序模型是推荐系统中的核心组件，用于预测用户对物品的点击率或偏好分数，从而对召回的候选集进行精排序。Torch-RecHub 提供了多种先进的排序模型，涵盖了不同的特征处理和建模方式。

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
    deep_features=sparse_features + dense_features,
    cross_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    cross_num_layers=3
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| cross_features | list | Cross 部分使用的特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |
| cross_num_layers | int | Cross Network 的层数 | 3 |

### 适用场景

- 需要显式特征交叉的场景
- 计算资源有限的场景
- 点击率预测任务

## 4. DCNv2

### 功能描述

DCNv2 是 DCN 的增强版本，引入了特征选择单元和动态缩放机制，进一步提高了模型的表达能力和效率。

### 论文引用

```
Wang, Ruoxi, et al. "DCN V2: Improved deep & cross network and practical lessons for web-scale learning to rank systems." Proceedings of the web conference 2021. 2021.
```

### 核心原理

- **特征选择单元**：为每个特征分配动态权重，自动选择重要特征
- **动态缩放机制**：引入标量参数，自适应调整特征交叉的贡献
- **更灵活的交叉网络**：支持不同的交叉形式

### 使用方法

```python
from torch_rechub.models.ranking import DCNv2

# 创建模型
model = DCNv2(
    deep_features=sparse_features + dense_features,
    cross_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    cross_num_layers=3
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| cross_features | list | Cross 部分使用的特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |
| cross_num_layers | int | Cross Network 的层数 | 3 |

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
    deep_features=sparse_features + dense_features,
    cross_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    cross_num_layers=3
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| cross_features | list | Cross 部分使用的特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |
| cross_num_layers | int | Cross Network 的层数 | 3 |

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
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    attention_params={"attention_dim": 64, "dropout": 0.2}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| fm_features | list | FM 部分使用的特征列表 | None |
| attention_params | dict | 注意力网络参数 | None |

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

# 创建模型
model = DeepFFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)

# 创建 FatDeepFFM 模型（增强版本）
fat_model = FatDeepFFM(
    deep_features=sparse_features + dense_features,
    fm_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| fm_features | list | FFM 部分使用的特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |

### 适用场景

- 场感知特征交互重要的场景
- 复杂特征交互场景
- 点击率预测任务

## 9. BST

### 功能描述

BST（Behavior Sequence Transformer）是一种使用 Transformer 建模用户行为序列的模型，能够捕获用户行为序列中的长距离依赖关系。

### 论文引用

```
Sun, Fei, et al. "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer." Proceedings of the 28th ACM international conference on information and knowledge management. 2019.
```

### 核心原理

- **Transformer Encoder**：使用多头自注意力机制捕获序列中的依赖关系
- **位置编码**：添加位置信息，保留序列的顺序信息
- **特征融合**：将序列特征与其他特征融合，得到最终的预测结果

### 使用方法

```python
from torch_rechub.models.ranking import BST

# 定义序列特征
sequence_features = [SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling="mean")]

# 创建模型
model = BST(
    deep_features=sparse_features + dense_features,
    sequence_features=sequence_features,
    transformer_params={"num_heads": 4, "num_layers": 2, "hidden_size": 128, "dropout": 0.2}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| sequence_features | list | 序列特征列表 | None |
| transformer_params | dict | Transformer 参数 | None |

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
from torch_rechub.models.ranking import DIN

# 定义序列特征
sequence_features = [SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling=None)]

# 创建模型
model = DIN(
    deep_features=sparse_features + dense_features,
    sequence_features=sequence_features,
    target_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| sequence_features | list | 序列特征列表 | None |
| target_features | list | 目标特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |

### 适用场景

- 用户兴趣动态变化的场景
- 目标物品相关的兴趣建模
- 点击率预测任务

## 11. DIEN

### 功能描述

DIEN（Deep Interest Evolution Network）是一种用于建模用户兴趣演化过程的深度网络，能够捕获用户兴趣的动态变化。

### 论文引用

```
Zhou, Guorui, et al. "Deep interest evolution network for click-through rate prediction." Proceedings of the AAAI conference on artificial intelligence. 2019.
```

### 核心原理

- **GRU**：使用 GRU 捕捉用户兴趣的时序变化
- **兴趣抽取层**：从原始行为序列中提取兴趣序列
- **兴趣演化层**：使用 GRU 和注意力机制建模兴趣的动态演化
- **兴趣激活层**：根据目标物品激活相关兴趣

### 使用方法

```python
from torch_rechub.models.ranking import DIEN

# 定义序列特征
sequence_features = [SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling=None)]

# 创建模型
model = DIEN(
    deep_features=sparse_features + dense_features,
    sequence_features=sequence_features,
    target_features=sparse_features,
    mlp_params={"dims": [256, 128, 64], "dropout": 0.2, "activation": "relu"},
    dien_params={"gru_layers": 2, "attention_dim": 64, "dropout": 0.2}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| sequence_features | list | 序列特征列表 | None |
| target_features | list | 目标特征列表 | None |
| mlp_params | dict | 深度神经网络参数 | None |
| dien_params | dict | DIEN 网络参数 | None |

### 适用场景

- 用户兴趣动态演化的场景
- 长序列兴趣建模
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
    deep_features=sparse_features + dense_features,
    attention_params={"num_heads": 4, "num_layers": 2, "hidden_size": 128, "dropout": 0.2}
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| deep_features | list | Deep 部分使用的特征列表 | None |
| attention_params | dict | 注意力网络参数 | None |

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
    SparseFeature(name="gender", vocab_size=3, embed_dim=8),
    SparseFeature(name="occupation", vocab_size=20, embed_dim=12)
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
    device="cuda:0",
    model_path="saved/deepfm"
)

# 6. 训练模型
trainer.fit(train_dl, val_dl)

# 7. 评估模型
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc}")

# 8. 导出 ONNX 模型
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
A: 可以尝试以下方法：
- 使用 GPU 训练
- 增加 batch size
- 使用混合精度训练
- 选择计算效率高的模型
- 数据并行训练