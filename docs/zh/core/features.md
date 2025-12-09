---
title: 特征定义
description: Torch-RecHub 特征类型定义
---

# 特征定义

Torch-RecHub提供了三种核心特征类，用于处理不同类型的特征：

## DenseFeature

处理数值型特征，如年龄、薪资、日点击量等。

```python
from torch_rechub.basic.features import DenseFeature

# 创建数值型特征
dense_feature = DenseFeature(name="age", embed_dim=1)
```

**参数说明：**
- `name`：特征名称
- `embed_dim`：嵌入向量长度，固定为1

## SparseFeature

处理类别型特征，如城市、学历、性别等。

```python
from torch_rechub.basic.features import SparseFeature

# 创建类别型特征
sparse_feature = SparseFeature(
    name="city",
    vocab_size=100,  # 词汇表大小
    embed_dim=16,     # 嵌入向量长度
    shared_with=None  # 与其他特征共享嵌入表
)
```

**参数说明：**
- `name`：特征名称
- `vocab_size`：词汇表大小
- `embed_dim`：嵌入向量长度，若为None则自动计算
- `shared_with`：共享嵌入表的其他特征名称
- `padding_idx`：填充索引，在InputMask层中会被掩码为0
- `initializer`：嵌入层权重初始化器

## SequenceFeature

处理序列特征或多热特征，如用户行为序列、物品标签等。

```python
from torch_rechub.basic.features import SequenceFeature

# 创建序列特征
sequence_feature = SequenceFeature(
    name="user_history",
    vocab_size=10000,  # 词汇表大小
    embed_dim=32,       # 嵌入向量长度
    pooling="mean"       # 池化方式：mean, sum, concat
)
```

**参数说明：**
- `name`：特征名称
- `vocab_size`：词汇表大小
- `embed_dim`：嵌入向量长度，若为None则自动计算
- `pooling`：池化方式，支持mean、sum、concat
- `shared_with`：共享嵌入表的其他特征名称
- `padding_idx`：填充索引，在InputMask层中会被掩码为0
- `initializer`：嵌入层权重初始化器

## 特征使用示例

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature, SequenceFeature

# 定义特征
dense_features = [
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1)
]

sparse_features = [
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    SparseFeature(name="gender", vocab_size=3, embed_dim=8)
]

sequence_features = [
    SequenceFeature(name="user_history", vocab_size=10000, embed_dim=32, pooling="mean")
]

# 合并所有特征
all_features = dense_features + sparse_features + sequence_features
```