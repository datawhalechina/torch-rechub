---
title: 特征定义
description: Torch-RecHub 特征类型定义
---

# 特征定义

Torch-RecHub提供了三种核心特征类，用于处理不同类型的特征：

![特征类型到模型输入示意图](/img/diagrams/feature_types_flow.png)

## DenseFeature

处理数值型特征，如年龄、薪资、日点击量等。

```python
from torch_rechub.basic.features import DenseFeature

# 创建数值型特征
dense_feature = DenseFeature(name="age", embed_dim=1)
```

**参数说明：**
- `name`：特征名称
- `embed_dim`：输入维度，标量特征默认为 1；如果该字段本身是向量，应设为实际向量维度

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
    pooling="mean",      # 池化方式：mean, sum, concat
    padding_idx=0,       # 序列中用 0 补齐
)
```

**参数说明：**
- `name`：特征名称
- `vocab_size`：词汇表大小
- `embed_dim`：嵌入向量长度，若为None则自动计算
- `pooling`：支持 `mean`、`sum`、`concat`；`mean/sum` 把 `(batch_size, seq_len, embed_dim)` 压缩为 `(batch_size, embed_dim)`，`concat` 保留原序列维度
- `shared_with`：共享嵌入表的其他特征名称
- `padding_idx`：填充索引，在InputMask层中会被掩码为0
- `initializer`：嵌入层权重初始化器

由于 `concat` 返回保留序列维度的结果，不能在一次 `EmbeddingLayer` 调用中把使用 `pooling="concat"` 的 `SequenceFeature` 与稀疏特征或已池化的序列特征混用。序列嵌入应与普通特征嵌入分开获取：

```python
input_user = embedding(x, user_features, squeeze_dim=True)  # (B, D)
history_emb = embedding(x, history_features)                 # (B, H, L, D)
single_history_emb = history_emb.squeeze(1)                  # (B, L, D)，仅当 H == 1
```

## 模型输入约定

Feature 对象只定义模型如何解读字段，不会自动对原始数据做类别编码或序列补齐。传入模型的字典需满足：

| 特征类型 | 常用形状 | 值的约定 |
|---|---|---|
| `DenseFeature` | `(batch_size,)` 或 `(batch_size, embed_dim)` | 可转换为浮点数 |
| `SparseFeature` | `(batch_size,)` | 整数索引，范围为 `[0, vocab_size)` |
| `SequenceFeature` | `(batch_size, seq_len)` | 整数索引，需在进入模型前补齐到相同长度 |

当序列用 `0` 补齐时，应显式设置 `padding_idx=0`，否则默认掩码会把 `-1` 而不是 `0` 视为填充值。

使用 `shared_with="item_id"` 时，同一个 `EmbeddingLayer` 的特征列表中必须同时存在名为 `item_id` 且自己创建 Embedding 的特征；实际使用的词表大小和嵌入维度由被共享的 Embedding 表决定。

## Embedding 初始化

`initializer` 需要传入一个可调用的初始化器实例。项目内置 `RandomNormal`、`RandomUniform`、`XavierNormal`、`XavierUniform` 和 `Pretrained`；`SparseFeature` 与 `SequenceFeature` 默认使用 `RandomNormal(0, 0.0001)`。

```python
import torch

from torch_rechub.basic.features import SparseFeature
from torch_rechub.basic.initializers import XavierUniform, Pretrained

random_feature = SparseFeature(
    name="item_id",
    vocab_size=1000,
    embed_dim=32,
    padding_idx=0,
    initializer=XavierUniform(gain=1.0),
)

weights = torch.randn(1000, 32)
pretrained_feature = SparseFeature(
    name="pretrained_item_id",
    vocab_size=1000,
    embed_dim=32,
    padding_idx=0,
    initializer=Pretrained(weights, freeze=False),
)
```

`Pretrained` 会校验权重形状与 `vocab_size/embed_dim` 完全一致；`freeze=True` 是默认值，表示训练时不更新该 Embedding。对于内置的随机/Xavier 初始化器，设置 `padding_idx` 后对应行会被置零。

## Feature 实例与 Embedding 所有权

> **注意**：`SparseFeature` 和 `SequenceFeature` 会缓存通过 `get_embedding_layer()` 创建的 `nn.Embedding`。如果将**同一个 Feature 实例**传给多个模型，这些模型会使用同一份 Embedding 参数；训练或加载其中一个模型的权重时，其他模型看到的 Embedding 也会随之变化。

下面的两个 EmbeddingLayer 会意外共享 `city` 的 Embedding：

```python
from torch_rechub.basic.features import SparseFeature
from torch_rechub.basic.layers import EmbeddingLayer

features = [SparseFeature(name="city", vocab_size=100, embed_dim=16)]

embedding_a = EmbeddingLayer(features)
embedding_b = EmbeddingLayer(features)

assert embedding_a.embed_dict["city"] is embedding_b.embed_dict["city"]
```

进行独立的模型对比、交叉验证或集成训练时，请为每个模型重新创建 Feature 实例：

```python
def build_features():
    return [SparseFeature(name="city", vocab_size=100, embed_dim=16)]

embedding_a = EmbeddingLayer(build_features())
embedding_b = EmbeddingLayer(build_features())

assert embedding_a.embed_dict["city"] is not embedding_b.embed_dict["city"]
```

仅复制列表并不能解决该问题：`features.copy()` 仍然包含原来的 Feature 实例。这里所说的跨模型意外共享，也不同于在同一个模型内使用 `shared_with` 显式共享 Embedding；后者是预期行为。

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
    SequenceFeature(
        name="user_history",
        vocab_size=10000,
        embed_dim=32,
        pooling="mean",
        padding_idx=0,
    )
]

# 合并所有特征
all_features = dense_features + sparse_features + sequence_features
```
