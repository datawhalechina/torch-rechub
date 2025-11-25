---
title: 基础组件 API 参考
description: Torch-RecHub 基础组件详细 API 文档，包括特征处理和数据转换
---

# 基础组件 API 参考

本文档详细介绍 Torch-RecHub 中的基础组件，包括特征处理、数据转换等基础功能。

## 特征处理

### 特征列 (Feature Columns)

#### DenseFeature
- **简介**：处理连续型数值特征。
- **参数**：
  - `name` (str): 特征名称
  - `dimension` (int): 特征维度
  - `dtype` (str): 数据类型，默认'float32'

#### SparseFeature
- **简介**：处理离散型类别特征。
- **参数**：
  - `name` (str): 特征名称
  - `vocabulary_size` (int): 类别词表大小
  - `embedding_dim` (int): 嵌入向量维度
  - `dtype` (str): 数据类型，默认'int32'
  - `embedding_name` (str): 嵌入层名称，默认None

#### VarLenSparseFeature
- **简介**：处理变长离散型特征。
- **参数**：
  - `name` (str): 特征名称
  - `vocabulary_size` (int): 类别词表大小
  - `embedding_dim` (int): 嵌入向量维度
  - `maxlen` (int): 序列最大长度
  - `dtype` (str): 数据类型，默认'int32'
  - `embedding_name` (str): 嵌入层名称，默认None
  - `combiner` (str): 序列池化方式，可选'sum'，'mean'，'max'，默认'mean'

## 数据转换

### 数据预处理

#### MinMaxScaler
- **简介**：数值特征归一化。
- **参数**：
  - `feature_range` (tuple): 归一化范围，默认(0, 1)

#### StandardScaler
- **简介**：数值特征标准化。
- **参数**：
  - `with_mean` (bool): 是否移除均值，默认True
  - `with_std` (bool): 是否按标准差缩放，默认True

#### LabelEncoder
- **简介**：类别特征编码。
- **方法**：
  - `fit(values)`: 拟合编码器
  - `transform(values)`: 转换数据
  - `fit_transform(values)`: 拟合并转换

### 数据格式转换

#### pandas_to_torch
- **简介**：将Pandas数据转换为PyTorch张量。
- **参数**：
  - `df` (pd.DataFrame): 输入DataFrame
  - `dense_cols` (list): 连续特征列名列表
  - `sparse_cols` (list): 离散特征列名列表
  - `device` (str): 设备类型，'cpu'或'cuda'

#### numpy_to_torch
- **简介**：将NumPy数组转换为PyTorch张量。
- **参数**：
  - `arrays` (list): NumPy数组列表
  - `device` (str): 设备类型，'cpu'或'cuda'

## 模型组件

### 激活函数

#### Dice
- **简介**：Dice激活函数，在深度兴趣网络(DIN)中提出。
- **参数**：
  - `epsilon` (float): 平滑参数，默认1e-3
  - `device` (str): 设备类型，默认'cpu'

### 注意力机制

#### ScaledDotProductAttention
- **简介**：缩放点积注意力机制。
- **参数**：
  - `temperature` (float): 缩放温度参数
  - `attn_dropout` (float): 注意力Dropout比率

#### MultiHeadAttention
- **简介**：多头注意力机制。
- **参数**：
  - `d_model` (int): 模型维度
  - `n_heads` (int): 注意力头数
  - `d_k` (int): 键向量维度
  - `d_v` (int): 值向量维度
  - `dropout` (float): Dropout比率

