---
title: Basic Components API Reference
description: Detailed API documentation for basic components including feature processing and data transformation
---

# Basic Components API Reference

This document provides detailed documentation for basic components in Torch-RecHub, including feature processing, data transformation, and other fundamental functionalities.

## Feature Processing

### Feature Columns

#### DenseFeature
- **Introduction**: Process continuous numerical features.
- **Parameters**:
  - `name` (str): Feature name
  - `dimension` (int): Feature dimension
  - `dtype` (str): Data type, default 'float32'

#### SparseFeature
- **Introduction**: Process discrete categorical features.
- **Parameters**:
  - `name` (str): Feature name
  - `vocabulary_size` (int): Size of category vocabulary
  - `embedding_dim` (int): Embedding vector dimension
  - `dtype` (str): Data type, default 'int32'
  - `embedding_name` (str): Embedding layer name, default None

#### VarLenSparseFeature
- **Introduction**: Process variable-length discrete features.
- **Parameters**:
  - `name` (str): Feature name
  - `vocabulary_size` (int): Size of category vocabulary
  - `embedding_dim` (int): Embedding vector dimension
  - `maxlen` (int): Maximum sequence length
  - `dtype` (str): Data type, default 'int32'
  - `embedding_name` (str): Embedding layer name, default None
  - `combiner` (str): Sequence pooling method, options: 'sum', 'mean', 'max', default 'mean'

## Data Transformation

### Data Preprocessing

#### MinMaxScaler
- **Introduction**: Normalize numerical features.
- **Parameters**:
  - `feature_range` (tuple): Normalization range, default (0, 1)

#### StandardScaler
- **Introduction**: Standardize numerical features.
- **Parameters**:
  - `with_mean` (bool): Whether to remove mean, default True
  - `with_std` (bool): Whether to scale by standard deviation, default True

#### LabelEncoder
- **Introduction**: Encode categorical features.
- **Methods**:
  - `fit(values)`: Fit the encoder
  - `transform(values)`: Transform data
  - `fit_transform(values)`: Fit and transform

### Data Format Conversion

#### pandas_to_torch
- **Introduction**: Convert Pandas data to PyTorch tensors.
- **Parameters**:
  - `df` (pd.DataFrame): Input DataFrame
  - `dense_cols` (list): List of continuous feature column names
  - `sparse_cols` (list): List of discrete feature column names
  - `device` (str): Device type, 'cpu' or 'cuda'

#### numpy_to_torch
- **Introduction**: Convert NumPy arrays to PyTorch tensors.
- **Parameters**:
  - `arrays` (list): List of NumPy arrays
  - `device` (str): Device type, 'cpu' or 'cuda'

## Model Components

### Activation Functions

#### Dice
- **Introduction**: Dice activation function, proposed in Deep Interest Network (DIN).
- **Parameters**:
  - `epsilon` (float): Smoothing parameter, default 1e-3
  - `device` (str): Device type, default 'cpu'

### Attention Mechanisms

#### ScaledDotProductAttention
- **Introduction**: Scaled dot-product attention mechanism.
- **Parameters**:
  - `temperature` (float): Temperature parameter for scaling
  - `attn_dropout` (float): Attention dropout rate

#### MultiHeadAttention
- **Introduction**: Multi-head attention mechanism.
- **Parameters**:
  - `d_model` (int): Model dimension
  - `n_heads` (int): Number of attention heads
  - `d_k` (int): Key vector dimension
  - `d_v` (int): Value vector dimension
  - `dropout` (float): Dropout rate

