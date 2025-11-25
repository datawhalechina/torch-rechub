---
title: Utilities API Reference
description: API documentation for Torch-RecHub utilities including data processing, vector retrieval, and multi-task learning tools
---

# Utilities API Reference

This document provides detailed API documentation for utility classes and functions in Torch-RecHub.

## Data Processing Tools (data.py)

### Dataset Classes

#### TorchDataset
- **Introduction**: Basic PyTorch dataset implementation for handling feature and label data.
- **Parameters**:
  - `x` (dict): Feature dictionary with feature names as keys and feature data as values
  - `y` (array): Label data

#### PredictDataset
- **Introduction**: Dataset class for prediction stage containing only feature data.
- **Parameters**:
  - `x` (dict): Feature dictionary with feature names as keys and feature data as values

#### MatchDataGenerator
- **Introduction**: Data generator for recall tasks to generate training and test data loaders.
- **Main Methods**:
  - `generate_dataloader(x_test_user, x_all_item, batch_size, num_workers=8)`: Generate train, test, and item data loaders
  - **Parameters**:
    - `x_test_user` (dict): Test user features
    - `x_all_item` (dict): All item features
    - `batch_size` (int): Batch size
    - `num_workers` (int): Number of worker processes for data loading

#### DataGenerator
- **Introduction**: General-purpose data generator supporting dataset splitting and loading.
- **Main Methods**:
  - `generate_dataloader(x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16, num_workers=0)`: Generate train, validation, and test data loaders
  - **Parameters**:
    - `x_val`, `y_val`: Validation set features and labels
    - `x_test`, `y_test`: Test set features and labels
    - `split_ratio` (list): Split ratio for train, validation, and test sets
    - `batch_size` (int): Batch size
    - `num_workers` (int): Number of worker processes for data loading

### Utility Functions

#### get_auto_embedding_dim
- **Introduction**: Automatically calculate embedding dimension based on number of categories.
- **Parameters**:
  - `num_classes` (int): Number of categories
- **Returns**:
  - int: Embedding dimension, calculated as `[6 * (num_classes)^(1/4)]`

#### get_loss_func
- **Introduction**: Get loss function.
- **Parameters**:
  - `task_type` (str): Task type, "classification" or "regression"
- **Returns**:
  - torch.nn.Module: Corresponding loss function

#### get_metric_func
- **Introduction**: Get evaluation metric function.
- **Parameters**:
  - `task_type` (str): Task type, "classification" or "regression"
- **Returns**:
  - function: Corresponding evaluation metric function

#### generate_seq_feature
- **Introduction**: Generate sequence features and negative samples.
- **Parameters**:
  - `data` (pd.DataFrame): Raw data
  - `user_col` (str): User ID column name
  - `item_col` (str): Item ID column name
  - `time_col` (str): Timestamp column name
  - `item_attribute_cols` (list): Item attribute columns for sequence feature generation
  - `min_item` (int): Minimum number of items per user
  - `shuffle` (bool): Whether to shuffle data
  - `max_len` (int): Maximum sequence length

## Recall Tools (match.py)

### Data Processing Functions

#### gen_model_input
- **Introduction**: Merge user and item features, handle sequence features.
- **Parameters**:
  - `df` (pd.DataFrame): Data with historical sequence features
  - `user_profile` (pd.DataFrame): User feature data
  - `user_col` (str): User column name
  - `item_profile` (pd.DataFrame): Item feature data
  - `item_col` (str): Item column name
  - `seq_max_len` (int): Maximum sequence length
  - `padding` (str): Padding method, 'pre' or 'post'
  - `truncating` (str): Truncation method, 'pre' or 'post'

#### negative_sample
- **Introduction**: Negative sampling method for recall models.
- **Parameters**:
  - `items_cnt_order` (dict): Item count dictionary sorted by count in descending order
  - `ratio` (int): Negative sample ratio
  - `method_id` (int): Sampling method ID
    - 0: Random sampling
    - 1: Word2Vec-style popularity sampling
    - 2: Log popularity sampling
    - 3: Tencent RALM sampling

### Vector Retrieval Classes

#### Annoy
- **Introduction**: Vector retrieval tool based on Annoy library.
- **Parameters**:
  - `metric` (str): Distance metric
  - `n_trees` (int): Number of trees
  - `search_k` (int): Search parameter
- **Main Methods**:
  - `fit(X)`: Build index
  - `query(v, n)`: Query nearest neighbors

#### Milvus
- **Introduction**: Vector retrieval tool based on Milvus.
- **Parameters**:
  - `dim` (int): Vector dimension
  - `host` (str): Milvus server address
  - `port` (str): Milvus server port
- **Main Methods**:
  - `fit(X)`: Build index
  - `query(v, n)`: Query nearest neighbors

## Multi-Task Learning Tools (mtl.py)

### Utility Functions

#### shared_task_layers
- **Introduction**: Get shared and task-specific layer parameters from multi-task models.
- **Parameters**:
  - `model` (torch.nn.Module): Multi-task model supporting MMOE, SharedBottom, PLE, AITM
- **Returns**:
  - list: Shared layer parameters
  - list: Task-specific layer parameters

### Optimizer Classes

#### MetaBalance
- **Introduction**: MetaBalance optimizer for balancing gradients across tasks in multi-task learning.
- **Parameters**:
  - `parameters` (list): Model parameters
  - `relax_factor` (float): Gradient scaling relaxation factor, default 0.7
  - `beta` (float): Moving average coefficient, default 0.9
- **Main Methods**:
  - `step(losses)`: Perform optimization step and update parameters

### Gradient Processing Functions

#### gradnorm
- **Introduction**: Implement GradNorm algorithm for dynamic task weight adjustment in multi-task learning.
- **Parameters**:
  - `loss_list` (list): Loss list for each task
  - `loss_weight` (list): Task weight list
  - `share_layer` (torch.nn.Parameter): Shared layer parameters
  - `initial_task_loss` (list): Initial task loss list
  - `alpha` (float): GradNorm algorithm hyperparameter

