# Utilities API Reference

This document provides detailed API documentation for utility classes and functions in Torch-RecHub.

## Data Processing Tools (data.py)

### Dataset Classes

#### TorchDataset
- **Introduction**: Basic implementation of PyTorch dataset for handling features and labels.
- **Parameters**:
  - `x` (dict): Feature dictionary, keys are feature names, values are feature data
  - `y` (array): Label data

#### PredictDataset
- **Introduction**: Dataset class for prediction phase, containing only feature data.
- **Parameters**:
  - `x` (dict): Feature dictionary, keys are feature names, values are feature data

#### MatchDataGenerator
- **Introduction**: Data generator for recall tasks, used to generate training and testing data loaders.
- **Main Methods**:
  - `generate_dataloader(x_test_user, x_all_item, batch_size, num_workers=8)`: Generate training, testing, and item data loaders
  - **Parameters**:
    - `x_test_user` (dict): Test user features
    - `x_all_item` (dict): All item features
    - `batch_size` (int): Batch size
    - `num_workers` (int): Number of worker processes for data loading

#### DataGenerator
- **Introduction**: General data generator supporting dataset splitting and loading.
- **Main Methods**:
  - `generate_dataloader(x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16, num_workers=0)`: Generate training, validation, and test data loaders
  - **Parameters**:
    - `x_val`, `y_val`: Validation set features and labels
    - `x_test`, `y_test`: Test set features and labels
    - `split_ratio` (list): Split ratios for train, validation, and test sets
    - `batch_size` (int): Batch size
    - `num_workers` (int): Number of worker processes for data loading

### Utility Functions

#### get_auto_embedding_dim
- **Introduction**: Automatically calculate embedding vector dimension based on number of categories.
- **Parameters**:
  - `num_classes` (int): Number of categories
- **Returns**:
  - int: Embedding vector dimension, formula: `[6 * (num_classes)^(1/4)]`

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
- **Introduction**: Merge user and item features, process sequence features.
- **Parameters**:
  - `df` (pd.DataFrame): Data with history sequence features
  - `user_profile` (pd.DataFrame): User feature data
  - `user_col` (str): User column name
  - `item_profile` (pd.DataFrame): Item feature data
  - `item_col` (str): Item column name
  - `seq_max_len` (int): Maximum sequence length
  - `padding` (str): Padding method, 'pre' or 'post'
  - `truncating` (str): Truncating method, 'pre' or 'post'

#### negative_sample
- **Introduction**: Negative sampling method for recall models.
- **Parameters**:
  - `items_cnt_order` (dict): Item count dictionary, sorted by count in descending order
  - `ratio` (int): Negative sample ratio
  - `method_id` (int): Sampling method ID
    - 0: Random sampling
    - 1: Word2Vec-style popularity sampling
    - 2: Log popularity sampling
    - 3: Tencent RALM sampling

### Vector Retrieval Classes

#### Annoy
- **Introduction**: Vector recall tool based on Annoy.
- **Parameters**:
  - `metric` (str): Distance metric method
  - `n_trees` (int): Number of trees
  - `search_k` (int): Search parameter
- **Main Methods**:
  - `fit(X)`: Build index
  - `query(v, n)`: Query nearest neighbors

#### Milvus
- **Introduction**: Vector recall tool based on Milvus.
- **Parameters**:
  - `dim` (int): Vector dimension
  - `host` (str): Milvus server address
  - `port` (str): Milvus server port
- **Main Methods**:
  - `fit(X)`: Build index
  - `query(v, n)`: Query nearest neighbors

## Multi-task Learning Tools (mtl.py)

### Utility Functions

#### shared_task_layers
- **Introduction**: Get shared layer and task-specific layer parameters in multi-task models.
- **Parameters**:
  - `model` (torch.nn.Module): Multi-task model, supports MMOE, SharedBottom, PLE, AITM
- **Returns**:
  - list: Shared layer parameter list
  - list: Task-specific layer parameter list

### Optimizer Classes

#### MetaBalance
- **Introduction**: MetaBalance optimizer for balancing gradients in multi-task learning.
- **Parameters**:
  - `parameters` (list): Model parameters
  - `relax_factor` (float): Relaxation factor for gradient scaling, default 0.7
  - `beta` (float): Moving average coefficient, default 0.9
- **Main Methods**:
  - `step(losses)`: Execute optimization step, update parameters

### Gradient Processing Functions

#### gradnorm
- **Introduction**: Implement GradNorm algorithm for dynamically adjusting task weights in multi-task learning.
- **Parameters**:
  - `loss_list` (list): List of task losses
  - `loss_weight` (list): List of task weights
  - `share_layer` (torch.nn.Parameter): Shared layer parameters
  - `initial_task_loss` (list): List of initial task losses
  - `alpha` (float): GradNorm algorithm hyperparameter