---
title: 工具类 API 参考
description: Torch-RecHub 工具类的 API 文档，包括数据处理、向量检索和多任务学习工具
---

# 工具类 API 参考

本文档详细介绍 Torch-RecHub 中各个工具类的 API 接口和参数说明。

## 数据处理工具 (data.py)

### 数据集类

#### TorchDataset
- **简介**：PyTorch数据集的基础实现，用于处理特征和标签数据。
- **参数**：
  - `x` (dict): 特征字典，键为特征名，值为特征数据
  - `y` (array): 标签数据

#### PredictDataset
- **简介**：用于预测阶段的数据集类，只包含特征数据。
- **参数**：
  - `x` (dict): 特征字典，键为特征名，值为特征数据

#### MatchDataGenerator
- **简介**：召回任务的数据生成器，用于生成训练和测试数据加载器。
- **主要方法**：
  - `generate_dataloader(x_test_user, x_all_item, batch_size, num_workers=8)`: 生成训练、测试和物品数据加载器
  - **参数**：
    - `x_test_user` (dict): 测试用户特征
    - `x_all_item` (dict): 所有物品特征
    - `batch_size` (int): 批次大小
    - `num_workers` (int): 数据加载的工作进程数

#### DataGenerator
- **简介**：通用数据生成器，支持数据集的划分和加载。
- **主要方法**：
  - `generate_dataloader(x_val=None, y_val=None, x_test=None, y_test=None, split_ratio=None, batch_size=16, num_workers=0)`: 生成训练、验证和测试数据加载器
  - **参数**：
    - `x_val`, `y_val`: 验证集特征和标签
    - `x_test`, `y_test`: 测试集特征和标签
    - `split_ratio` (list): 训练集、验证集、测试集的划分比例
    - `batch_size` (int): 批次大小
    - `num_workers` (int): 数据加载的工作进程数

### 工具函数

#### get_auto_embedding_dim
- **简介**：根据类别数自动计算嵌入向量维度。
- **参数**：
  - `num_classes` (int): 类别数量
- **返回**：
  - int: 嵌入向量维度，计算公式：`[6 * (num_classes)^(1/4)]`

#### get_loss_func
- **简介**：获取损失函数。
- **参数**：
  - `task_type` (str): 任务类型，"classification"或"regression"
- **返回**：
  - torch.nn.Module: 对应的损失函数

#### get_metric_func
- **简介**：获取评估指标函数。
- **参数**：
  - `task_type` (str): 任务类型，"classification"或"regression"
- **返回**：
  - function: 对应的评估指标函数

#### generate_seq_feature
- **简介**：生成序列特征和负样本。
- **参数**：
  - `data` (pd.DataFrame): 原始数据
  - `user_col` (str): 用户ID列名
  - `item_col` (str): 物品ID列名
  - `time_col` (str): 时间戳列名
  - `item_attribute_cols` (list): 需要生成序列特征的物品属性列
  - `min_item` (int): 用户最少交互物品数
  - `shuffle` (bool): 是否打乱数据
  - `max_len` (int): 序列最大长度

## 召回工具 (match.py)

### 数据处理函数

#### gen_model_input
- **简介**：合并用户和物品特征，处理序列特征。
- **参数**：
  - `df` (pd.DataFrame): 带有历史序列特征的数据
  - `user_profile` (pd.DataFrame): 用户特征数据
  - `user_col` (str): 用户列名
  - `item_profile` (pd.DataFrame): 物品特征数据
  - `item_col` (str): 物品列名
  - `seq_max_len` (int): 序列最大长度
  - `padding` (str): 填充方式，'pre'或'post'
  - `truncating` (str): 截断方式，'pre'或'post'

#### negative_sample
- **简介**：召回模型的负采样方法。
- **参数**：
  - `items_cnt_order` (dict): 物品计数字典，按计数降序排序
  - `ratio` (int): 负样本比例
  - `method_id` (int): 采样方法ID
    - 0: 随机采样
    - 1: Word2Vec式流行度采样
    - 2: 对数流行度采样
    - 3: 腾讯RALM采样

### 向量检索类

#### Annoy
- **简介**：基于Annoy的向量召回工具。
- **参数**：
  - `metric` (str): 距离度量方式
  - `n_trees` (int): 树的数量
  - `search_k` (int): 搜索参数
- **主要方法**：
  - `fit(X)`: 构建索引
  - `query(v, n)`: 查询最近邻

#### Milvus
- **简介**：基于Milvus的向量召回工具。
- **参数**：
  - `dim` (int): 向量维度
  - `host` (str): Milvus服务器地址
  - `port` (str): Milvus服务器端口
- **主要方法**：
  - `fit(X)`: 构建索引
  - `query(v, n)`: 查询最近邻

## 多任务学习工具 (mtl.py)

### 工具函数

#### shared_task_layers
- **简介**：获取多任务模型中的共享层和任务特定层参数。
- **参数**：
  - `model` (torch.nn.Module): 多任务模型，支持MMOE、SharedBottom、PLE、AITM
- **返回**：
  - list: 共享层参数列表
  - list: 任务特定层参数列表

### 优化器类

#### MetaBalance
- **简介**：MetaBalance优化器，用于平衡多任务学习中各任务的梯度。
- **参数**：
  - `parameters` (list): 模型参数
  - `relax_factor` (float): 梯度缩放的松弛因子，默认0.7
  - `beta` (float): 移动平均系数，默认0.9
- **主要方法**：
  - `step(losses)`: 执行优化步骤，更新参数

### 梯度处理函数

#### gradnorm
- **简介**：实现GradNorm算法，用于动态调整多任务学习中的任务权重。
- **参数**：
  - `loss_list` (list): 各任务的损失列表
  - `loss_weight` (list): 任务权重列表
  - `share_layer` (torch.nn.Parameter): 共享层参数
  - `initial_task_loss` (list): 初始任务损失列表
  - `alpha` (float): GradNorm算法的超参数

