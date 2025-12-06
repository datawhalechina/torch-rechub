---
title: 模型 API 参考
description: Torch-RecHub 所有模型的完整 API 文档，包括召回、排序和多任务学习模型
---

# 模型 API 参考

这里详细介绍 Torch-RecHub 中各个模型的 API 接口和参数说明。

## 召回模型 (Recall Models)

召回模型主要用于召回阶段，从海量候选集中快速检索相关物品。通常采用双塔结构或序列模型结构，以满足召回阶段的效率要求。

### 双塔模型系列

#### DSSM (Deep Structured Semantic Model)
- **简介**：最早由微软提出的语义匹配模型，后被广泛应用于推荐系统。采用经典的双塔结构，分别对用户和物品进行表征，通过内积计算相似度。这种结构允许在线服务时预先计算物品向量，极大提高了服务效率。模型的关键在于如何学习到有效的用户和物品表征。
- **参数**：
  - `user_features` (list): 用户特征列表
  - `item_features` (list): 物品特征列表
  - `hidden_units` (list): 隐藏层单元数列表
  - `dropout_rates` (list): Dropout比率列表
  - `embedding_dim` (int): 最终的表征向量维度

#### Facebook DSSM
- **简介**：Facebook对DSSM的改进版本，引入多任务学习框架。除了主要的召回任务外，还增加了辅助任务来帮助学习更好的特征表征。模型可以同时优化多个相关目标，如点击、收藏、购买等，从而学习到更丰富的用户和物品表征。
- **参数**：
  - `user_features` (list): 用户特征列表
  - `item_features` (list): 物品特征列表
  - `hidden_units` (list): 隐藏层单元数列表
  - `num_tasks` (int): 任务数量
  - `task_types` (list): 任务类型列表

#### YouTube DNN
- **简介**：YouTube提出的深度召回模型，针对大规模视频推荐场景设计。模型将用户观看历史通过平均池化进行聚合，并结合用户其他特征进行建模。创新性地引入了负采样技术和多任务学习框架，以提高模型的训练效率和效果。
- **参数**：
  - `user_features` (list): 用户特征列表
  - `item_features` (list): 物品特征列表
  - `hidden_units` (list): 隐藏层单元数列表
  - `embedding_dim` (int): 嵌入维度
  - `max_seq_len` (int): 最大序列长度

### 序列推荐系列

#### GRU4Rec
- **简介**：首次将GRU网络应用于会话序列推荐的开创性工作。通过GRU网络捕捉用户行为序列中的时序依赖关系，每个时间步的隐藏状态都包含了用户历史行为的信息。模型还引入了特殊的mini-batch构造方法和损失函数设计，以适应序列推荐的特点。
- **参数**：
  - `item_num` (int): 物品总数
  - `hidden_size` (int): GRU隐藏层大小
  - `num_layers` (int): GRU层数
  - `dropout_rate` (float): Dropout比率
  - `embedding_dim` (int): 物品嵌入维度

#### NARM (Neural Attentive Recommendation Machine)
- **简介**：在GRU4Rec基础上引入注意力机制的序列推荐模型。通过注意力机制，模型可以根据当前预测目标动态关注序列中的相关行为。模型维护全局和局部序列表征，全面捕捉用户的短期兴趣。这种设计能更好地处理用户兴趣的多样性和动态性。
- **参数**：
  - `item_num` (int): 物品总数
  - `hidden_size` (int): 隐藏层大小
  - `attention_size` (int): 注意力层大小
  - `dropout_rate` (float): Dropout比率
  - `embedding_dim` (int): 物品嵌入维度

#### SASRec (Self-Attentive Sequential Recommendation)
- **简介**：将Transformer结构应用于序列推荐的代表性工作。通过自注意力机制，模型可以直接计算和学习序列中任意两个行为之间的关系，不受RNN固有的序列依赖限制。位置编码帮助保留行为的时序信息，多层结构使模型能逐层提取越来越抽象的行为模式。相比RNN模型，具有更好的并行性和可扩展性。
- **参数**：
  - `item_num` (int): 物品总数
  - `max_len` (int): 最大序列长度
  - `num_heads` (int): 注意力头数
  - `num_layers` (int): Transformer层数
  - `hidden_size` (int): 隐藏层维度
  - `dropout_rate` (float): Dropout比率

#### MIND (Multi-Interest Network with Dynamic routing)
- **简介**：为用户多样化兴趣设计的召回模型。通过胶囊网络和动态路由机制，从用户行为序列中提取多个兴趣向量。每个兴趣向量代表用户在不同方面的偏好，提供了更全面的用户兴趣分布表征。
- **参数**：
  - `item_num` (int): 物品总数
  - `num_interests` (int): 兴趣向量数量
  - `routing_iterations` (int): 动态路由迭代次数
  - `hidden_size` (int): 隐藏层维度
  - `embedding_dim` (int): 物品嵌入维度

## 排序模型 (Ranking Models)

排序模型主要用于精排阶段，对候选物品进行精确排序。通过深度学习方法学习用户与物品之间的复杂交互，生成最终的排序分数。

### Wide & Deep 系列

#### WideDeep
- **简介**：Google在2016年提出的经典模型，结合了线性模型和深度神经网络的优势。Wide部分通过特征交叉进行记忆，适合建模直接、显式的特征相关性；Deep部分通过深度网络进行泛化，能学习隐式、高阶的特征关系。这种结合使模型既能记忆历史模式，又能泛化到新模式。
- **参数**：
  - `wide_features` (list): Wide部分的特征列表，用于线性层
  - `deep_features` (list): Deep部分的特征列表，用于深度网络
  - `hidden_units` (list): 深度网络的隐藏层单元数列表，如[256, 128, 64]
  - `dropout_rates` (list): 各层的Dropout比率，用于防止过拟合

#### DeepFM
- **简介**：结合因子分解机(FM)特征交互和深度学习模型的模型。FM部分高效地建模二阶特征交互，Deep部分学习高阶特征关系。相比Wide&Deep，DeepFM无需手工特征工程，能自动学习特征交叉。模型包含三部分：一阶特征、FM的二阶交互和深度网络的高阶交互。
- **参数**：
  - `features` (list): 特征列表
  - `hidden_units` (list): DNN部分的隐藏层单元数
  - `dropout_rates` (list): Dropout比率列表
  - `embedding_dim` (int): 特征嵌入维度

#### DCN / DCN-V2
- **简介**：通过特殊设计的交叉网络层显式学习特征交互。每个交叉层在特征向量与其原始形式之间进行交互，随着深度增加，特征交叉的阶数也增加。DCN-V2改进了交叉网络参数化，提供"向量"和"矩阵"两种选项，在保持模型表达能力的同时提高效率。
- **参数**：
  - `features` (list): 特征列表
  - `cross_num` (int): 交叉层数
  - `hidden_units` (list): DNN部分的隐藏层单元数
  - `cross_parameterization` (str, DCN-V2): 交叉参数化方法，"vector"或"matrix"

#### AFM (Attentional Factorization Machine)
- **简介**：在FM基础上引入注意力机制，为不同的特征交互分配不同的重要性权重。通过注意力网络，自适应地学习特征交互的重要性，识别与预测目标更相关的特征组合。
- **参数**：
  - `features` (list): 特征列表
  - `attention_units` (list): 注意力网络的隐藏层单元数
  - `embedding_dim` (int): 特征嵌入维度
  - `dropout_rate` (float): 注意力网络的Dropout比率

#### FiBiNET (Feature Importance and Bilinear feature Interaction Network)
- **简介**：通过SENET机制动态学习特征重要性，使用双线性层进行特征交互。SENET模块帮助识别重要特征，双线性交互提供比内积更丰富的特征交互方式。
- **参数**：
  - `features` (list): 特征列表
  - `bilinear_type` (str): 双线性层类型，可选"field_all"/"field_each"/"field_interaction"
  - `hidden_units` (list): DNN部分的隐藏层单元数
  - `reduction_ratio` (int): SENET模块的缩放比率

### 基于注意力的系列

#### DIN (Deep Interest Network)
- **简介**：为用户兴趣多样性设计的模型，使用注意力机制自适应学习用户历史行为。模型根据当前候选物品动态计算用户历史行为的相关性权重，从而激活相关用户兴趣，捕捉多样化用户偏好。创新性地将注意力机制引入推荐系统，开创了行为序列建模的新范式。
- **参数**：
  - `features` (list): 基础特征列表
  - `behavior_features` (list): 用于注意力计算的行为特征列表
  - `attention_units` (list): 注意力网络的隐藏层单元数
  - `hidden_units` (list): DNN部分的隐藏层单元数
  - `activation` (str): 激活函数类型

#### DIEN (Deep Interest Evolution Network)
- **简介**：DIN的高级版本，通过兴趣演化层建模用户兴趣的动态演化。使用GRU结构捕捉兴趣演化，创新性地设计AUGRU(带注意力更新门的GRU)使兴趣演化过程感知目标物品。还包含辅助损失监督兴趣提取层的训练。这种设计既捕捉用户兴趣的动态变化，又建模兴趣的时序依赖。
- **参数**：
  - `features` (list): 基础特征列表
  - `behavior_features` (list): 行为特征列表
  - `interest_units` (list): 兴趣提取层的单元数
  - `gru_type` (str): GRU类型，"AUGRU"或"AIGRU"
  - `hidden_units` (list): DNN部分的隐藏层单元数

#### BST (Behavior Sequence Transformer)
- **简介**：将Transformer架构引入推荐系统建模用户行为序列的开创性工作。通过自注意力机制，模型可以直接计算序列中任意两个行为之间的关系，克服RNN模型处理长序列的局限。位置嵌入帮助模型感知行为的时序信息，多头注意力使模型从多个角度理解用户行为模式。
- **参数**：
  - `features` (list): 基础特征列表
  - `behavior_features` (list): 行为特征列表
  - `num_heads` (int): 注意力头数
  - `num_layers` (int): Transformer层数
  - `hidden_size` (int): 隐藏层维度
  - `dropout_rate` (float): Dropout比率

#### EDCN (Enhancing Explicit and Implicit Feature Interactions)
- **简介**：增强显式和隐式特征交互的深度交叉网络。通过新设计的交叉网络结构，同时考虑显式和隐式特征交互。引入门控机制调节不同阶特征交互的重要性，使用残差连接缓解深度网络的训练问题。
- **参数**：
  - `features` (list): 特征列表
  - `cross_num` (int): 交叉层数
  - `hidden_units` (list): DNN部分的隐藏层单元数
  - `gate_type` (str): 门控类型，"FGU"或"BGU"

## 多任务模型 (Multi-task Models)

多任务模型联合学习多个相关任务，实现知识共享和迁移，提升整体模型性能。

### SharedBottom
- **简介**：最基础的多任务学习模型，在底层网络中共享参数以提取通用特征表征。共享层学习任务间的通用特征，而任务特定层学习各任务的个性化特征。这种简单而有效的结构为多任务学习奠定了基础。
- **参数**：
  - `features` (list): 特征列表
  - `hidden_units` (list): 共享网络的隐藏层单元数
  - `task_hidden_units` (list): 任务特定网络的隐藏层单元数
  - `num_tasks` (int): 任务数量
  - `task_types` (list): 任务类型列表

### ESMM (Entire Space Multi-Task Model)
- **简介**：阿里巴巴提出的创新多任务模型，专门设计用于解决推荐系统中的样本选择偏差问题。通过联合建模CVR和CTR任务，在全空间进行参数学习。核心创新在于引入CTR作为辅助任务，通过任务乘法关系优化CVR估计。这种设计不仅解决了传统CVR估计中的样本选择偏差，还提供了无偏的CTR和CTCVR估计。
- **参数**：
  - `features` (list): 特征列表
  - `hidden_units` (list): 隐藏层单元数列表
  - `tower_units` (list): 任务塔层的单元数列表
  - `embedding_dim` (int): 特征嵌入维度

### MMoE (Multi-gate Mixture-of-Experts)
- **简介**：Google提出的多任务学习模型，通过专家机制和任务相关的门控网络实现软参数共享。每个专家网络可学习特定的特征转换，门控网络动态为各任务分配专家重要性。这种设计使模型能根据任务需求灵活组合专家知识，有效处理任务差异。
- **参数**：
  - `features` (list): 特征列表
  - `expert_units` (list): 专家网络的隐藏层单元数
  - `num_experts` (int): 专家数量
  - `num_tasks` (int): 任务数量
  - `expert_activation` (str): 专家网络的激活函数
  - `gate_activation` (str): 门控网络的激活函数

### PLE (Progressive Layered Extraction)
- **简介**：MMoE的改进版本，通过渐进式分层提取更好地建模任务关系。引入任务特定专家和共享专家的概念，通过多层专家网络实现渐进式特征提取。每层包含任务特定专家和共享专家，使模型既能学习任务的共性，又能学习个性。这种渐进式设计增强了模型的知识提取和迁移能力。
- **参数**：
  - `features` (list): 特征列表
  - `expert_units` (list): 专家网络的单元数
  - `num_experts` (int): 每层的专家数量
  - `num_layers` (int): 层数
  - `num_shared_experts` (int): 共享专家数量
  - `task_types` (list): 任务类型列表

