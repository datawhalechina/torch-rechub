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
- **简介**：在GRU4Rec基础上引入注意力机制的序列推荐模型。通过注意力机制，模型可以根据当前预测目标，动态地关注序列中的相关行为。同时维护全局和局部两个序列表征，全面捕捉用户的短期兴趣。这种设计让模型能够更好地处理用户兴趣的多样性和动态性。
- **参数**：
  - `item_num` (int): 物品总数
  - `hidden_size` (int): 隐藏层大小
  - `attention_size` (int): 注意力层大小
  - `dropout_rate` (float): Dropout比率
  - `embedding_dim` (int): 物品嵌入维度

#### SASRec (Self-Attentive Sequential Recommendation)
- **简介**：将Transformer结构应用于序列推荐的代表性工作。通过自注意力机制，模型可以直接计算并学习序列中任意两个行为之间的关系，不受RNN固有的序列依赖限制。位置编码帮助保留了行为的时序信息，多层结构则允许模型逐层抽取越来越抽象的行为模式。相比RNN类模型，具有更好的并行性和可扩展性。
- **参数**：
  - `item_num` (int): 物品总数
  - `max_len` (int): 最大序列长度
  - `num_heads` (int): 注意力头数
  - `num_layers` (int): Transformer层数
  - `hidden_size` (int): 隐藏层维度
  - `dropout_rate` (float): Dropout比率

#### MIND (Multi-Interest Network with Dynamic routing)
- **简介**：针对用户多样化兴趣设计的召回模型。通过胶囊网络和动态路由机制，从用户的行为序列中提取多个兴趣向量。每个兴趣向量代表用户在不同方面的偏好，这种多兴趣表示方式能更全面地刻画用户的兴趣分布。
- **参数**：
  - `item_num` (int): 物品总数
  - `num_interests` (int): 兴趣向量数量
  - `routing_iterations` (int): 动态路由迭代次数
  - `hidden_size` (int): 隐藏层维度
  - `embedding_dim` (int): 物品嵌入维度

## 排序模型 (Ranking Models)

排序模型主要用于精排阶段，对候选集进行精确排序。通过深度学习方法学习用户和物品之间的复杂交互关系，生成最终的排序得分。

### Wide & Deep 系列

#### WideDeep
- **简介**：Google 在 2016 年提出的经典模型，结合线性模型和深度神经网络的优势。Wide 部分通过特征交叉进行记忆，适合建模直接的、显式的特征相关性；Deep 部分通过深度网络进行泛化，可以学习特征间的隐式、高阶关系。这种结合让模型既能记住历史规律，又能泛化到新模式。
- **参数**：
  - `wide_features` (list): Wide部分特征列表，用于线性层
  - `deep_features` (list): Deep部分特征列表，用于深度网络
  - `hidden_units` (list): Deep网络的隐藏层单元数列表，如 [256, 128, 64]
  - `dropout_rates` (list): 每层的dropout比率，用于防止过拟合

#### DeepFM
- **简介**：将FM（因子分解机）的特征交互与深度学习模型相结合的模型。FM部分可以高效地建模二阶特征交互，而Deep部分则可以学习高阶特征关系。相比Wide&Deep，DeepFM不需要手动进行特征工程，可以自动学习特征交叉。模型包含三部分：一阶特征、FM的二阶交互、深度网络的高阶交互。
- **参数**：
  - `features` (list): 特征列表
  - `hidden_units` (list): DNN部分隐藏层单元数
  - `dropout_rates` (list): Dropout比率列表
  - `embedding_dim` (int): 特征嵌入维度

#### DCN / DCN-V2
- **简介**：通过特殊设计的交叉网络层（Cross Network）显式学习特征交互。每一层交叉网络都执行特征向量与其原始形式的交互，使得交叉特征的程度随着层数的加深而提高。DCN-V2改进了交叉网络的参数化方式，提供了"向量"和"矩阵"两种参数化选项，在保持模型表达能力的同时提高了效率。
- **参数**：
  - `features` (list): 特征列表
  - `cross_num` (int): 交叉层数量
  - `hidden_units` (list): DNN部分隐藏层单元数
  - `cross_parameterization` (str, DCN-V2): 交叉参数化方式，"vector"或"matrix"

#### AFM (Attentional Factorization Machine)
- **简介**：在FM的基础上引入注意力机制，对不同特征交互赋予不同的重要性权重。通过注意力网络自适应地学习特征交互的重要性，能够识别出对预测目标更重要的特征组合。
- **参数**：
  - `features` (list): 特征列表
  - `attention_units` (list): 注意力网络隐藏层单元数
  - `embedding_dim` (int): 特征嵌入维度
  - `dropout_rate` (float): 注意力网络的dropout比率

#### FiBiNET (Feature Importance and Bilinear feature Interaction Network)
- **简介**：通过SENET机制动态学习特征重要性，并使用双线性层进行特征交互。SENET模块帮助模型识别重要特征，双线性交互则提供了比内积更丰富的特征交互方式。
- **参数**：
  - `features` (list): 特征列表
  - `bilinear_type` (str): 双线性层类型，可选"field_all"/"field_each"/"field_interaction"
  - `hidden_units` (list): DNN部分隐藏层单元数
  - `reduction_ratio` (int): SENET模块的压缩比率

### 注意力机制系列

#### DIN (Deep Interest Network)
- **简介**：针对用户兴趣多样性设计的模型，通过注意力机制对用户历史行为进行自适应学习。模型会根据当前候选广告，动态地计算用户历史行为的相关性权重，从而激活用户相关的兴趣，捕捉用户多样化的兴趣爱好。创新性地将注意力机制引入推荐系统，开创了行为序列建模的新范式。
- **参数**：
  - `features` (list): 基础特征列表
  - `behavior_features` (list): 行为特征列表，用于注意力计算
  - `attention_units` (list): 注意力网络隐藏层单元数
  - `hidden_units` (list): DNN部分隐藏层单元数
  - `activation` (str): 激活函数类型

#### DIEN (Deep Interest Evolution Network)
- **简介**：DIN的进阶版本，通过引入兴趣进化层来建模用户兴趣的动态变化过程。使用GRU结构捕捉用户兴趣的演变，并创新性地设计了AUGRU（GRU with Attentional Update Gate）来让兴趣进化过程感知目标物品。此外还包含辅助损失来监督兴趣抽取层的训练。这种设计不仅能捕捉用户兴趣的动态变化，还能建模兴趣的时序依赖关系。
- **参数**：
  - `features` (list): 基础特征列表
  - `behavior_features` (list): 行为特征列表
  - `interest_units` (list): 兴趣抽取层单元数
  - `gru_type` (str): GRU类型，"AUGRU"或"AIGRU"
  - `hidden_units` (list): DNN部分隐藏层单元数

#### BST (Behavior Sequence Transformer)
- **简介**：将Transformer架构引入推荐系统的开创性工作，用于建模用户行为序列。通过自注意力机制，模型可以直接计算行为序列中任意两个行为之间的关系，克服了RNN类模型在处理长序列时的局限。Position embedding帮助模型感知行为的时序信息，多头注意力机制则让模型能够从多个角度理解用户行为模式。
- **参数**：
  - `features` (list): 基础特征列表
  - `behavior_features` (list): 行为特征列表
  - `num_heads` (int): 注意力头数
  - `num_layers` (int): Transformer层数
  - `hidden_size` (int): 隐藏层维度
  - `dropout_rate` (float): Dropout比率

#### EDCN (Enhancing Explicit and Implicit Feature Interactions)
- **简介**：增强显式和隐式特征交互的深度交叉网络。通过设计新的交叉网络结构，同时考虑特征的显式和隐式交互。引入门控机制来调控不同阶特征交互的重要性，并使用残差连接来缓解深层网络的训练问题。
- **参数**：
  - `features` (list): 特征列表
  - `cross_num` (int): 交叉层数量
  - `hidden_units` (list): DNN部分隐藏层单元数
  - `gate_type` (str): 门控类型，"FGU"或"BGU"

## 多任务模型 (Multi-task Models)

多任务模型通过联合学习多个相关任务，实现知识共享和迁移，提高模型整体性能。

### SharedBottom
- **简介**：最基础的多任务学习模型，在底层网络共享参数来提取通用特征表示。共享层学习任务间的共性特征，而任务特定层则负责学习每个任务的个性化特征。这种简单而有效的结构为多任务学习奠定了基础。
- **参数**：
  - `features` (list): 特征列表
  - `hidden_units` (list): 共享网络隐藏层单元数
  - `task_hidden_units` (list): 任务特定网络隐藏层单元数
  - `num_tasks` (int): 任务数量
  - `task_types` (list): 任务类型列表

### ESMM (Entire Space Multi-Task Model)
- **简介**：阿里巴巴提出的创新多任务模型，专门解决推荐系统中的样本选择偏差问题。通过CVR和CTR任务的联合建模，在完整空间上进行参数学习。模型的核心创新在于引入了CTR作为辅助任务，并通过任务间的乘积关系优化CVR预估。这种设计不仅解决了传统CVR预估中的样本选择偏差，还提供了无偏的CTR和CTCVR预估。
- **参数**：
  - `features` (list): 特征列表
  - `hidden_units` (list): 隐藏层单元数列表
  - `tower_units` (list): 任务塔层单元数列表
  - `embedding_dim` (int): 特征嵌入维度

### MMoE (Multi-gate Mixture-of-Experts)
- **简介**：Google提出的多任务学习模型，通过专家机制和任务相关的门控网络来实现任务间的软参数共享。每个专家网络可以学习特定的特征转换，而门控网络则为每个任务动态分配专家的重要性。这种设计让模型能够根据任务的需求灵活地组合专家知识，有效处理任务间的差异性。
- **参数**：
  - `features` (list): 特征列表
  - `expert_units` (list): 专家网络隐藏层单元数
  - `num_experts` (int): 专家数量
  - `num_tasks` (int): 任务数量
  - `expert_activation` (str): 专家网络激活函数
  - `gate_activation` (str): 门控网络激活函数

### PLE (Progressive Layered Extraction)
- **简介**：对MMoE的改进版本，通过分层提取的方式更好地建模任务间的关系。引入了任务特定专家和共享专家的概念，并通过多层级的专家网络实现渐进式特征提取。每一层都包含特定任务的专家和共享专家，让模型能够同时学习任务的共性和个性。这种渐进式的设计提高了模型对知识提取和迁移的能力。
- **参数**：
  - `features` (list): 特征列表
  - `expert_units` (list): 专家网络单元数
  - `num_experts` (int): 每层专家数量
  - `num_layers` (int): 层数
  - `num_shared_experts` (int): 共享专家数量
  - `task_types` (list): 任务类型列表