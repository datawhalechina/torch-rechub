---
title: 多任务模型
description: Torch-RecHub 多任务模型详细介绍
---

# 多任务模型

多任务学习是一种机器学习范式，通过同时学习多个相关任务来提高模型的泛化能力和性能。在推荐系统中，多任务模型常用于同时优化点击率（CTR）、转化率（CVR）、用户留存等多个相关目标。

## 1. SharedBottom

### 功能描述

SharedBottom 是一种经典的多任务学习模型，所有任务共享底层网络，上层有各自的任务特定网络。

### 核心原理

- **共享底层**：所有任务共享一个底层神经网络，学习任务间的共享表示
- **任务特定顶层**：每个任务有自己的顶层网络，学习任务特定的表示
- **联合训练**：所有任务同时训练，通过反向传播更新共享底层和任务特定顶层

### 使用方法

```python
from torch_rechub.models.multi_task import SharedBottom
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 定义特征
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

# 创建模型
model = SharedBottom(
    features=common_features,
    task_types=["classification", "classification"],  # 两个分类任务
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 任务1的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 任务2的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | None |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | None |
| bottom_params | dict | 共享底层网络参数 | None |
| tower_params_list | list | 任务特定顶层网络参数列表 | None |

### 适用场景

- 任务相关性强的场景
- 数据量有限的场景
- 计算资源有限的场景

## 2. ESMM

### 功能描述

ESMM（Entire Space Multi-Task Model）是一种用于解决样本选择偏差问题的多任务学习模型，特别适用于CTR和CVR联合优化场景。

### 论文引用

```
Xiao, Jun, et al. "Entire space multi-task model: An effective approach for estimating post-click conversion rate." Proceedings of the 41st international ACM SIGIR conference on research & development in information retrieval. 2018.
```

### 核心原理

- **全空间建模**：在整个样本空间中建模CTR和CVR，避免样本选择偏差
- **级联关系**：利用CTR和CVR的级联关系（CTCVR = CTR * CVR）
- **共享底层**：CTR和CVR任务共享底层网络
- **任务特定顶层**：每个任务有自己的顶层网络

### 使用方法

```python
from torch_rechub.models.multi_task import ESMM

# 创建模型
model = ESMM(
    features=common_features,
    task_types=["classification", "classification"],  # CTR和CVR都是分类任务
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # CTR任务的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # CVR任务的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | None |
| task_types | list | 任务类型列表，支持 "classification" | None |
| bottom_params | dict | 共享底层网络参数 | None |
| tower_params_list | list | 任务特定顶层网络参数列表 | None |

### 适用场景

- 点击率（CTR）和转化率（CVR）联合优化
- 存在样本选择偏差的场景
- 电商推荐场景

## 3. MMOE

### 功能描述

MMOE（Multi-gate Mixture-of-Experts）是一种多门控专家混合模型，通过多个专家网络和门控机制，为不同任务学习不同的专家组合。

### 论文引用

```
Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

### 核心原理

- **专家网络**：多个独立的专家网络，每个专家网络学习不同的特征表示
- **门控机制**：每个任务有自己的门控网络，动态选择专家网络的组合
- **任务特定顶层**：每个任务有自己的顶层网络
- **联合训练**：所有任务同时训练，更新专家网络、门控网络和任务特定顶层

### 使用方法

```python
from torch_rechub.models.multi_task import MMOE

# 创建模型
model = MMOE(
    features=common_features,
    task_types=["classification", "regression"],  # 分类任务和回归任务
    n_expert=8,  # 专家网络数量
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 分类任务的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 回归任务的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | None |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | None |
| n_expert | int | 专家网络数量 | None |
| expert_params | dict | 专家网络参数 | None |
| tower_params_list | list | 任务特定顶层网络参数列表 | None |

### 适用场景

- 任务间存在冲突的场景
- 任务数量较多的场景
- 需要动态调整任务权重的场景

## 4. PLE

### 功能描述

PLE（Progressive Layered Extraction）是一种渐进式分层提取模型，引入了任务特定专家和共享专家，缓解了负迁移问题。

### 论文引用

```
Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations." Fourteenth ACM Conference on Recommender Systems. 2020.
```

### 核心原理

- **分层结构**：包含多个PLE层，每层有任务特定专家和共享专家
- **任务特定专家**：只对特定任务有用的专家网络
- **共享专家**：对所有任务都有用的专家网络
- **门控机制**：每个任务有自己的门控网络，选择专家网络的组合
- **渐进式提取**：通过多层PLE结构，渐进式提取任务特定和共享表示

### 使用方法

```python
from torch_rechub.models.multi_task import PLE

# 创建模型
model = PLE(
    features=common_features,
    task_types=["classification", "regression"],
    n_level=2,  # PLE层数
    n_expert_specific=4,  # 每层任务特定专家数量
    n_expert_shared=4,  # 每层共享专家数量
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 分类任务的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 回归任务的顶层参数
    ]
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | None |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | None |
| n_level | int | PLE层数 | None |
| n_expert_specific | int | 每层任务特定专家数量 | None |
| n_expert_shared | int | 每层共享专家数量 | None |
| expert_params | dict | 专家网络参数 | None |
| tower_params_list | list | 任务特定顶层网络参数列表 | None |

### 适用场景

- 任务间存在较强冲突的场景
- 需要缓解负迁移问题的场景
- 复杂多任务学习场景

## 5. AITM

### 功能描述

AITM（Adaptive Information Transfer Multi-task）是一种自适应信息迁移多任务模型，能够自动学习任务间的信息迁移关系。

### 论文引用

```
Tang, Jiaxi, et al. "Learning Task Relationships in Multi-task Learning with Adaptive Information Transfer." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

### 核心原理

- **信息迁移机制**：自动学习任务间的信息迁移关系
- **任务特定网络**：每个任务有自己的网络
- **注意力机制**：使用注意力机制自适应地传递任务间的信息
- **联合训练**：所有任务同时训练，更新任务特定网络和信息迁移机制

### 使用方法

```python
from torch_rechub.models.multi_task import AITM

# 创建模型
model = AITM(
    features=common_features,
    task_types=["classification", "classification"],
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # 任务1的顶层参数
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # 任务2的顶层参数
    ],
    attention_params={"attention_dim": 64, "dropout": 0.2}  # 注意力机制参数
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| features | list | 所有任务共享的特征列表 | None |
| task_types | list | 任务类型列表，支持 "classification" 和 "regression" | None |
| bottom_params | dict | 共享底层网络参数 | None |
| tower_params_list | list | 任务特定顶层网络参数列表 | None |
| attention_params | dict | 注意力机制参数 | None |

### 适用场景

- 任务间存在依赖关系的场景
- 需要自适应信息迁移的场景
- 复杂多任务学习场景

## 6. 模型比较

| 模型 | 复杂度 | 表达能力 | 计算效率 | 适用场景 |
| --- | --- | --- | --- | --- |
| SharedBottom | 低 | 中 | 高 | 任务相关性强、资源有限 |
| ESMM | 低 | 中 | 高 | CTR/CVR联合优化、样本选择偏差 |
| MMOE | 中 | 高 | 中 | 任务冲突、多任务场景 |
| PLE | 高 | 高 | 低 | 复杂多任务、负迁移缓解 |
| AITM | 中 | 高 | 中 | 任务依赖、自适应信息迁移 |

## 7. 使用建议

1. **根据任务关系选择模型**：
   - 任务相关性强时推荐使用 SharedBottom
   - 任务间存在冲突时推荐使用 MMOE 或 PLE
   - 任务间存在依赖关系时推荐使用 AITM
   - 处理CTR/CVR联合优化时推荐使用 ESMM

2. **根据计算资源选择模型**：
   - 计算资源有限时推荐使用 SharedBottom 或 ESMM
   - 计算资源充足时可以尝试 MMOE、PLE 或 AITM

3. **根据数据量选择模型**：
   - 数据量较小时推荐使用简单模型（如 SharedBottom）
   - 数据量较大时可以尝试更复杂的模型（如 PLE、AITM）

4. **多任务权重调整**：
   - 可以通过调整损失权重来平衡不同任务的重要性
   - 尝试使用自适应权重方法，如 UWLLoss、GradNorm 等

## 8. 代码示例：完整的多任务模型训练流程

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 1. 定义特征
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1)
]

# 2. 准备数据
# 假设 x 包含所有特征，y 包含两个任务的标签
x = {
    "user_id": user_id_data,
    "city": city_data,
    "age": age_data,
    "income": income_data
}
y = {
    "task1": task1_labels,  # 点击率标签
    "task2": task2_labels   # 转化率标签
}

# 3. 创建数据生成器
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

# 4. 创建模型
model = MMOE(
    features=common_features,
    task_types=["classification", "classification"],
    n_expert=8,
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}
    ]
)

# 5. 创建训练器
trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    adaptive_params={"method": "uwl"},  # 使用自适应损失权重
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/mmoe"
)

# 6. 训练模型
trainer.fit(train_dl, val_dl)

# 7. 评估模型
scores = trainer.evaluate(trainer.model, test_dl)
print(f"Task 1 AUC: {scores[0]}")
print(f"Task 2 AUC: {scores[1]}")

# 8. 导出ONNX模型
trainer.export_onnx("mmoe.onnx")

# 9. 模型预测
preds = trainer.predict(trainer.model, test_dl)
print(f"Predictions shape: {np.array(preds).shape}")
```

## 9. 常见问题与解决方案

### Q: 如何处理不同任务的数据分布差异？
A: 可以尝试以下方法：
- 对每个任务的数据进行标准化或归一化
- 使用任务特定的Embedding层
- 调整任务权重，平衡不同任务的重要性
- 使用自适应权重方法，如UWLLoss、GradNorm等

### Q: 如何缓解多任务学习中的负迁移问题？
A: 可以尝试以下方法：
- 使用 PLE 模型，引入任务特定专家和共享专家
- 使用注意力机制，自适应地选择任务间的信息迁移
- 减少共享层的深度，增加任务特定层的深度
- 尝试使用模型选择策略，选择合适的任务组合

### Q: 如何选择合适的任务组合？
A: 可以考虑以下因素：
- 任务间的相关性：选择相关性较高的任务组合
- 任务的重要性：选择对业务更重要的任务
- 数据量：选择数据量充足的任务
- 计算资源：考虑模型的计算复杂度

### Q: 如何评估多任务模型的效果？
A: 常用的评估指标包括：
- 单个任务的评估指标（如 AUC、F1、RMSE 等）
- 所有任务指标的加权平均
- 帕累托最优分析：在多个任务间寻找最佳平衡点
- 业务指标：最终的业务效果（如点击率提升、转化率提升等）

### Q: 如何调整多任务模型的超参数？
A: 可以尝试以下方法：
- 网格搜索或随机搜索：调整专家数量、网络深度、dropout率等
- 贝叶斯优化：更高效地搜索最优超参数
- 迁移学习：从简单模型的超参数开始调整
- 经验法则：专家数量一般选择 4-16 个，网络深度选择 2-4 层

## 10. 多任务学习在推荐系统中的应用场景

1. **电商推荐**：
   - 同时优化点击率（CTR）、转化率（CVR）、客单价
   - 优化商品推荐、广告推荐、个性化搜索

2. **内容推荐**：
   - 同时优化点击率、阅读时长、点赞率、评论率
   - 优化新闻推荐、视频推荐、音乐推荐

3. **社交媒体**：
   - 同时优化好友推荐、内容推荐、广告推荐
   - 优化用户留存、活跃度、互动率

4. **金融推荐**：
   - 同时优化贷款申请率、还款率、逾期率
   - 优化理财产品推荐、信用卡推荐

## 11. 未来发展趋势

1. **动态任务权重调整**：
   - 自适应地调整不同任务的权重，根据任务的重要性和难度动态变化

2. **跨领域多任务学习**：
   - 利用不同领域的数据和任务，提高模型的泛化能力

3. **层次化多任务学习**：
   - 构建任务间的层次关系，更有效地利用任务间的结构信息

4. **多模态多任务学习**：
   - 结合文本、图像、音频等多种模态，同时学习多个任务

5. **大规模多任务学习**：
   - 支持数百个甚至数千个任务的同时学习，处理更复杂的推荐场景

多任务学习在推荐系统中具有广阔的应用前景，能够充分利用任务间的关系，提高模型的泛化能力和性能。Torch-RecHub 提供了多种先进的多任务模型，方便开发者根据业务需求选择和使用。