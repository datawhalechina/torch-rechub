---
title: Multi-Task Models
description: Torch-RecHub multi-task learning models detailed introduction
---

# Multi-Task Models

Multi-task learning is a machine learning paradigm that improves model generalization and performance by learning multiple related tasks simultaneously. In recommendation systems, multi-task models are commonly used to jointly optimize multiple related objectives such as click-through rate (CTR), conversion rate (CVR), and user retention.

## 1. SharedBottom

### Description

SharedBottom is a classic multi-task learning model where all tasks share a bottom network, with task-specific networks on top.

### Core Principles

- **Shared Bottom**: All tasks share a bottom neural network to learn shared representations
- **Task-specific Towers**: Each task has its own tower network to learn task-specific representations
- **Joint Training**: All tasks are trained simultaneously, updating shared bottom and task-specific towers through backpropagation

### Usage

```python
from torch_rechub.models.multi_task import SharedBottom
from torch_rechub.basic.features import SparseFeature, DenseFeature

# Define features
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    DenseFeature(name="age", embed_dim=1)
]

# Create model
model = SharedBottom(
    features=common_features,
    task_types=["classification", "classification"],  # Two classification tasks
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # Task 1 tower params
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Task 2 tower params
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Shared feature list for all tasks | None |
| task_types | list | Task type list, supports "classification" and "regression" | None |
| bottom_params | dict | Shared bottom network parameters | None |
| tower_params_list | list | Task-specific tower network parameters list | None |

### Use Cases

- Scenarios with highly related tasks
- Limited data scenarios
- Limited computational resources

## 2. ESMM

### Description

ESMM (Entire Space Multi-Task Model) is a multi-task learning model designed to address sample selection bias, particularly suitable for joint CTR and CVR optimization.

### Paper Reference

```
Xiao, Jun, et al. "Entire space multi-task model: An effective approach for estimating post-click conversion rate." Proceedings of the 41st international ACM SIGIR conference on research & development in information retrieval. 2018.
```

### Core Principles

- **Full-space Modeling**: Models CTR and CVR in the entire sample space to avoid sample selection bias
- **Cascade Relationship**: Leverages the cascade relationship between CTR and CVR (CTCVR = CTR * CVR)
- **Shared Bottom**: CTR and CVR tasks share the bottom network
- **Task-specific Towers**: Each task has its own tower network

### Usage

```python
from torch_rechub.models.multi_task import ESMM

# Create model
model = ESMM(
    features=common_features,
    task_types=["classification", "classification"],  # Both CTR and CVR are classification tasks
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # CTR tower params
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # CVR tower params
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Shared feature list for all tasks | None |
| task_types | list | Task type list, supports "classification" | None |
| bottom_params | dict | Shared bottom network parameters | None |
| tower_params_list | list | Task-specific tower network parameters list | None |

### Use Cases

- Joint CTR and CVR optimization
- Scenarios with sample selection bias
- E-commerce recommendation

## 3. MMOE

### Description

MMOE (Multi-gate Mixture-of-Experts) is a multi-gate mixture-of-experts model that learns different expert combinations for different tasks through multiple expert networks and gating mechanisms.

### Paper Reference

```
Ma, Jiaqi, et al. "Modeling task relationships in multi-task learning with multi-gate mixture-of-experts." Proceedings of the 24th ACM SIGKDD international conference on knowledge discovery & data mining. 2018.
```

### Core Principles

- **Expert Networks**: Multiple independent expert networks, each learning different feature representations
- **Gating Mechanism**: Each task has its own gating network to dynamically select expert combinations
- **Task-specific Towers**: Each task has its own tower network
- **Joint Training**: All tasks are trained simultaneously, updating expert networks, gating networks, and task-specific towers

### Usage

```python
from torch_rechub.models.multi_task import MMOE

# Create model
model = MMOE(
    features=common_features,
    task_types=["classification", "regression"],  # Classification and regression tasks
    n_expert=8,  # Number of expert networks
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # Classification tower params
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Regression tower params
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Shared feature list for all tasks | None |
| task_types | list | Task type list, supports "classification" and "regression" | None |
| n_expert | int | Number of expert networks | None |
| expert_params | dict | Expert network parameters | None |
| tower_params_list | list | Task-specific tower network parameters list | None |

### Use Cases

- Scenarios with task conflicts
- Multiple tasks scenarios
- Scenarios requiring dynamic task weight adjustment

## 4. PLE

### Description

PLE (Progressive Layered Extraction) is a progressive layered extraction model that introduces task-specific experts and shared experts to mitigate negative transfer.

### Paper Reference

```
Tang, Hongyan, et al. "Progressive layered extraction (ple): A novel multi-task learning (mtl) model for personalized recommendations." Fourteenth ACM Conference on Recommender Systems. 2020.
```

### Core Principles

- **Layered Structure**: Contains multiple PLE layers, each with task-specific and shared experts
- **Task-specific Experts**: Expert networks useful only for specific tasks
- **Shared Experts**: Expert networks useful for all tasks
- **Gating Mechanism**: Each task has its own gating network to select expert combinations
- **Progressive Extraction**: Progressively extracts task-specific and shared representations through multi-layer PLE structure

### Usage

```python
from torch_rechub.models.multi_task import PLE

# Create model
model = PLE(
    features=common_features,
    task_types=["classification", "regression"],
    n_level=2,  # Number of PLE layers
    n_expert_specific=4,  # Number of task-specific experts per layer
    n_expert_shared=4,  # Number of shared experts per layer
    expert_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # Classification tower params
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Regression tower params
    ]
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Shared feature list for all tasks | None |
| task_types | list | Task type list, supports "classification" and "regression" | None |
| n_level | int | Number of PLE layers | None |
| n_expert_specific | int | Number of task-specific experts per layer | None |
| n_expert_shared | int | Number of shared experts per layer | None |
| expert_params | dict | Expert network parameters | None |
| tower_params_list | list | Task-specific tower network parameters list | None |

### Use Cases

- Scenarios with strong task conflicts
- Scenarios requiring negative transfer mitigation
- Complex multi-task learning scenarios

## 5. AITM

### Description

AITM (Adaptive Information Transfer Multi-task) is an adaptive information transfer multi-task model that automatically learns information transfer relationships between tasks.

### Paper Reference

```
Tang, Jiaxi, et al. "Learning Task Relationships in Multi-task Learning with Adaptive Information Transfer." Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining. 2021.
```

### Core Principles

- **Information Transfer Mechanism**: Automatically learns information transfer relationships between tasks
- **Task-specific Networks**: Each task has its own network
- **Attention Mechanism**: Uses attention mechanism to adaptively transfer information between tasks
- **Joint Training**: All tasks are trained simultaneously, updating task-specific networks and information transfer mechanism

### Usage

```python
from torch_rechub.models.multi_task import AITM

# Create model
model = AITM(
    features=common_features,
    task_types=["classification", "classification"],
    bottom_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
    tower_params_list=[
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"},  # Task 1 tower params
        {"dims": [64, 32], "dropout": 0.2, "activation": "relu"}   # Task 2 tower params
    ],
    attention_params={"attention_dim": 64, "dropout": 0.2}  # Attention mechanism params
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| features | list | Shared feature list for all tasks | None |
| task_types | list | Task type list, supports "classification" and "regression" | None |
| bottom_params | dict | Shared bottom network parameters | None |
| tower_params_list | list | Task-specific tower network parameters list | None |
| attention_params | dict | Attention mechanism parameters | None |

### Use Cases

- Scenarios with task dependencies
- Scenarios requiring adaptive information transfer
- Complex multi-task learning scenarios

## 6. Model Comparison

| Model | Complexity | Expressiveness | Efficiency | Use Cases |
| --- | --- | --- | --- | --- |
| SharedBottom | Low | Medium | High | Highly related tasks, limited resources |
| ESMM | Low | Medium | High | CTR/CVR joint optimization, sample selection bias |
| MMOE | Medium | High | Medium | Task conflicts, multi-task scenarios |
| PLE | High | High | Low | Complex multi-task, negative transfer mitigation |
| AITM | Medium | High | Medium | Task dependencies, adaptive information transfer |

## 7. Usage Recommendations

1. **Choose models based on task relationships**:
   - For highly related tasks, use SharedBottom
   - For task conflicts, use MMOE or PLE
   - For task dependencies, use AITM
   - For CTR/CVR joint optimization, use ESMM

2. **Choose models based on computational resources**:
   - With limited resources, use SharedBottom or ESMM
   - With sufficient resources, try MMOE, PLE, or AITM

3. **Choose models based on data volume**:
   - With limited data, use simple models (e.g., SharedBottom)
   - With large data, try more complex models (e.g., PLE, AITM)

4. **Multi-task weight adjustment**:
   - Adjust loss weights to balance task importance
   - Try adaptive weight methods like UWLLoss, GradNorm, etc.

## 8. Complete Training Example

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator
from torch_rechub.basic.features import SparseFeature, DenseFeature

# 1. Define features
common_features = [
    SparseFeature(name="user_id", vocab_size=10000, embed_dim=32),
    SparseFeature(name="city", vocab_size=100, embed_dim=16),
    DenseFeature(name="age", embed_dim=1),
    DenseFeature(name="income", embed_dim=1)
]

# 2. Prepare data
# Assume x contains all features, y contains labels for two tasks
x = {
    "user_id": user_id_data,
    "city": city_data,
    "age": age_data,
    "income": income_data
}
y = {
    "task1": task1_labels,  # CTR labels
    "task2": task2_labels   # CVR labels
}

# 3. Create data generator
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

# 4. Create model
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

# 5. Create trainer
trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    adaptive_params={"method": "uwl"},  # Use adaptive loss weighting
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/mmoe"
)

# 6. Train model
trainer.fit(train_dl, val_dl)

# 7. Evaluate model
scores = trainer.evaluate(trainer.model, test_dl)
print(f"Task 1 AUC: {scores[0]}")
print(f"Task 2 AUC: {scores[1]}")

# 8. Export ONNX model
trainer.export_onnx("mmoe.onnx")

# 9. Model prediction
preds = trainer.predict(trainer.model, test_dl)
print(f"Predictions shape: {np.array(preds).shape}")
```

## 9. FAQ

### Q: How to handle data distribution differences between tasks?
A: Try the following approaches:
- Standardize or normalize data for each task
- Use task-specific embedding layers
- Adjust task weights to balance task importance
- Use adaptive weight methods like UWLLoss, GradNorm, etc.

### Q: How to mitigate negative transfer in multi-task learning?
A: Try the following approaches:
- Use PLE model with task-specific and shared experts
- Use attention mechanism to adaptively select information transfer between tasks
- Reduce shared layer depth, increase task-specific layer depth
- Try model selection strategies to choose appropriate task combinations

### Q: How to choose appropriate task combinations?
A: Consider the following factors:
- Task correlation: Choose highly correlated task combinations
- Task importance: Choose tasks more important to business
- Data volume: Choose tasks with sufficient data
- Computational resources: Consider model computational complexity

### Q: How to evaluate multi-task model performance?
A: Common evaluation metrics include:
- Individual task metrics (e.g., AUC, F1, RMSE)
- Weighted average of all task metrics
- Pareto optimal analysis: Find the best balance among multiple tasks
- Business metrics: Final business impact (e.g., CTR improvement, CVR improvement)

### Q: How to tune multi-task model hyperparameters?
A: Try the following approaches:
- Grid search or random search: Tune expert count, network depth, dropout rate, etc.
- Bayesian optimization: More efficient hyperparameter search
- Transfer learning: Start from simple model hyperparameters
- Rules of thumb: Expert count typically 4-16, network depth 2-4 layers

## 10. Multi-Task Learning Applications in Recommendation Systems

1. **E-commerce Recommendation**:
   - Jointly optimize CTR, CVR, and average order value
   - Optimize product recommendation, ad recommendation, personalized search

2. **Content Recommendation**:
   - Jointly optimize CTR, reading time, like rate, comment rate
   - Optimize news recommendation, video recommendation, music recommendation

3. **Social Media**:
   - Jointly optimize friend recommendation, content recommendation, ad recommendation
   - Optimize user retention, activity, engagement rate

4. **Financial Recommendation**:
   - Jointly optimize loan application rate, repayment rate, default rate
   - Optimize financial product recommendation, credit card recommendation

## 11. Future Trends

1. **Dynamic Task Weight Adjustment**:
   - Adaptively adjust task weights based on task importance and difficulty

2. **Cross-domain Multi-task Learning**:
   - Leverage data and tasks from different domains to improve model generalization

3. **Hierarchical Multi-task Learning**:
   - Build hierarchical relationships between tasks to better utilize structural information

4. **Multi-modal Multi-task Learning**:
   - Combine text, image, audio, and other modalities to learn multiple tasks simultaneously

5. **Large-scale Multi-task Learning**:
   - Support learning hundreds or thousands of tasks simultaneously for more complex recommendation scenarios

Multi-task learning has broad application prospects in recommendation systems, effectively leveraging task relationships to improve model generalization and performance. Torch-RecHub provides various advanced multi-task models for developers to choose based on business requirements.
