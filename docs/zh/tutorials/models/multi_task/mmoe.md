---
title: MMOE 使用示例
description: Multi-gate Mixture-of-Experts 多任务模型完整使用教程
---

# MMOE 使用示例

## 1. 模型简介与适用场景

MMOE（Multi-gate Mixture-of-Experts）是 Google 在 KDD'2018 上提出的多任务学习模型。通过**多个专家网络**和**门控机制**，让不同任务可以灵活地选择使用不同的专家组合，有效缓解多任务学习中的**负迁移问题**。

**论文**: [Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts](https://dl.acm.org/doi/pdf/10.1145/3219819.3220007)

### 模型结构

- **多个 Expert 网络**: 共享的底层专家网络，每个专家学习不同的特征表示
- **多个 Gate 网络**: 每个任务一个门控，学习如何组合不同专家的输出
- **多个 Tower 网络**: 每个任务一个 Tower，基于门控输出进行最终预测

### 适用场景

- 多目标优化（如同时预测 CTR 和 CVR）
- 任务之间存在冲突但又有相关性的场景
- 需要共享底层表示但允许任务差异化的场景
- 电商推荐（点击、购买、加购多目标同时优化）

---

## 2. 数据准备与预处理

本示例使用 **Ali-CCP** (阿里巴巴点击转化预测) 数据集，同时预测用户的**点击率 (CTR)** 和**转化率 (CVR)**。

### 2.1 加载和处理数据

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# 加载预处理好的 Ali-CCP 采样数据
df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")
print(f"训练集: {len(df_train)}, 验证集: {len(df_val)}, 测试集: {len(df_test)}")

# 合并数据以统一特征处理
train_idx = df_train.shape[0]
val_idx = train_idx + df_val.shape[0]
data = pd.concat([df_train, df_val, df_test], axis=0)

# 重命名标签列
data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)
```

### 2.2 定义特征和标签

```python
col_names = data.columns.tolist()

# 区分连续特征和离散特征
dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
sparse_cols = [
    col for col in col_names
    if col not in dense_cols and col not in ['cvr_label', 'ctr_label']
]

# 定义特征
features = [
    SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols
] + [
    DenseFeature(col) for col in dense_cols
]

# 定义多任务标签 (CVR, CTR)
label_cols = ['cvr_label', 'ctr_label']
used_cols = sparse_cols + dense_cols
```

### 2.3 构建训练/验证/测试集

```python
x_train = {name: data[name].values[:train_idx] for name in used_cols}
y_train = data[label_cols].values[:train_idx]

x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
y_val = data[label_cols].values[train_idx:val_idx]

x_test = {name: data[name].values[val_idx:] for name in used_cols}
y_test = data[label_cols].values[val_idx:]

# 创建 DataLoader
dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val, y_val=y_val,
    x_test=x_test, y_test=y_test,
    batch_size=1024
)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.multi_task import MMOE

model = MMOE(
    features=features,
    task_types=["classification", "classification"],   # 两个分类任务
    n_expert=8,                                         # 专家数量
    expert_params={"dims": [16]},                       # 专家网络参数
    tower_params_list=[{"dims": [8]}, {"dims": [8]}]    # 每个任务的 Tower 参数
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 全部特征列表 | Dense + Sparse 特征 |
| `task_types` | `list[str]` | 每个任务的类型 | `"classification"` 或 `"regression"` |
| `n_expert` | `int` | Expert 网络数量 | 4 ~ 16 |
| `expert_params.dims` | `list[int]` | Expert 网络维度 | `[16]` 或 `[32, 16]` |
| `tower_params_list` | `list[dict]` | 每个任务的 Tower 参数列表 | 每个 Tower: `{"dims": [8]}` |

> **Expert 数量选择**: Expert 数量一般设为 4~16。太少则任务间无法差异化，太多则参数冗余。经验上 `n_expert=8` 是一个较好的起点。

---

## 4. 训练过程与代码示例

### 4.1 训练模型

```python
from torch_rechub.trainers import MTLTrainer

torch.manual_seed(2022)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={
        "lr": 1e-3,
        "weight_decay": 1e-4
    },
    n_epoch=50,
    earlystop_patience=30,
    device="cpu",
    model_path="./saved/mmoe"
)

mtl_trainer.fit(train_dl, val_dl)
```

### 4.2 使用自适应损失权重 (可选)

当多个任务的 loss 量级差异较大时，可以使用自适应权重：

```python
mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    adaptive_params={"method": "uwl"},   # Uncertainty Weight Loss
    n_epoch=50,
    earlystop_patience=30,
    device="cpu",
    model_path="./saved/mmoe"
)
```

---

## 5. 模型评估与结果分析

### 5.1 多任务评估

```python
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC: {auc}")
# 输出: [cvr_auc, ctr_auc]
```

评估结果是一个列表，按 `task_types` 的顺序返回每个任务的 AUC。

### 5.2 预期性能

| 任务 | 预期 AUC (Sample) |
|------|-------------------|
| CVR  | 0.55 ~ 0.65       |
| CTR  | 0.58 ~ 0.68       |

> Sample 数据量较小，实际 Full 数据集上效果会更好。

---

## 6. 参数调优建议

### 6.1 关键调优点

1. **Expert 数量 vs 任务数量**:
   - 经验法则: `n_expert >= 2 * n_task`
   - 2 个任务时推荐 4~8 个 Expert

2. **Expert 维度**:
   - Expert 不宜过深，`[16]` 或 `[32, 16]` 通常足够
   - 配合较浅的 Tower `[8]`

3. **任务权重**:
   - 如果某个任务的 AUC 明显低于其他任务，可以尝试 `adaptive_params={"method": "uwl"}` 自适应权重
   - 或手动调整任务 loss 的权重比例

4. **学习率**: 多任务场景建议 `1e-3` ~ `1e-4`

### 6.2 MMOE vs 其他多任务模型对比

| 模型 | 参数共享方式 | 适用场景 |
|------|------------|---------|
| SharedBottom | 完全共享底层 | 任务高度相关 |
| MMOE | 门控选择共享 | 任务有冲突 |
| PLE | 分层渐进提取 | 复杂多任务 |

---

## 7. 常见问题与解决方案

### Q1: 某个任务的 AUC 持续较低？
可能存在**负迁移**（一个任务的训练损害了另一个任务）。尝试：
- 增加 Expert 数量（让门控有更多选择空间）
- 使用自适应损失权重 `adaptive_params={"method": "uwl"}`
- 考虑使用 PLE 模型（每个任务有独立 Expert）

### Q2: task_types 支持回归任务吗？
是的，设置 `task_types=["classification", "regression"]` 即可。回归任务使用 MSE Loss，分类任务使用 BCE Loss。

### Q3: 如何添加更多任务？
在 `task_types`、`tower_params_list` 和训练标签中添加对应任务即可：

```python
model = MMOE(
    features=features,
    task_types=["classification", "classification", "regression"],  # 3 个任务
    n_expert=8,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}, {"dims": [8]}]
)
```

### Q4: n_expert 增加后训练变慢？
Expert 数量增加会线性增加计算量。可以减小 Expert 维度来平衡效率和效果。

---

## 完整代码

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)

    # 1. 加载数据
    df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
    df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
    df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

    train_idx = df_train.shape[0]
    val_idx = train_idx + df_val.shape[0]
    data = pd.concat([df_train, df_val, df_test], axis=0)
    data.rename(columns={'purchase': 'cvr_label', 'click': 'ctr_label'}, inplace=True)

    # 2. 定义特征
    dense_cols = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_cols = [
        col for col in data.columns
        if col not in dense_cols and col not in ['cvr_label', 'ctr_label']
    ]

    features = [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] \
        + [DenseFeature(col) for col in dense_cols]

    label_cols = ['cvr_label', 'ctr_label']
    used_cols = sparse_cols + dense_cols

    # 3. 构建数据集
    x_train = {name: data[name].values[:train_idx] for name in used_cols}
    y_train = data[label_cols].values[:train_idx]
    x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
    y_val = data[label_cols].values[train_idx:val_idx]
    x_test = {name: data[name].values[val_idx:] for name in used_cols}
    y_test = data[label_cols].values[val_idx:]

    dg = DataGenerator(x_train, y_train)
    train_dl, val_dl, test_dl = dg.generate_dataloader(
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=1024
    )

    # 4. 创建模型
    model = MMOE(
        features=features,
        task_types=["classification", "classification"],
        n_expert=8,
        expert_params={"dims": [16]},
        tower_params_list=[{"dims": [8]}, {"dims": [8]}]
    )

    # 5. 训练
    mtl_trainer = MTLTrainer(
        model,
        task_types=["classification", "classification"],
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
        n_epoch=50,
        earlystop_patience=30,
        device="cpu",
        model_path="./saved/mmoe"
    )
    mtl_trainer.fit(train_dl, val_dl)

    # 6. 评估
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
    print(f"Test AUC (CVR, CTR): {auc}")


if __name__ == "__main__":
    main()
```
