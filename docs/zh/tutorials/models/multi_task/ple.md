---
title: PLE 使用示例
description: Progressive Layered Extraction (PLE) 多任务模型完整使用教程
---

# PLE 使用示例

## 1. 模型简介与适用场景

PLE (Progressive Layered Extraction) 是腾讯在 RecSys'2020 提出的多任务学习模型。PLE 通过**渐进式分层提取** (Progressive Layered Extraction) 解决了多任务学习中的 **seesaw 现象**（即优化一个任务时损害另一个任务的表现），采用 **Customized Gate Control (CGC)** 机制，为每个任务设置专属的 Expert 和 Shared Expert，并通过 Gate 网络自适应地融合。

**论文**: [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)

### 模型结构

<div align="center">
  <img src="/img/models/ple_arch.png" alt="PLE Model Architecture" width="600"/>
</div>

- **Task-Specific Experts**: 每个任务有自己专属的 Expert 网络
- **Shared Experts**: 所有任务共享的 Expert 网络
- **Customized Gate (CGC)**: 每个任务的 Gate 网络融合 task-specific experts 和 shared experts 的输出
- **Multi-Level**: 支持多层 CGC 堆叠，实现渐进式特征提取
- **Task Towers**: 每个任务独立的预测 Tower

### 适用场景

- 多目标优化（如：同时优化 CTR + CVR，点击 + 收藏 + 购买）
- 任务之间存在相关性但又有各自独立的需求
- 需要比 MMOE 更强的任务区分能力的场景

---

## 2. 数据准备与预处理

使用 **Ali-CCP** (阿里妈妈点击和转化预测) 数据集，与 MMOE 的数据准备方式相同。

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

# 加载数据
data = pd.read_csv("examples/ranking/data/ali-ccp/ali-ccp_sample.csv")

# 区分特征
col_names = data.columns.values.tolist()
dense_features = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
sparse_features = [col for col in col_names if col not in dense_features and col not in ['ctr_label', 'cvr_label']]

# 填充缺失值
data[sparse_features] = data[sparse_features].fillna('-996')
data[dense_features] = data[dense_features].fillna(0)

# 定义特征
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16) for name in sparse_features]
features = dense_feas + sparse_feas

# 标签（多任务）
y = data[["ctr_label", "cvr_label"]]
del data["ctr_label"]
del data["cvr_label"]
x = data

# DataLoader
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=2048)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.multi_task import PLE

model = PLE(
    features=features,
    task_types=["classification", "classification"],  # 两个分类任务
    n_level=2,                    # CGC 层数
    n_expert_specific=2,          # 每个任务的专属 Expert 数
    n_expert_shared=1,            # 共享 Expert 数
    expert_params={
        "dims": [256, 128],
        "dropout": 0.2
    },
    tower_params_list=[
        {"dims": [64]},           # CTR Tower
        {"dims": [64]}            # CVR Tower
    ]
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 特征列表 | Dense + Sparse |
| `task_types` | `list[str]` | 任务类型列表 | `"classification"` 或 `"regression"` |
| `n_level` | `int` | CGC 层数（Progressive 的层数） | 1 ~ 3 |
| `n_expert_specific` | `int` | 每个任务的专属 Expert 数量 | 1 ~ 4 |
| `n_expert_shared` | `int` | 共享 Expert 数量 | 1 ~ 2 |
| `expert_params` | `dict` | Expert MLP 参数 | `{"dims": [256, 128]}` |
| `tower_params_list` | `list[dict]` | 每个 Task Tower 的 MLP 参数 | 每个任务一个 dict |

> **PLE vs MMOE**: MMOE 所有 Expert 都是共享的，PLE 区分了 task-specific 和 shared experts，通常在任务间相关性较低时效果更好。

---

## 4. 训练过程与代码示例

```python
from torch_rechub.trainers import MTLTrainer

torch.manual_seed(2022)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    adaptive_params={"method": "uwl"},  # Uncertainty Weighting Loss
    n_epoch=20,
    earlystop_patience=5,
    device="cpu",
    model_path="./saved/ple"
)

mtl_trainer.fit(train_dl, val_dl)
```

### 多任务损失平衡方式

| 方法 | adaptive_params | 说明 |
|------|----------------|------|
| 等权重 | 不设置 | 简单加和各任务 Loss |
| UWL | `{"method": "uwl"}` | Uncertainty Weighting Loss |
| GradNorm | `{"method": "gradnorm"}` | 梯度归一化 |
| MetaBalance | `{"method": "metabalance"}` | MetaBalance 方法 |

---

## 5. 模型评估与结果分析

```python
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC (CTR): {auc[0]:.4f}, Test AUC (CVR): {auc[1]:.4f}")
```

---

## 6. 参数调优建议

1. **CGC 层数** (`n_level`): 1~2 层通常足够，更多层可能过拟合
2. **Expert 数量**: `n_expert_specific` 通常 2~4，`n_expert_shared` 通常 1~2
3. **损失平衡**: UWL 是推荐的起点，如果任务间梯度冲突严重，可以尝试 GradNorm
4. **Tower 结构**: 保持较浅（1~2 层），因为 Expert 已经完成了特征提取

---

## 7. 常见问题与解决方案

### Q1: PLE 和 MMOE 如何选择？
- 如果任务间**相关性强**，MMOE 通常足够
- 如果任务间**相关性弱**或存在**跷跷板效应**（seesaw），优先选择 PLE

### Q2: 如何处理分类 + 回归混合任务？
在 `task_types` 中分别指定：`["classification", "regression"]`，模型会自动应用不同的预测层（Sigmoid vs Identity）。

### Q3: n_level 设多大比较好？
通常 `n_level=2` 即可。更多层会显著增加参数量和训练时间。

---

## 8. 模型可视化

```python
from torch_rechub.utils.visualization import visualize_model
visualize_model(model, save_path="ple_architecture.png", dpi=300)
```

---

## 9. ONNX 导出

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("ple.onnx", verbose=True)
```

---

## 完整代码

```python
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import PLE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)

    # 1. 加载数据
    data = pd.read_csv("examples/ranking/data/ali-ccp/ali-ccp_sample.csv")

    col_names = data.columns.values.tolist()
    dense_features = ['D109_14', 'D110_14', 'D127_14', 'D150_14', 'D508', 'D509', 'D702', 'D853']
    sparse_features = [col for col in col_names if col not in dense_features and col not in ['ctr_label', 'cvr_label']]

    data[sparse_features] = data[sparse_features].fillna('-996')
    data[dense_features] = data[dense_features].fillna(0)

    # 2. 定义特征
    dense_feas = [DenseFeature(name) for name in dense_features]
    sparse_feas = [SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16) for name in sparse_features]
    features = dense_feas + sparse_feas

    y = data[["ctr_label", "cvr_label"]]
    del data["ctr_label"]
    del data["cvr_label"]
    x = data

    # 3. 创建 DataLoader
    dg = DataGenerator(x, y)
    train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=2048)

    # 4. 创建 PLE 模型
    model = PLE(
        features=features,
        task_types=["classification", "classification"],
        n_level=2, n_expert_specific=2, n_expert_shared=1,
        expert_params={"dims": [256, 128], "dropout": 0.2},
        tower_params_list=[{"dims": [64]}, {"dims": [64]}]
    )

    # 5. 训练
    mtl_trainer = MTLTrainer(
        model, task_types=["classification", "classification"],
        optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
        adaptive_params={"method": "uwl"},
        n_epoch=20, earlystop_patience=5, device="cpu", model_path="./saved/ple"
    )
    mtl_trainer.fit(train_dl, val_dl)

    # 6. 评估
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
    print(f"Test AUC (CTR): {auc[0]:.4f}, Test AUC (CVR): {auc[1]:.4f}")


if __name__ == "__main__":
    main()
```
