---
title: PLE 使用示例
description: 用 PLE 的共享专家和任务专属专家预测点击和购买
---

# PLE 使用示例

## 1. 模型简介与适用场景

如果点击和购买需要的特征不完全一样，PLE 允许为两个任务分别留一些专家网络。本例中，点击和购买各有 2 个专属专家，还共用 1 个专家。每个任务的门控网络把能访问的专家输出组合起来，再交给自己的预测层。

**论文**: [Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations](https://dl.acm.org/doi/abs/10.1145/3383313.3412236)

### 模型结构

<div align="center">
  <img src="/img/models/ple_arch.png" alt="PLE Model Architecture" width="600"/>
</div>

图中的一层包含专属专家、共享专家和选择专家的门控网络。`n_level` 表示叠多少层这样的结构。下面从一层开始，标签和 MMOE 教程保持相同顺序，方便逐项对照。

---

## 2. 数据准备与预处理

使用 **Ali-CCP** (阿里妈妈点击和转化预测) 数据集，与 MMOE 的数据准备方式相同。

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
    batch_size=2048
)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.multi_task import PLE

model = PLE(
    features=features,
    task_types=["classification", "classification"],  # 两个分类任务
    n_level=1,                    # CGC 层数
    n_expert_specific=2,          # 每个任务自己的 experts
    n_expert_shared=1,            # 任务共享 experts；这是它和 MMOE 的关键差异
    expert_params={
        "dims": [16]
    },
    tower_params_list=[
        {"dims": [8]},            # CVR Tower
        {"dims": [8]}             # CTR Tower
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
| `expert_params` | `dict` | Expert MLP 参数 | `{"dims": [16]}` |
| `tower_params_list` | `list[dict]` | 每个 Task Tower 的 MLP 参数 | `{"dims": [8]}` |

`n_expert_specific=2` 是每个任务的专属专家数。这里有两个任务，所以这一层共有 4 个专属专家，再加上 1 个共享专家。

---

## 4. 训练过程与代码示例

```python
import os
from torch_rechub.trainers import MTLTrainer

torch.manual_seed(2022)
os.makedirs("./saved/ple", exist_ok=True)

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

省略 `adaptive_params` 时，训练器等权合并任务损失。上面的 `uwl` 会在训练中学习各任务的损失权重。查看结果时仍应分别看购买和点击的 AUC。

---

## 5. 模型评估与结果分析

```python
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC (CVR): {auc[0]:.4f}, Test AUC (CTR): {auc[1]:.4f}")
```

---

## 6. 参数调优建议

若要与 MMOE 比较，先保留相同的训练集、任务顺序和损失权重设置。在一层 PLE 上改变每个任务的专属专家数，分别记录点击和购买 AUC。这样可以看清增加专家对哪个任务有帮助。

## 7. 常见问题与解决方案

### Q1: PLE 和 MMOE 的区别在哪里？

MMOE 的专家都可以被两个任务使用。PLE 给每个任务配置专属专家，另外配置一组共享专家。两者都能给同一批样本同时做点击和购买预测。

### Q2: 如何处理分类 + 回归混合任务？
在 `task_types` 中分别指定：`["classification", "regression"]`，模型会为两类任务使用相应的预测层。若以第二个任务的 MSE 早停，在 `MTLTrainer` 中指定 `earlystop_taskid=1, earlystop_mode="min"`。

### Q3: n_level 设多大比较好？
示例使用一层。再加一层，上一层的输出会送入下一组专家和门控网络。比较两种设置时，在同一批验证样本上分别看两个任务的分数。

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
import os
import pandas as pd
import torch

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.multi_task import PLE
from torch_rechub.trainers import MTLTrainer
from torch_rechub.utils.data import DataGenerator


def main():
    torch.manual_seed(2022)
    os.makedirs("./saved/ple", exist_ok=True)

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
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=2048
    )

    # 4. 创建 PLE 模型
    model = PLE(
        features=features,
        task_types=["classification", "classification"],
        n_level=1, n_expert_specific=2, n_expert_shared=1,
        expert_params={"dims": [16]},
        tower_params_list=[{"dims": [8]}, {"dims": [8]}]
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
    print(f"Test AUC (CVR): {auc[0]:.4f}, Test AUC (CTR): {auc[1]:.4f}")


if __name__ == "__main__":
    main()
```
