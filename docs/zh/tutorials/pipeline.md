---
title: 多任务学习教程
description: Torch-RecHub 多任务学习教程，覆盖 Ali-CCP 数据准备、MMOE、PLE 与 ESMM
---

# 多任务学习教程

本教程使用仓库内置的 `Ali-CCP` 样本数据，介绍当前 Torch-RecHub 中多任务学习模型的真实训练方式。文中的代码默认在**仓库根目录**执行。

## 一、数据准备

### 1. 加载样本数据

```python
import pandas as pd

df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

# 先把 train / val / test 拼起来统一做特征定义，后面再按索引切回去
train_idx = df_train.shape[0]
val_idx = train_idx + df_val.shape[0]

data = pd.concat([df_train, df_val, df_test], axis=0)
# ctcvr_label 是 ESMM 常用的第三个任务标签：click * conversion
data.rename(columns={"purchase": "cvr_label", "click": "ctr_label"}, inplace=True)
data["ctcvr_label"] = data["cvr_label"] * data["ctr_label"]
```

### 2. 构建 Dense / Sparse 特征

```python
from torch_rechub.basic.features import DenseFeature, SparseFeature

# Ali-CCP 里稀疏特征占多数，少量列按 dense feature 处理
dense_cols = ["D109_14", "D110_14", "D127_14", "D150_14", "D508", "D509", "D702", "D853"]
sparse_cols = [
    col for col in data.columns
    if col not in dense_cols and col not in ["cvr_label", "ctr_label", "ctcvr_label"]
]

# 多任务场景里，所有任务默认共享同一套底层输入特征
features = [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] + [
    DenseFeature(col) for col in dense_cols
]

label_cols = ["cvr_label", "ctr_label"]
used_cols = sparse_cols + dense_cols
```

### 3. 构建训练 / 验证 / 测试集

```python
from torch_rechub.utils.data import DataGenerator

# DataGenerator 在多任务场景下仍然复用同一套接口，只是 y 变成二维标签
x_train = {name: data[name].values[:train_idx] for name in used_cols}
y_train = data[label_cols].values[:train_idx]

x_val = {name: data[name].values[train_idx:val_idx] for name in used_cols}
y_val = data[label_cols].values[train_idx:val_idx]

x_test = {name: data[name].values[val_idx:] for name in used_cols}
y_test = data[label_cols].values[val_idx:]

dg = DataGenerator(x_train, y_train)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=x_val,
    y_val=y_val,
    x_test=x_test,
    y_test=y_test,
    batch_size=1024,
)
```

## 二、MMOE

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer

# MMOE：共享 expert + 任务专属 gate，是最常见的多任务基线
model = MMOE(
    features=features,
    task_types=["classification", "classification"],
    n_expert=8,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}],
)
```

### 训练方式

```python
import os
import torch

torch.manual_seed(2022)
# MTLTrainer 不会自动创建 model_path，这里先手动创建目录
os.makedirs("./saved/mmoe", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-4},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/mmoe",
)

mtl_trainer.fit(train_dl, val_dl)
# evaluate 返回的是一个列表，顺序与 task_types 一致
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(f"Test AUC: {auc}")  # [cvr_auc, ctr_auc]
```

## 三、PLE

```python
from torch_rechub.models.multi_task import PLE

# PLE 在任务差异更大时通常比 MMOE 更稳，因为它区分 shared / task-specific experts
model = PLE(
    features=features,
    task_types=["classification", "classification"],
    n_level=1,
    n_expert_specific=2,
    n_expert_shared=1,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}],
)
```

### 自适应损失权重（可选）

```python
# adaptive_params 用于打开动态 loss 平衡；这里示例用的是 UWL
os.makedirs("./saved/ple", exist_ok=True)

mtl_trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    adaptive_params={"method": "uwl"},
    n_epoch=5,
    earlystop_patience=5,
    device="cpu",
    model_path="./saved/ple",
)

mtl_trainer.fit(train_dl, val_dl)
```

## 四、ESMM

`ESMM` 的输入与 `MMOE / PLE` 不同：它只使用稀疏特征，并且标签顺序通常为 `["cvr_label", "ctr_label", "ctcvr_label"]`。

```python
from torch_rechub.models.multi_task import ESMM

item_cols = ["129", "205", "206", "207", "210", "216"]
user_cols = [col for col in sparse_cols if col not in item_cols]

user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]

label_cols = ["cvr_label", "ctr_label", "ctcvr_label"]
x_train = {name: data[name].values[:train_idx] for name in sparse_cols}
y_train = data[label_cols].values[:train_idx]
```

```python
# ESMM 会把 user tower 和 item tower 的组合用于估计 CTR / CVR / CTCVR
model = ESMM(
    user_features,
    item_features,
    cvr_params={"dims": [16, 8]},
    ctr_params={"dims": [16, 8]},
)
```

## 五、训练器接口

```python
from torch_rechub.trainers import MTLTrainer

trainer = MTLTrainer(
    model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 1e-3},
    regularization_params={"embedding_l2": 0.0, "dense_l2": 0.0},
    adaptive_params=None,   # 可选: {"method": "uwl"} / {"method": "gradnorm"} / {"method": "metabalance"}
    n_epoch=10,
    earlystop_taskid=0,
    earlystop_patience=10,
    device="cpu",
    model_path="./saved/mtl",
)
```

## 六、评估与调优建议

### 1. 评估输出

```python
scores = mtl_trainer.evaluate(mtl_trainer.model, test_dl)
print(scores)
```

`evaluate()` 会返回一个列表，顺序与 `task_types` 一致。例如：

- `[cvr_auc, ctr_auc]`
- 或 ESMM 下的三个任务分数

### 2. 调优重点

- `MMOE`：优先调 `n_expert`
- `PLE`：优先调 `n_level / n_expert_specific / n_expert_shared`
- 任务失衡明显时：尝试 `adaptive_params={"method": "uwl"}`
- 多任务 AUC 波动较大时：先减小学习率，再缩短专家网络维度

## 七、常见问题

### Q1：为什么这页不再使用 `from torch_rechub.utils import DataGenerator`？

因为 `DataGenerator` 位于 `torch_rechub.utils.data`，不是从 `torch_rechub.utils` 顶层导出。

### Q2：为什么文档不再使用 `n_epochs`？

`MTLTrainer` 的参数名是 `n_epoch`。

### Q3：为什么这里没有 `evaluate_multi_task()`？

框架直接使用 `MTLTrainer.evaluate(model, data_loader)`，返回每个任务的分数列表，不存在 `evaluate_multi_task()` 这个公开函数。

### Q4：为什么训练前要先执行 `os.makedirs`？

`MTLTrainer` 不会自动创建 `model_path` 目录，因此文档示例中显式创建保存目录。

### Q5：还想继续看具体模型页？

- [MMOE 教程](/zh/tutorials/models/multi_task/mmoe)
- [PLE 教程](/zh/tutorials/models/multi_task/ple)
