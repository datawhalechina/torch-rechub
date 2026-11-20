---
title: 多任务学习教程
description: Torch-RecHub 多任务学习教程，覆盖 Ali-CCP 数据准备、MMOE、PLE 与 ESMM
---

# 多任务学习教程

Ali-CCP 的一条记录有两个结果，用户有没有点击，以及有没有购买。我们先用同一批样本训练 MMOE 和 PLE，对照两个任务各自的 AUC。最后换成 ESMM，看看点击和购买之间的关系如何体现在标签里。请从仓库根目录运行。

## 一、数据准备

### 1. 加载样本数据

```python
import pandas as pd

df_train = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_train_sample.csv")
df_val = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_val_sample.csv")
df_test = pd.read_csv("examples/ranking/data/ali-ccp/ali_ccp_test_sample.csv")

# 仓库样本已使用同一套类别编码，保留给定的三份切分。
# 换成原始业务数据时，词表只在训练集上拟合。
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

def make_features():
    return [SparseFeature(col, data[col].max() + 1, embed_dim=4) for col in sparse_cols] + [
        DenseFeature(col) for col in dense_cols
    ]
```

### 3. 按模型组织特征和标签

`label_cols` 决定了输出顺序。下面先放购买，再放点击，因此 `y_train` 的第 0 列是购买标签，第 1 列是点击标签。后面查看预测和 AUC 时也按这个顺序读取。

```python
from torch_rechub.utils.data import DataGenerator

def make_loaders(feature_cols, label_cols):
    def split(part):
        return {name: part[name].to_numpy() for name in feature_cols}, part[label_cols].to_numpy()

    x_train, y_train = split(data.iloc[:train_idx])
    x_val, y_val = split(data.iloc[train_idx:val_idx])
    x_test, y_test = split(data.iloc[val_idx:])
    return DataGenerator(x_train, y_train).generate_dataloader(
        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test, batch_size=256,
    )

label_cols = ["cvr_label", "ctr_label"]
train_dl, val_dl, test_dl = make_loaders(sparse_cols + dense_cols, label_cols)
```

## 二、MMOE

两个任务使用同一组专家网络。点击和购买各有一个门控网络，分别学习该给每个专家的输出多大权重。

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer

# MMOE：共享 expert + 任务专属 gate，是最常见的多任务基线
model = MMOE(
    features=make_features(),
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

PLE 为点击和购买各留了一组专属专家，同时保留共享专家。可以沿用上面的 DataLoader。这里再次调用 `make_features()`，让 PLE 从自己的嵌入参数开始训练。

```python
from torch_rechub.models.multi_task import PLE

# 每个模型使用独立的 Feature 实例，PLE 同时包含共享专家和任务专属专家
model = PLE(
    features=make_features(),
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

这个例子把类别特征分成用户侧和物品侧两组。除了购买、点击，还要计算第三列 `ctcvr_label`，表示同一条曝光是否既点击又购买。训练、验证和测试都按下面的顺序取这三列。

```python
from torch_rechub.models.multi_task import ESMM

item_cols = ["129", "205", "206", "207", "210", "216"]
user_cols = [col for col in sparse_cols if col not in item_cols]

user_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in user_cols]
item_features = [SparseFeature(col, data[col].max() + 1, embed_dim=16) for col in item_cols]

label_cols = ["cvr_label", "ctr_label", "ctcvr_label"]
esmm_train, esmm_val, esmm_test = make_loaders(sparse_cols, label_cols)

model = ESMM(
    user_features, item_features,
    cvr_params={"dims": [16, 8]}, ctr_params={"dims": [16, 8]},
)
os.makedirs("./saved/esmm", exist_ok=True)
esmm_trainer = MTLTrainer(
    model, task_types=["classification"] * 3, n_epoch=2,
    device="cpu", model_path="./saved/esmm",
)
esmm_trainer.fit(esmm_train, esmm_val)
scores = esmm_trainer.evaluate(esmm_trainer.model, esmm_test)
print(dict(zip(label_cols, scores)))
```

例如，一条“点击、未购买”记录对应三列 `[0, 1, 0]`。模型按 CVR、CTR、CTCVR 的顺序输出，训练损失取 CTR 和 CTCVR 两项，评估按同一列序返回三个分数。

## 五、怎样读评估结果？

对 MMOE 和 PLE，`evaluate()` 返回的第一个 AUC 对应购买，第二个对应点击。仓库的验证集和测试集各有 50 行，其中购买正例各只有一条。若这条记录的排名变化，购买 AUC 就可能明显波动。换成业务数据时，先查看每个任务有多少正例，再比较模型。

想单独调整专家网络的数量或层数，可从对应的模型教程接着看。

- [MMOE 教程](/zh/tutorials/models/multi_task/mmoe)
- [PLE 教程](/zh/tutorials/models/multi_task/ple)
