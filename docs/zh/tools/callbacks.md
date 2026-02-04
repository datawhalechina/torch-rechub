---
title: 回调函数
description: Torch-RecHub 训练回调函数
---

# 回调函数

回调函数是在训练过程中执行特定操作的工具，用于实现早停、模型保存、学习率调整等功能。Torch-RecHub 提供了简洁易用的回调函数接口。

## EarlyStopper

EarlyStopper 是一个早停器，用于在验证集性能不再提升时停止训练，防止过拟合并节省训练时间。

### 功能描述

- 监控验证集指标（如 AUC）
- 当指标连续多轮没有提升时触发早停
- 自动保存最佳模型权重

### 使用方法

```python
from torch_rechub.basic.callback import EarlyStopper

# 创建早停器
early_stopper = EarlyStopper(patience=10)

# 在训练循环中使用
for epoch in range(n_epoch):
    # 训练一个 epoch
    train_one_epoch(model, train_dataloader)

    # 验证
    val_auc = evaluate(model, val_dataloader)

    # 检查是否需要早停
    if early_stopper.stop_training(val_auc, model.state_dict()):
        print(f'Early stopping at epoch {epoch}')
        print(f'Best validation AUC: {early_stopper.best_auc}')
        # 恢复最佳权重
        model.load_state_dict(early_stopper.best_weights)
        break
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| `patience` | int | 早停耐心值，即连续多少轮验证集性能没有提升就停止训练 | 必需 |

### 属性说明

| 属性 | 类型 | 描述 |
| --- | --- | --- |
| `best_auc` | float | 记录的最佳验证集 AUC |
| `best_weights` | dict | 最佳模型权重的深拷贝 |
| `trial_counter` | int | 当前连续未提升的轮数 |

### 方法说明

#### stop_training(val_auc, weights)

判断是否需要停止训练。

**参数：**
- `val_auc` (float): 当前验证集 AUC 分数
- `weights` (dict): 当前模型权重（`model.state_dict()`）

**返回值：**
- `bool`: 如果需要停止训练返回 `True`，否则返回 `False`

## 在 Trainer 中使用

Torch-RecHub 的训练器已经内置了早停功能，通过 `earlystop_patience` 参数控制：

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001},
    n_epoch=50,
    earlystop_patience=10,  # 早停耐心值
    device="cuda:0",
    model_path="saved/model"
)

trainer.fit(train_dataloader, val_dataloader)
```

## 完整示例

```python
import torch
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.callback import EarlyStopper

# 创建模型
model = DeepFM(
    deep_features=deep_features,
    fm_features=fm_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2}
)

# 方式一：使用 Trainer 内置的早停
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0"
)
trainer.fit(train_dl, val_dl)

# 方式二：手动使用 EarlyStopper
early_stopper = EarlyStopper(patience=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    for batch in train_dl:
        # 训练步骤
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # 验证
    model.eval()
    val_auc = evaluate(model, val_dl)
    print(f"Epoch {epoch}, Val AUC: {val_auc:.4f}")

    # 早停检查
    if early_stopper.stop_training(val_auc, model.state_dict()):
        print(f"Early stopping! Best AUC: {early_stopper.best_auc:.4f}")
        model.load_state_dict(early_stopper.best_weights)
        break
```

## 最佳实践

1. **选择合适的 patience 值**：
   - 太小可能导致过早停止，错过更好的结果
   - 太大可能浪费训练时间
   - 建议从 5-10 开始尝试

2. **结合学习率调度**：
   - 可以在早停前先尝试降低学习率
   - 使用 `scheduler_fn` 和 `scheduler_params` 配置学习率调度器

3. **保存检查点**：
   - 早停器会自动保存最佳权重
   - 建议同时使用 `model_path` 参数保存模型到磁盘

