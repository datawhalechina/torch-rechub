---
title: 研发工具导览
description: Torch-RecHub 研发工具概述
---

# 研发工具导览

Torch-RecHub 提供了丰富的研发工具，帮助开发者更高效地进行模型开发、调试和部署。这些工具涵盖了训练过程监控、实验追踪、模型可视化等多个方面。

## 工具概览

| 工具类别 | 功能描述 | 文档链接 |
| --- | --- | --- |
| **回调函数** | 训练过程中的早停、模型保存等 | [回调函数](/zh/tools/callbacks) |
| **实验追踪** | WandB、SwanLab、TensorBoardX 集成 | [实验追踪](/zh/tools/tracking) |
| **模型可视化** | 模型架构图生成与展示 | [可视化监控](/zh/tools/visualization) |

## 回调函数

回调函数用于在训练过程中执行特定操作，如早停、模型保存等。

### EarlyStopper

早停器用于在验证集性能不再提升时停止训练，防止过拟合。

```python
from torch_rechub.basic.callback import EarlyStopper

# 创建早停器，连续10轮验证集性能没有提升就停止训练
early_stopper = EarlyStopper(patience=10)

# 在训练循环中使用
for epoch in range(n_epoch):
    # ... 训练代码 ...
    val_auc = evaluate(model, val_dataloader)

    if early_stopper.stop_training(val_auc, model.state_dict()):
        print(f'Early stopping! Best AUC: {early_stopper.best_auc}')
        model.load_state_dict(early_stopper.best_weights)
        break
```

详情请参考 [回调函数](/zh/tools/callbacks) 页面。

## 实验追踪

Torch-RecHub 内置了可选的实验跟踪能力，支持三种主流工具：

- **Weights & Biases (wandb)**：云端实验管理平台
- **SwanLab**：国产实验追踪工具
- **TensorBoardX**：本地可视化工具

```python
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger

# 创建 logger
logger = WandbLogger(project="my-ctr", name="exp1")

# 传入 Trainer
trainer = CTRTrainer(model, model_logger=logger)
trainer.fit(train_dl, val_dl)
```

详情请参考 [实验追踪](/zh/tools/tracking) 页面。

## 模型可视化

Torch-RecHub 提供了模型架构可视化功能，可以生成模型的计算图。

```python
from torch_rechub.utils.visualization import visualize_model

# 可视化模型架构
graph = visualize_model(model, depth=4)

# 保存为 PDF
visualize_model(model, save_path="model_arch.pdf", dpi=300)
```

详情请参考 [可视化监控](/zh/tools/visualization) 页面。

## 损失函数

Torch-RecHub 提供了多种推荐系统常用的损失函数：

### RegularizationLoss

统一的 L1/L2 正则化损失，支持对 Embedding 层和全连接层分别设置正则化系数。

```python
from torch_rechub.basic.loss_func import RegularizationLoss

reg_loss_fn = RegularizationLoss(
    embedding_l1=0.0,
    embedding_l2=1e-5,
    dense_l1=0.0,
    dense_l2=1e-5
)

# 计算正则化损失
reg_loss = reg_loss_fn(model)
total_loss = task_loss + reg_loss
```

### BPRLoss

用于召回模型的 pairwise 损失函数。

```python
from torch_rechub.basic.loss_func import BPRLoss

bpr_loss = BPRLoss()
loss = bpr_loss(pos_score, neg_score)
```

### HingeLoss

用于 pairwise 学习的 Hinge 损失。

```python
from torch_rechub.basic.loss_func import HingeLoss

hinge_loss = HingeLoss(margin=2)
loss = hinge_loss(pos_score, neg_score)
```

### NCELoss

噪声对比估计损失，用于生成式推荐模型。

```python
from torch_rechub.basic.loss_func import NCELoss

nce_loss = NCELoss(temperature=0.1)
loss = nce_loss(logits, targets)
```

## 下一步

- 了解 [回调函数](/zh/tools/callbacks) 的详细用法
- 了解 [实验追踪](/zh/tools/tracking) 的配置方法
- 了解 [可视化监控](/zh/tools/visualization) 的使用方式

