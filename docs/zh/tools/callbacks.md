---
title: 回调函数
description: Torch-RecHub 训练回调函数
---

# 回调函数

## 训练器中的早停

假设前三轮的验证 AUC 分别为 0.70、0.75、0.73。第二轮结束时，`CTRTrainer` 会把权重写到 `model_path/model.pth`。第三轮比第二轮低，文件里仍是第二轮的权重。分数持平也沿用上次保存的结果。

下面最多训练 20 轮，连续 3 轮没有提高就提前结束。`fit()` 正常返回后，`trainer.model` 已经载入验证 AUC 最高的那一轮。如果手动中断训练，可以从文件读取此前已经保存的权重。

```python
from pathlib import Path
from torch_rechub.trainers import CTRTrainer

model_dir = Path("saved/deepfm")
model_dir.mkdir(parents=True, exist_ok=True)
trainer = CTRTrainer(model, n_epoch=20, earlystop_patience=3, model_path=str(model_dir))
trainer.fit(train_dl, val_dl)
```

多任务训练用 `earlystop_taskid` 指定要监控的任务。AUC 取最大值（默认），选中回归任务的 MSE 时传 `earlystop_mode="min"`。各训练器的输入与评估输出见[训练与评估](/zh/core/evaluation)。

## 自定义训练循环

自己写训练循环时，每轮验证后把指标和当前权重传给 `stop_training`。下面监控的是验证损失，`mode="min"` 表示越低越好。最佳分数保存在 `best_auc`，对应的权重在 `best_weights`。这两个值都在内存中，需要留文件时由训练循环写入磁盘。

```python
from torch_rechub.basic.callback import EarlyStopper

stopper = EarlyStopper(patience=3, mode="min")
for epoch in range(n_epoch):
    train_one_epoch(model, train_dl)
    val_loss = validation_loss(model, val_dl)
    if stopper.stop_training(val_loss, model.state_dict()):
        break
model.load_state_dict(stopper.best_weights)
```

