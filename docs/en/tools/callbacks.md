---
title: Callbacks
description: Use validation metrics to stop training and keep the best weights
---

# Callbacks

## Early stopping in a trainer

Suppose the first three validation AUCs are 0.70, 0.75 and 0.73. At the end of epoch two, `CTRTrainer` writes that epoch's weights to `model_path/model.pth`. The third score is lower, so the file still contains the second epoch's weights. A tie also leaves the saved weights in place.

The example runs for at most 20 epochs and stops after three consecutive epochs without improvement. When `fit()` returns normally, `trainer.model` holds the weights from the highest validation AUC. If you interrupt training, you can load the weights already saved to disk.

```python
from pathlib import Path
from torch_rechub.trainers import CTRTrainer

model_dir = Path("saved/deepfm")
model_dir.mkdir(parents=True, exist_ok=True)
trainer = CTRTrainer(model, n_epoch=20, earlystop_patience=3, model_path=str(model_dir))
trainer.fit(train_dl, val_dl)
```

For multi-task training, choose the task to monitor with `earlystop_taskid`. AUC is maximized by default. To monitor a regression task's MSE, pass `earlystop_mode="min"`. [Trainer inputs and evaluation results](/core/evaluation)

## Use a custom training loop

After each validation pass, give `stop_training` the metric and current weights. The example monitors validation loss with `mode="min"`, so lower is better. The stopper keeps the best score in `best_auc` and the corresponding weights in `best_weights`. Both live in memory. Save them to disk in your loop if you need a checkpoint.

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
