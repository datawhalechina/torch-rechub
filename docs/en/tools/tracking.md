---
title: Experiment tracking
description: Record and compare training runs in Torch-RecHub
---

# Experiment tracking

To compare two DeepFM runs, record the loss and validation AUC after each epoch. TensorBoardX writes curves to local files. If your team uses W&B or SwanLab, pass its logger to the same `model_logger` argument. Leave the argument at its default to follow progress in the terminal.

## Save a local training curve

Install `tensorboardX` in the training environment and `tensorboard` to view the curves. Continue in the Python session from the DeepFM Quick Start so `model`, `train_dl` and `val_dl` are available. [DeepFM Quick Start](/guide/quick_start)

```python
from torch_rechub.basic.tracking import TensorBoardXLogger
from torch_rechub.trainers import CTRTrainer

logger = TensorBoardXLogger(log_dir="./runs/deepfm")
trainer = CTRTrainer(model, n_epoch=2, device="cpu", model_logger=logger)
trainer.fit(train_dl, val_dl)
```

After training, `./runs/deepfm` contains the event files. Start TensorBoard from the same working directory and open the address printed in the terminal to see `train/loss`, `learning_rate` and `val/auc`. For a second experiment, choose a directory such as `./runs/deepfm-lr2` so you can distinguish the curves.

```bash
tensorboard --logdir ./runs
```

## Track experiments with W&B or SwanLab

Install `wandb` for W&B or `swanlab` for SwanLab in the current Python environment. To install all three backends, run `pip install "torch-rechub[tracking]"`. Sign in with your account or set the platform's API key in the terminal where you start training.

```bash
# bash/zsh: set the variable in the current shell.
export WANDB_API_KEY="..."
export SWANLAB_API_KEY="..."
```

```powershell
# PowerShell: set the variable for the current session.
$env:WANDB_API_KEY = "..."
$env:SWANLAB_API_KEY = "..."
```

This example logs a run named `exp1` under the W&B project `my-ctr`. For SwanLab, use `SwanLabLogger(project="my-ctr", experiment_name="exp1")` instead.

```python
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger

logger = WandbLogger(project="my-ctr", name="exp1")  # Optional

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3},
    n_epoch=10,
    model_logger=logger,   # Use None to leave tracking off.
)

trainer.fit(train_dl, val_dl)
# The trainer closes its logger when fit() finishes.
```

### Use more than one logger

To keep local curves and share metrics with your team, pass a list of loggers. This example uses all three backends. You can use any subset.

```python
from torch_rechub.basic.tracking import TensorBoardXLogger, WandbLogger, SwanLabLogger

tb = TensorBoardXLogger(log_dir="./runs/exp1")
wb = WandbLogger(project="my-ctr", name="exp1")
sb = SwanLabLogger(project="my-ctr")

trainer = CTRTrainer(model, model_logger=[tb, wb, sb])
```

## Metrics by trainer

The trainer logs hyperparameters at the start and training metrics after each epoch. Validation metrics appear when you supply validation data. For multi-task runs, `i` starts at zero and follows the task order passed to the trainer.

| Trainer | Common metrics |
| --- | --- |
| `CTRTrainer`, `MatchTrainer` | `train/loss`, `learning_rate`, `val/auc` |
| `MTLTrainer` | `train/task_i_loss`, `val/task_i_score` and `learning_rate`, plus `loss_weight/task_i` when using adaptive task weights |
| `SeqTrainer` | `train/loss`, `learning_rate`, `val/loss`, `val/accuracy` |
| `RQVAETrainer` | `train/loss`, `train/recon_loss` and, at evaluation, `val/collision_rate` |

### Log additional metrics after fit()

The trainer closes its logger when `fit()` returns normally. To record a test AUC afterwards, start a separate run with a name that distinguishes it from the training run.

```python
logger = WandbLogger(project="demo", name="train-exp1")
trainer = CTRTrainer(model, model_logger=logger)
trainer.fit(train_dl, val_dl)  # The training logger is now closed.

# Create a new logger for the test results.
test_logger = WandbLogger(project="demo", name="test-exp1")
test_auc = trainer.evaluate(trainer.model, test_dl)
test_logger.log_metrics({"test/auc": test_auc})
test_logger.finish()
```

If you only need the final test AUC, manage the logger yourself. Log the result once after testing, then close the logger.

```python
logger = WandbLogger(project="demo")

# Keep this logger outside the trainer.
trainer = CTRTrainer(model, model_logger=None)
trainer.fit(train_dl, val_dl)

# Log the test result.
test_auc = trainer.evaluate(trainer.model, test_dl)
logger.log_metrics({"test/auc": test_auc})
logger.finish()  # Close the logger.
```
