---
title: Experiment Tracking
description: Torch-RecHub experiment logging and comparison
---

# Experiment Tracking (Lite)

Torch-RecHub offers optional experiment tracking with:
- **Weights & Biases (wandb)**
- **SwanLab**
- **TensorBoardX**

Goal: zero overhead by default. If you want tracking, pass `model_logger` to a Trainer.

## Installation

```bash
# all tracking deps
pip install torch-rechub[tracking]

# or individually
pip install wandb
pip install swanlab
pip install tensorboardX
```

## Quickstart

```python
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger

logger = WandbLogger(project="my-ctr", name="exp1")  # optional

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3},
    n_epoch=10,
    model_logger=logger,   # pass None for zero tracking
)

trainer.fit(train_dl, val_dl)
```

### Multiple loggers

`model_logger` can be a single instance or a list:

```python
tb = TensorBoardXLogger(log_dir="./runs/exp1")
wb = WandbLogger(project="my-ctr", name="exp1")
sb = SwanLabLogger(project="my-ctr")

trainer = CTRTrainer(model, model_logger=[tb, wb])
```

### Set API keys in code

```python
import os
os.environ['WANDB_API_KEY'] = "your API_KEY"
os.environ['SWANLAB_API_KEY'] = "your API_KEY"
```

### Overhead when disabled

When `model_logger=None`, training runs without any tracking calls—no extra cost.

## API Overview

```python
class BaseLogger:
    log_metrics(metrics: Dict[str, Any], step: Optional[int] = None)
    log_hyperparams(params: Dict[str, Any])
    finish()
```

- **WandbLogger(project, name=None, config=None, tags=None, notes=None, dir=None, …)**
- **SwanLabLogger(project=None, experiment_name=None, description=None, config=None, logdir=None, …)**
- **TensorBoardXLogger(log_dir, comment="", …)**

## Trainer behavior (auto-logging)

- `train/loss`, `learning_rate`
- Validation metrics: `val/auc` (CTR/Match), `val/task_i_score` (MTL), `val/accuracy` (Seq)
- Metrics are logged only when `model_logger` is provided

## FAQ

1) **How to skip tracking?**  
   Do not pass `model_logger` (default None).

2) **Disable hyperparameter logging?**  
   Trainers log hyperparameters once at start; you can override/ignore as needed.

3) **A library is missing?**  
   Instantiating a logger raises a friendly ImportError; training still works.

4) **Custom metrics?**  
   Keep the logger instance and call `log_metrics` yourself.

```python
logger = WandbLogger(project="demo")
logger.log_metrics({"custom_metric": 0.9}, step=epoch)
```

## Best Practices
- Local quick debug: TensorBoardX
- Team/remote: WandB or SwanLab
- Need local + cloud: pass multiple loggers
- No logger specified: minimal overhead

