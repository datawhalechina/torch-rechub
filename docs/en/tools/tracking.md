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
# Logger is automatically closed after fit(), no need to call finish() manually
```

### Multiple loggers

`model_logger` can be a single instance or a list:

```python
from torch_rechub.basic.tracking import TensorBoardXLogger, WandbLogger, SwanLabLogger

tb = TensorBoardXLogger(log_dir="./runs/exp1")
wb = WandbLogger(project="my-ctr", name="exp1")
sb = SwanLabLogger(project="my-ctr")

trainer = CTRTrainer(model, model_logger=[tb, wb, sb])
```

### Set API keys in code

```python
import os
os.environ['WANDB_API_KEY'] = "your API_KEY"
os.environ['SWANLAB_API_KEY'] = "your API_KEY"
```

### Overhead when disabled

When `model_logger=None`, training runs without any tracking calls—no extra cost.

## Supported Trainers

| Trainer        | Supported | Auto-logged Metrics                                                            |
| -------------- | --------- | ------------------------------------------------------------------------------ |
| `CTRTrainer`   | ✅         | `train/loss`, `learning_rate`, `val/auc`                                       |
| `MatchTrainer` | ✅         | `train/loss`, `learning_rate`, `val/auc`                                       |
| `MTLTrainer`   | ✅         | `train/task_i_loss`, `learning_rate`, `val/task_i_score`, `loss_weight/task_i` |
| `SeqTrainer`   | ✅         | `train/loss`, `learning_rate`, `val/loss`, `val/accuracy`                      |

## Lifecycle Management

### Automatic Management (Recommended)

Trainer automatically manages the logger lifecycle:

```python
logger = WandbLogger(project="demo")
trainer = CTRTrainer(model, model_logger=logger)

trainer.fit(train_dl, val_dl)
# ✅ fit() internally calls:
#    1. logger.log_hyperparams() - at training start
#    2. logger.log_metrics()     - at end of each epoch
#    3. logger.finish()          - at training end
```

### Logging Additional Metrics After fit()

Since `fit()` automatically calls `finish()` to close the logger, if you need to log test metrics after training, use one of these approaches:

**Approach A: Manual management (don't pass to Trainer)**

```python
logger = WandbLogger(project="demo")

# Don't pass to Trainer
trainer = CTRTrainer(model, model_logger=None)
trainer.fit(train_dl, val_dl)

# Log manually
test_auc = trainer.evaluate(trainer.model, test_dl)
logger.log_metrics({"test/auc": test_auc})
logger.finish()  # Close manually
```

**Approach B: Create a new logger for test metrics**

```python
logger = WandbLogger(project="demo", name="train-exp1")
trainer = CTRTrainer(model, model_logger=logger)
trainer.fit(train_dl, val_dl)  # logger is closed

# Create new logger for test results
test_logger = WandbLogger(project="demo", name="test-exp1")
test_auc = trainer.evaluate(trainer.model, test_dl)
test_logger.log_metrics({"test/auc": test_auc})
test_logger.finish()
```

## API Overview

### BaseLogger Interface

```python
class BaseLogger:
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metric values"""
        pass

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters"""
        pass

    def finish(self) -> None:
        """Close logger and release resources"""
        pass
```

### Logger Implementations

**WandbLogger**

```python
WandbLogger(
    project: str,              # Project name (required)
    name: str = None,          # Run name
    config: dict = None,       # Initial config/hyperparameters
    tags: List[str] = None,    # Tags
    notes: str = None,         # Notes
    dir: str = None,           # Local cache directory
    **kwargs                   # Additional args for wandb.init()
)
```

**SwanLabLogger**

```python
SwanLabLogger(
    project: str = None,           # Project name
    experiment_name: str = None,   # Experiment name
    description: str = None,       # Description
    config: dict = None,           # Initial config
    logdir: str = None,            # Log directory
    **kwargs                       # Additional args for swanlab.init()
)
```

**TensorBoardXLogger**

```python
TensorBoardXLogger(
    log_dir: str,          # Log directory (required)
    comment: str = "",     # Suffix for directory name
    **kwargs               # Additional args for SummaryWriter
)
```

## Full Example

```python
import os
import torch
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger, SwanLabLogger, TensorBoardXLogger

# Set API keys (optional, can also login via CLI)
os.environ['WANDB_API_KEY'] = "your_key"
os.environ['SWANLAB_API_KEY'] = "your_key"

# Config
SEED = 2022
EPOCH = 10
LR = 1e-3
BATCH_SIZE = 2048

# Choose loggers (can use multiple)
loggers = []

# Local TensorBoard
loggers.append(TensorBoardXLogger(log_dir=f"./runs/deepfm-{SEED}"))

# Cloud WandB
loggers.append(WandbLogger(
    project="ctr-experiment",
    name=f"deepfm-{SEED}",
    config={"seed": SEED, "lr": LR, "batch_size": BATCH_SIZE},
    tags=["deepfm", "criteo"]
))

# Prepare model and data (omitted)
model = DeepFM(...)
train_dl, val_dl, test_dl = ...

# Train
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": LR, "weight_decay": 1e-5},
    n_epoch=EPOCH,
    earlystop_patience=3,
    device="cuda:0",
    model_logger=loggers,  # Pass logger list
)

trainer.fit(train_dl, val_dl)
# Training complete, all loggers automatically closed

# Evaluate test set
test_auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {test_auc}")
```

## FAQ

**1. How to skip tracking?**

Don't pass `model_logger` or pass `None` (default value).

**2. What if a library is not installed?**

Creating the corresponding logger raises a friendly `ImportError`; other features still work.

**3. Why does `log_metrics()` fail after `fit()`?**

Because `fit()` automatically calls `finish()` to close the logger. To log metrics after training, see the "Logging Additional Metrics After fit()" section above.

**4. Can I disable automatic hyperparameter logging?**

Not currently supported. Trainer automatically logs hyperparameters once at training start. For full customization, don't pass `model_logger` and manage manually.

**5. How to view TensorBoard logs?**

```bash
tensorboard --logdir ./runs
```

## Best Practices

| Scenario                         | Recommendation                              |
| -------------------------------- | ------------------------------------------- |
| Local quick debug                | `TensorBoardXLogger`                        |
| Team collaboration/cloud logging | `WandbLogger` or `SwanLabLogger`            |
| Both local + cloud               | Pass a list of multiple loggers             |
| No tracking needed               | Don't pass `model_logger` (zero overhead)   |
| Need to log test metrics         | Manage logger manually or create new logger |
