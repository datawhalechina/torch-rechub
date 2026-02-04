---
title: Tooling Overview
description: Torch-RecHub tooling overview
---

# Tooling Overview

Torch-RecHub provides a rich set of development tools to help developers more efficiently develop, debug, and deploy models. These tools cover training process monitoring, experiment tracking, model visualization, and more.

## Tool Overview

| Tool Category | Description | Documentation |
| --- | --- | --- |
| **Callbacks** | Early stopping, model saving during training | [Callbacks](/en/tools/callbacks) |
| **Experiment Tracking** | WandB, SwanLab, TensorBoardX integration | [Experiment Tracking](/en/tools/tracking) |
| **Model Visualization** | Model architecture graph generation and display | [Visualization](/en/tools/visualization) |

## Callbacks

Callbacks are used to perform specific operations during training, such as early stopping and model saving.

### EarlyStopper

Early stopper stops training when validation performance stops improving, preventing overfitting.

```python
from torch_rechub.basic.callback import EarlyStopper

# Create early stopper, stop training if no improvement for 10 consecutive epochs
early_stopper = EarlyStopper(patience=10)

# Use in training loop
for epoch in range(n_epoch):
    # ... training code ...
    val_auc = evaluate(model, val_dataloader)

    if early_stopper.stop_training(val_auc, model.state_dict()):
        print(f'Early stopping! Best AUC: {early_stopper.best_auc}')
        model.load_state_dict(early_stopper.best_weights)
        break
```

See [Callbacks](/en/tools/callbacks) for details.

## Experiment Tracking

Torch-RecHub has built-in optional experiment tracking capabilities, supporting three mainstream tools:

- **Weights & Biases (wandb)**: Cloud experiment management platform
- **SwanLab**: Experiment tracking tool
- **TensorBoardX**: Local visualization tool

```python
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger

# Create logger
logger = WandbLogger(project="my-ctr", name="exp1")

# Pass to Trainer
trainer = CTRTrainer(model, model_logger=logger)
trainer.fit(train_dl, val_dl)
```

See [Experiment Tracking](/en/tools/tracking) for details.

## Model Visualization

Torch-RecHub provides model architecture visualization to generate computation graphs.

```python
from torch_rechub.utils.visualization import visualize_model

# Visualize model architecture
graph = visualize_model(model, depth=4)

# Save as PDF
visualize_model(model, save_path="model_arch.pdf", dpi=300)
```

See [Visualization](/en/tools/visualization) for details.

## Loss Functions

Torch-RecHub provides various loss functions commonly used in recommendation systems:

### RegularizationLoss

Unified L1/L2 regularization loss, supporting separate regularization coefficients for Embedding layers and fully connected layers.

```python
from torch_rechub.basic.loss_func import RegularizationLoss

reg_loss_fn = RegularizationLoss(
    embedding_l1=0.0,
    embedding_l2=1e-5,
    dense_l1=0.0,
    dense_l2=1e-5
)

# Calculate regularization loss
reg_loss = reg_loss_fn(model)
total_loss = task_loss + reg_loss
```

### BPRLoss

Pairwise loss function for retrieval models.

```python
from torch_rechub.basic.loss_func import BPRLoss

bpr_loss = BPRLoss()
loss = bpr_loss(pos_score, neg_score)
```

### HingeLoss

Hinge loss for pairwise learning.

```python
from torch_rechub.basic.loss_func import HingeLoss

hinge_loss = HingeLoss(margin=2)
loss = hinge_loss(pos_score, neg_score)
```

### NCELoss

Noise Contrastive Estimation loss for generative recommendation models.

```python
from torch_rechub.basic.loss_func import NCELoss

nce_loss = NCELoss(temperature=0.1)
loss = nce_loss(logits, targets)
```

## Next Steps

- Learn about [Callbacks](/en/tools/callbacks) in detail
- Learn about [Experiment Tracking](/en/tools/tracking) configuration
- Learn about [Visualization](/en/tools/visualization) usage

