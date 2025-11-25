---
title: Trainers API Reference
description: API documentation for all trainers including CTRTrainer, MatchTrainer, and MTLTrainer
---

# Trainers API Reference

This section provides detailed API documentation for all trainers in Torch-RecHub.

## CTRTrainer

CTRTrainer is a general trainer for single task learning, primarily used for binary classification tasks such as Click-Through Rate (CTR) prediction.

### Parameters

- `model` (nn.Module): Any single task learning model
- `optimizer_fn` (torch.optim): PyTorch optimizer function, defaults to `torch.optim.Adam`
- `optimizer_params` (dict): Optimizer parameters, defaults to `{"lr": 1e-3, "weight_decay": 1e-5}`
- `scheduler_fn` (torch.optim.lr_scheduler): PyTorch learning rate scheduler, e.g., `torch.optim.lr_scheduler.StepLR`
- `scheduler_params` (dict): Learning rate scheduler parameters
- `n_epoch` (int): Number of training epochs
- `earlystop_patience` (int): Number of epochs to wait before early stopping when validation performance doesn't improve, defaults to 10
- `device` (str): Device to use, either `"cpu"` or `"cuda:0"`
- `gpus` (list): List of GPU IDs, defaults to empty. If length >=1, model will be wrapped by nn.DataParallel
- `loss_mode` (bool): Training mode, defaults to True
- `model_path` (str): Path to save the model, defaults to `"./"`

### Main Methods

- `train_one_epoch(data_loader, log_interval=10)`: Train for one epoch
- `fit(train_dataloader, val_dataloader=None)`: Train the model
- `evaluate(model, data_loader)`: Evaluate the model
- `predict(model, data_loader)`: Make predictions

## MatchTrainer

MatchTrainer is a trainer for matching/retrieval tasks, supporting multiple training modes.

### Parameters

- `model` (nn.Module): Any matching model
- `mode` (int): Training mode, options:
  - 0: point-wise
  - 1: pair-wise
  - 2: list-wise
- `optimizer_fn` (torch.optim): Same as CTRTrainer
- `optimizer_params` (dict): Same as CTRTrainer
- `scheduler_fn` (torch.optim.lr_scheduler): Same as CTRTrainer
- `scheduler_params` (dict): Same as CTRTrainer
- `n_epoch` (int): Same as CTRTrainer
- `earlystop_patience` (int): Same as CTRTrainer
- `device` (str): Same as CTRTrainer
- `gpus` (list): Same as CTRTrainer
- `model_path` (str): Same as CTRTrainer

### Main Methods

- `train_one_epoch(data_loader, log_interval=10)`: Train for one epoch
- `fit(train_dataloader, val_dataloader=None)`: Train the model
- `evaluate(model, data_loader)`: Evaluate the model
- `predict(model, data_loader)`: Make predictions
- `inference_embedding(model, mode, data_loader, model_path)`: Infer embeddings
  - `mode`: Either "user" or "item"

## MTLTrainer

MTLTrainer is a trainer for multi-task learning, supporting various adaptive loss weighting methods.

### Parameters

- `model` (nn.Module): Any multi-task learning model
- `task_types` (list): List of task types, supports ["classification", "regression"]
- `optimizer_fn` (torch.optim): Same as CTRTrainer
- `optimizer_params` (dict): Same as CTRTrainer
- `scheduler_fn` (torch.optim.lr_scheduler): Same as CTRTrainer
- `scheduler_params` (dict): Same as CTRTrainer
- `adaptive_params` (dict): Adaptive loss weighting method parameters, supports:
  - `{"method": "uwl"}`: Uncertainty Weighted Loss
  - `{"method": "metabalance"}`: MetaBalance method
  - `{"method": "gradnorm", "alpha": 0.16}`: GradNorm method
- `n_epoch` (int): Same as CTRTrainer
- `earlystop_taskid` (int): Task ID for early stopping, defaults to 0
- `earlystop_patience` (int): Same as CTRTrainer
- `device` (str): Same as CTRTrainer
- `gpus` (list): Same as CTRTrainer
- `model_path` (str): Same as CTRTrainer

### Main Methods

- `train_one_epoch(data_loader)`: Train for one epoch
- `fit(train_dataloader, val_dataloader, mode='base', seed=0)`: Train the model
- `evaluate(model, data_loader)`: Evaluate the model
- `predict(model, data_loader)`: Make predictions

### Special Features

1. Support for Multiple Adaptive Loss Weighting Methods:
   - UWL (Uncertainty Weighted Loss)
   - MetaBalance
   - GradNorm

2. Multi-task Early Stopping:
   - Early stopping based on specified task performance
   - Saves best model weights based on validation performance

3. Support for Multiple Task Type Combinations:
   - Classification tasks
   - Regression tasks

4. Training Log Recording:
   - Records loss for each task
   - Records loss weights (when using adaptive methods)
   - Records performance metrics on validation set

