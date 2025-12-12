---
title: Data Pipeline
description: Torch-RecHub data loading and preprocessing
---

# Data Pipeline

Torch-RecHub offers datasets, generators, and utilities for recommendation data.

## Datasets

### TorchDataset
Training/validation dataset with features and labels.

```python
from torch_rechub.utils.data import TorchDataset
dataset = TorchDataset(x, y)
```

### PredictDataset
Prediction-only dataset (features only).

```python
from torch_rechub.utils.data import PredictDataset
dataset = PredictDataset(x)
```

## Data Generators

### DataGenerator
Build dataloaders for ranking / multi-task models.

```python
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],
    batch_size=256,
    num_workers=8,
)
```

### MatchDataGenerator
Build dataloaders for matching/retrieval models.

```python
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator(x, y)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    x_test_user=x_test_user,
    x_all_item=x_all_item,
    batch_size=256,
    num_workers=8,
)
```

## Utilities

### get_auto_embedding_dim
Compute embedding dim from vocab size: ``int(floor(6 * num_classes**0.25))``.

```python
from torch_rechub.utils.data import get_auto_embedding_dim
embed_dim = get_auto_embedding_dim(vocab_size=1000)
```

### get_loss_func
Return default loss by task type: BCELoss for classification, MSELoss for regression.

```python
from torch_rechub.utils.data import get_loss_func
loss_fn = get_loss_func(task_type="classification")
```

## Typical Flow

1. Define features (Dense/Sparse/Sequence).  
2. Load raw data.  
3. Encode categorical features (e.g., LabelEncoder).  
4. Process sequences (pad/truncate).  
5. Construct samples (e.g., negative sampling).  
6. Use DataGenerator / MatchDataGenerator to build dataloaders.  
7. Train models with the trainers.

