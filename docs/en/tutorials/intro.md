---
title: Tutorial Overview
description: Torch-RecHub scenario tutorials overview
---

# Tutorial Overview

This section provides practical tutorials for Torch-RecHub in different recommendation scenarios to help developers get started quickly.

> **Code Resources**: The project provides interactive Jupyter Notebook tutorials (in the `tutorials/` directory) and complete Python example scripts (in the `examples/` directory) that can be used alongside this documentation.

## Tutorial List

| Tutorial | Description | Link |
| --- | --- | --- |
| **CTR Prediction** | Click-through rate prediction model training | [CTR Prediction Tutorial](/en/tutorials/ctr) |
| **Matching Models** | Two-tower matching model training | [Matching Models Tutorial](/en/tutorials/retrieval) |
| **Complete Pipeline** | End-to-end recommendation system | [Complete Pipeline Tutorial](/en/tutorials/pipeline) |

## Quick Navigation

### CTR Prediction (Ranking)

Learn how to use DeepFM, DCN, and other models for click-through rate prediction.

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

model = DeepFM(deep_features, fm_features, mlp_params)
trainer = CTRTrainer(model)
trainer.fit(train_dl, val_dl)
```

[View Full Tutorial →](/en/tutorials/ctr)

### Matching Models

Learn how to use DSSM, MIND, and other models for vector retrieval.

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

model = DSSM(user_features, item_features)
trainer = MatchTrainer(model)
trainer.fit(train_dl)
```

[View Full Tutorial →](/en/tutorials/retrieval)
