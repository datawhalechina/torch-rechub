---
title: Getting Started
description: Quick start guide for Torch-RecHub with examples for ranking, multi-task learning, and matching models
---

First, install Torch-RecHub:

```bash
pip install torch-rechub
```

Then use the following code to train recommender system models:

### Ranking (CTR Prediction)
```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

model = DeepFM(deep_features=deep_features, fm_features=fm_features, 
               mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dl, val_dl)
auc = ctr_trainer.evaluate(test_dl)
```

### Multi-Task Learning
```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

task_types = ["classification", "classification"]
model = MMOE(features, task_types, 8, 
            expert_params={"dims": [32,16]}, 
            tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

mtl_trainer = MTLTrainer(model)
mtl_trainer.fit(train_dl, val_dl)
```

### Matching Models
```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator(x, y)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

model = DSSM(user_features, item_features, temperature=0.02,
             user_params={"dims": [256, 128, 64], "activation": 'prelu'},
             item_params={"dims": [256, 128, 64], "activation": 'prelu'})

match_trainer = MatchTrainer(model)
match_trainer.fit(train_dl)
```

For more detailed examples and model implementations, please refer to the [API Reference](/manual/api-reference/basic) section.

