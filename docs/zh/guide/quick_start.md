---
title: 快速开始
description: Torch-RecHub 快速入门指南，包含排序、多任务学习和召回模型示例
---

首先，安装Torch-RecHub：

```bash
pip install torch-rechub
```

然后，使用以下代码进行推荐系统模型的训练：

### 精排（CTR预测）

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

dg = DataGenerator(x, y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```

### 多任务排序

```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

task_types = ["classification", "classification"] 
model = MMOE(features, task_types, 8, expert_params={"dims": [32,16]}, tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

mtl_trainer = MTLTrainer(model)
mtl_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```

### 召回模型

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator

dg = MatchDataGenerator(x, y)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

model = DSSM(user_features, item_features, temperature=0.02,
             user_params={
                 "dims": [256, 128, 64],
                 "activation": 'prelu',  
             },
             item_params={
                 "dims": [256, 128, 64],
                 "activation": 'prelu', 
             })

match_trainer = MatchTrainer(model)
match_trainer.fit(train_dl)
match_trainer.export_onnx("dssm.onnx")  # 添加：ONNX导出示例
# 双塔模型可分别导出用户塔和物品塔:
# match_trainer.export_onnx("user_tower.onnx", mode="user")
# match_trainer.export_onnx("item_tower.onnx", mode="item")
```
