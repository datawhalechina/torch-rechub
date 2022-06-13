# Torch-RecHub

<p align="left">
  <img src='https://img.shields.io/badge/python-3.8+-brightgreen'>
  <img src='https://img.shields.io/badge/torch-1.7+-brightgreen'>
  <img src='https://img.shields.io/badge/scikit_learn-0.23.2+-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5+-brightgreen'>
  <img src='https://img.shields.io/badge/annoy-1.17.0-brightgreen'>
  <img src="https://img.shields.io/pypi/l/torch-rechub">
 <a href="https://github.com/datawhalechina/torch-rechub"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fdatawhalechina%2Ftorch-rechub&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>

## 中文Wiki站

查看最新研发进度，认领感兴趣的研发任务，学习rechub模型复现心得，加入rechub共建者团队等

[点击链接](https://www.wolai.com/rechub/2qjdg3DPy1179e1vpcHZQC)

## 安装

```python
#稳定版 
pip install torch-rechub

#最新版（推荐）
1. git clone https://github.com/datawhalechina/torch-rechub.git
2. cd torch-rechub
3. python setup.py install
```

## 核心定位

易用易拓展，聚焦复现业界实用的推荐模型，以及泛生态化的推荐场景

## 主要特性

*   scikit-learn风格易用的API（fit、predict），即插即用

*   模型训练与模型定义解耦，易拓展，可针对不同类型的模型设置不同的训练机制

*   接受pandas的DataFrame、Dict数据输入，上手成本低

*   高度模块化，支持常见Layer，容易调用组装成新模型

    *   LR、MLP、FM、FFM、CIN

    *   target-attention、self-attention、transformer

*   支持常见排序模型

    *   WideDeep、DeepFM、DIN、DCN、xDeepFM等

*   支持常见召回模型

    *   DSSM、YoutubeDNN、YoutubeDSSM、FacebookEBR、MIND等

*   丰富的多任务学习支持

    *   SharedBottom、ESMM、MMOE、PLE、AITM等模型

    *   GradNorm、UWL、MetaBanlance等动态loss加权机制

*   聚焦更生态化的推荐场景

    - [ ] 冷启动

    - [ ] 延迟反馈

    *   [ ] 去偏

*   支持丰富的训练机制

    *   [ ] 对比学习

    *   [ ] 蒸馏学习

*   [ ] 第三方高性能开源Trainer支持（Pytorch Lighting）

*   [ ] 更多模型正在开发中

## 快速使用

### 使用案例

- 所有模型使用案例参考 `/examples`

- 202206 Datawhale-RecHub推荐课程 组队学习期间notebook教程参考 `/tutorials`

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

dg = MatchDataGenerator(x y)
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

```

