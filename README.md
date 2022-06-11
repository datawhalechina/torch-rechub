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

#最新版
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

### 单任务排序

```python
from torch_rechub.models.ranking import WideDeep, DeepFM, DIN
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.utils import DataGenerator

dg = DataGenerator(x, y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader()

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)


```

### 多任务排序

```python
from torch_rechub.models.multi_task import SharedBottom, ESMM, MMOE, PLE, AITM
from torch_rechub.trainers import MTLTrainer

model = MMOE(features, task_types, n_expert=3, expert_params={"dims": [64,32,16]}, tower_params_list=[{"dims": [8]}, {"dims": [8]}])

ctr_trainer = MTLTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)
```
