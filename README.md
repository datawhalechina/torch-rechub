# torch-ctr

一个轻量级的基于Pytorch开发的推荐系统框架，易用易拓展。

## 主要特性

- scikit-learn风格易用的API（fit、predict），即插即用
- 训练过程与模型定义解耦，易拓展，可针对不同类型的模型设置不同的训练机制
- 使用Pytorch原生Dataset、DataLoader，易修改，自定义数据
- 高度模块化，支持常见Layer，容易调用组装成新模型
  - 浅层模块：LR、MLP
  - 交互模块：FM、FFM
  - Attention机制：target-attention、self-attention等
- 支持常见排序模型（WideDeep、DeepFM、DIN、DCN、xDeepFM等）

- [ ] 支持常见召回模型（DSSM、YoutubeDNN、Swing、SARSRec等）
- 丰富的多任务学习支持
  - SharedBottom、ESMM、MMOE、PLE、AITM等模型
  - GradNorm、UWL等动态loss加权机制

- 聚焦更生态化的推荐场景
  - [ ] 冷启动
  - [ ] 延迟反馈
  - [ ] 去偏
- [ ] 支持丰富的训练机制（对比学习、蒸馏学习等）

- [ ] 第三方高性能开源Trainer支持（Pytorch Lighting等）
- [ ] 更多模型正在开发中

## 快速使用

```python
from torch_ctr.models import WideDeep, DeepFM, DIN
from torch_ctr.trainers import CTRTrainer
from torch_ctr.basic.utils import DataGenerator

dg = DataGenerator(x, y)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader()

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"})

ctr_trainer = CTRTrainer(model)
ctr_trainer.fit(train_dataloader, val_dataloader)
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dataloader)


```





> **Note:** 
>
> 所有模型均在大多数论文提及的多个知名公开数据集中测试，达到或者接近论文性能。
>
> 使用案例：[Examples](./examples)
>
> 每个数据集将会提供
>
> - 一个使用脚本，包含样本生成、模型训练与测试，并提供一套测评所用参数。
> - 一个预处理脚本，将原始数据进行预处理，转化成csv。
> - 数据格式参考文件（100条）。
> - 全量数据，统一的csv文件，提供高速网盘下载链接和原始数据链接。

