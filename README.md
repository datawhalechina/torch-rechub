# torch-ctr

一个轻量级的基于Pytorch开发的推荐系统框架，易用易拓展，具有以下特性：

- sklearn风格易用的API（fit、predict）
- 训练过程与模型代码解耦，易拓展，易于针对不同类型的模型设置不同的训练机制
- 高度模块化，支持常见Layer，容易调用组装成新模型
  - 浅层模块：LR、MLP
  - 交互模块：FM、FFM
  - Attention机制：target-attention、self-attention
- 支持常见排序模型（WideDeep、DeepFM、DIN、DCN、xDeepFM等）

- [ ] 支持常见召回模型（DSSM、YoutubeDNN、Swing、SARSRec等）

- 丰富的多任务学习支持

  - SharedBottom、ESMM、MMOE、PLE、AITM等模型

  - [ ] GradNorm、UWL等动态loss加权机制

- 聚焦更生态化的推荐场景
  - [ ] 冷启动
  - [ ] 延迟反馈
  - [ ] 去偏
- [ ] 支持更丰富的训练机制（对比学习、蒸馏学习等）

- [ ] 第三方高性能开源Trainer支持（Pytorch Lighting等）



> **Note:** 
>
> 所有模型均在大多数论文提及的5个知名公开数据集中测试，达到或者接近论文性能。
>
> 使用案例：[Examples](./examples)
>
> 每个数据集将会提供
>
> - 一个脚本，包含预处理、模型训练与测试，模型参数提供一个未经深度调参的可用参数。
> - 数据格式参考文件（1000条）
> - 全量数据，处理成统一csv文件（提供高速网盘下载链接和原始数据链接）