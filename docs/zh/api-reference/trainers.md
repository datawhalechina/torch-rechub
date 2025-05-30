# 训练器 API 参考

这里详细介绍 Torch-RecHub 中各个训练器的 API 接口和参数说明。

## CTRTrainer

CTRTrainer 是一个用于单任务学习的通用训练器，主要用于点击率预测（CTR）等二分类任务。

### 参数说明

- `model` (nn.Module): 任何单任务学习模型
- `optimizer_fn` (torch.optim): PyTorch优化器函数，默认为 `torch.optim.Adam`
- `optimizer_params` (dict): 优化器参数，默认为 `{"lr": 1e-3, "weight_decay": 1e-5}`
- `scheduler_fn` (torch.optim.lr_scheduler): PyTorch学习率调度器，例如 `torch.optim.lr_scheduler.StepLR`
- `scheduler_params` (dict): 学习率调度器参数
- `n_epoch` (int): 训练轮数
- `earlystop_patience` (int): 早停耐心值，即在验证集性能多少轮未提升后停止训练，默认为10
- `device` (str): 使用的设备，可选 `"cpu"` 或 `"cuda:0"`
- `gpus` (list): 多GPU ID列表，默认为空。如果长度>=1，模型将被 nn.DataParallel 包装
- `loss_mode` (bool): 训练模式，默认为True
- `model_path` (str): 模型保存路径，默认为 `"./"`

### 主要方法

- `train_one_epoch(data_loader, log_interval=10)`: 训练一个epoch
- `fit(train_dataloader, val_dataloader=None)`: 训练模型
- `evaluate(model, data_loader)`: 评估模型
- `predict(model, data_loader)`: 模型预测

## MatchTrainer

MatchTrainer 是一个用于匹配/检索任务的训练器，支持多种训练模式。

### 参数说明

- `model` (nn.Module): 任何匹配模型
- `mode` (int): 训练模式，可选值：
  - 0: 逐点式(point-wise)
  - 1: 成对式(pair-wise)
  - 2: 列表式(list-wise)
- `optimizer_fn` (torch.optim): 同CTRTrainer
- `optimizer_params` (dict): 同CTRTrainer
- `scheduler_fn` (torch.optim.lr_scheduler): 同CTRTrainer
- `scheduler_params` (dict): 同CTRTrainer
- `n_epoch` (int): 同CTRTrainer
- `earlystop_patience` (int): 同CTRTrainer
- `device` (str): 同CTRTrainer
- `gpus` (list): 同CTRTrainer
- `model_path` (str): 同CTRTrainer

### 主要方法

- `train_one_epoch(data_loader, log_interval=10)`: 训练一个epoch
- `fit(train_dataloader, val_dataloader=None)`: 训练模型
- `evaluate(model, data_loader)`: 评估模型
- `predict(model, data_loader)`: 模型预测
- `inference_embedding(model, mode, data_loader, model_path)`: 推理嵌入向量
  - `mode`: 可选 "user" 或 "item"

## MTLTrainer

MTLTrainer 是一个用于多任务学习的训练器，支持多种自适应损失权重方法。

### 参数说明

- `model` (nn.Module): 任何多任务学习模型
- `task_types` (list): 任务类型列表，支持 ["classification", "regression"]
- `optimizer_fn` (torch.optim): 同CTRTrainer
- `optimizer_params` (dict): 同CTRTrainer
- `scheduler_fn` (torch.optim.lr_scheduler): 同CTRTrainer
- `scheduler_params` (dict): 同CTRTrainer
- `adaptive_params` (dict): 自适应损失权重方法参数，支持：
  - `{"method": "uwl"}`: 不确定性加权损失
  - `{"method": "metabalance"}`: 元平衡方法
  - `{"method": "gradnorm", "alpha": 0.16}`: 梯度归一化方法
- `n_epoch` (int): 同CTRTrainer
- `earlystop_taskid` (int): 用于早停的任务ID，默认为0
- `earlystop_patience` (int): 同CTRTrainer
- `device` (str): 同CTRTrainer
- `gpus` (list): 同CTRTrainer
- `model_path` (str): 同CTRTrainer

### 主要方法

- `train_one_epoch(data_loader)`: 训练一个epoch
- `fit(train_dataloader, val_dataloader, mode='base', seed=0)`: 训练模型
- `evaluate(model, data_loader)`: 评估模型
- `predict(model, data_loader)`: 模型预测

### 特殊功能

1. 支持多种自适应损失权重方法：
   - UWL (Uncertainty Weighted Loss)
   - MetaBalance
   - GradNorm

2. 支持多任务早停：
   - 可以基于指定任务的性能进行早停
   - 保存验证集上最佳性能的模型权重

3. 支持多种任务类型组合：
   - 分类任务
   - 回归任务

4. 训练日志记录：
   - 记录每个任务的损失
   - 记录损失权重（当使用自适应方法时）
   - 记录验证集上的性能指标