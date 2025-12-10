---
title: 实验追踪
description: Torch-RecHub 实验记录与对比
---

# 实验跟踪（简化版）

torch-rechub 内置了可选的实验跟踪能力，支持：
- **Weights & Biases (wandb)**
- **SwanLab**
- **TensorBoardX**

设计目标：默认零开销，不想记录就不创建 logger；想记录时，只需把 `model_logger` 传给 Trainer。

## 安装

```bash
# 安装全部
pip install torch-rechub[tracking]

# 或按需
pip install wandb
pip install swanlab
pip install tensorboardX
```

## 快速使用

```python
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger

logger = WandbLogger(project="my-ctr", name="exp1")  # 可选

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3},
    n_epoch=10,
    model_logger=logger,   # 不想记录就传 None
)

trainer.fit(train_dl, val_dl)
```

### 同时使用多个 logger

`model_logger` 可以是单个实例或列表：

```python
tb = TensorBoardXLogger(log_dir="./runs/exp1")
wb = WandbLogger(project="my-ctr", name="exp1")
sb = SwanLabLogger(project="my-ctr")

trainer = CTRTrainer(model, model_logger=[tb, wb])
```

### 代码中设置API key
```python
import os
os.environ['WANDB_API_KEY'] = "your API_KEY"
os.environ['SWANLAB_API_KEY'] = "your API_KEY"
```

### 不记录时的开销

当 `model_logger=None` 时，训练流程不会调用任何跟踪代码，保持与原始实现一致、零额外开销。

## API 概览

```python
class BaseLogger:
    log_metrics(metrics: Dict[str, Any], step: Optional[int] = None)
    log_hyperparams(params: Dict[str, Any])
    finish()
```

- **WandbLogger(project, name=None, config=None, tags=None, notes=None, dir=None, …)**
- **SwanLabLogger(project=None, experiment_name=None, description=None, config=None, logdir=None, …)**
- **TensorBoardXLogger(log_dir, comment=\"\", …)**

## Trainer 行为（自动记录）

- `train/loss`、`learning_rate`
- 验证集指标：`val/auc`（CTR/Match）、`val/task_i_score`（MTL）、`val/accuracy`（Seq）
- 仅当提供 `model_logger` 时才记录

## 常见问题

1) **不用跟踪怎么办？**  
   不传 `model_logger`（默认 None）。

2) **能否关掉超参数记录？**  
   目前默认会在训练开始时记录一次超参数；如需自定义，可自行调用/忽略。

3) **某个库未安装？**  
   创建对应 logger 时会抛出友好提示，不影响训练。

4) **想自定义指标？**  
   可在外部拿到 logger 实例，自行调用 `log_metrics`。

```python
logger = WandbLogger(project="demo")
# 自定义记录
logger.log_metrics({"custom_metric": 0.9}, step=epoch)
```

## 最佳实践
- 本地快速调试：用 TensorBoardX
- 团队/远程协作：用 WandB 或 SwanLab
- 需要同时本地+云端：传入多个 logger 列表
- 未指定 logger 时保持最小开销
