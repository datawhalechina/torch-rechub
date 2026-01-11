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
# fit() 结束后 logger 会自动关闭，无需手动调用 finish()
```

### 同时使用多个 logger

`model_logger` 可以是单个实例或列表：

```python
from torch_rechub.basic.tracking import TensorBoardXLogger, WandbLogger, SwanLabLogger

tb = TensorBoardXLogger(log_dir="./runs/exp1")
wb = WandbLogger(project="my-ctr", name="exp1")
sb = SwanLabLogger(project="my-ctr")

trainer = CTRTrainer(model, model_logger=[tb, wb, sb])
```

### 代码中设置 API key

```python
import os
os.environ['WANDB_API_KEY'] = "your API_KEY"
os.environ['SWANLAB_API_KEY'] = "your API_KEY"
```

### 不记录时的开销

当 `model_logger=None` 时，训练流程不会调用任何跟踪代码，保持与原始实现一致、零额外开销。

## 支持的 Trainer

| Trainer        | 支持 | 自动记录的指标                                                                 |
| -------------- | ---- | ------------------------------------------------------------------------------ |
| `CTRTrainer`   | ✅    | `train/loss`, `learning_rate`, `val/auc`                                       |
| `MatchTrainer` | ✅    | `train/loss`, `learning_rate`, `val/auc`                                       |
| `MTLTrainer`   | ✅    | `train/task_i_loss`, `learning_rate`, `val/task_i_score`, `loss_weight/task_i` |
| `SeqTrainer`   | ✅    | `train/loss`, `learning_rate`, `val/loss`, `val/accuracy`                      |

## 生命周期管理

### 自动管理（推荐）

Trainer 会自动管理 logger 的生命周期：

```python
logger = WandbLogger(project="demo")
trainer = CTRTrainer(model, model_logger=logger)

trainer.fit(train_dl, val_dl)
# ✅ fit() 内部会自动调用：
#    1. logger.log_hyperparams() - 训练开始时
#    2. logger.log_metrics()     - 每个 epoch 结束时
#    3. logger.finish()          - 训练结束时
```

### 在 fit() 后记录额外指标

由于 `fit()` 结束时会自动调用 `finish()` 关闭 logger，如果需要在训练后记录测试集指标，有以下方案：

**方案 A：不传给 Trainer，完全手动管理**

```python
logger = WandbLogger(project="demo")

# 不传给 Trainer
trainer = CTRTrainer(model, model_logger=None)
trainer.fit(train_dl, val_dl)

# 手动记录
test_auc = trainer.evaluate(trainer.model, test_dl)
logger.log_metrics({"test/auc": test_auc})
logger.finish()  # 手动关闭
```

**方案 B：创建新 logger 记录测试指标**

```python
logger = WandbLogger(project="demo", name="train-exp1")
trainer = CTRTrainer(model, model_logger=logger)
trainer.fit(train_dl, val_dl)  # logger 已关闭

# 创建新 logger 记录测试结果
test_logger = WandbLogger(project="demo", name="test-exp1")
test_auc = trainer.evaluate(trainer.model, test_dl)
test_logger.log_metrics({"test/auc": test_auc})
test_logger.finish()
```

## API 概览

### BaseLogger 接口

```python
class BaseLogger:
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """记录指标值"""
        pass

    def log_hyperparams(self, params: Dict[str, Any]) -> None:
        """记录超参数"""
        pass

    def finish(self) -> None:
        """关闭 logger，释放资源"""
        pass
```

### Logger 实现类

**WandbLogger**

```python
WandbLogger(
    project: str,              # 项目名称（必填）
    name: str = None,          # 运行名称
    config: dict = None,       # 初始配置/超参数
    tags: List[str] = None,    # 标签
    notes: str = None,         # 备注
    dir: str = None,           # 本地缓存目录
    **kwargs                   # 其他传给 wandb.init() 的参数
)
```

**SwanLabLogger**

```python
SwanLabLogger(
    project: str = None,           # 项目名称
    experiment_name: str = None,   # 实验名称
    description: str = None,       # 描述
    config: dict = None,           # 初始配置
    logdir: str = None,            # 日志目录
    **kwargs                       # 其他传给 swanlab.init() 的参数
)
```

**TensorBoardXLogger**

```python
TensorBoardXLogger(
    log_dir: str,          # 日志目录（必填）
    comment: str = "",     # 目录名后缀
    **kwargs               # 其他传给 SummaryWriter 的参数
)
```

## 完整示例

```python
import os
import torch
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.tracking import WandbLogger, SwanLabLogger, TensorBoardXLogger

# 设置 API key（可选，也可通过命令行登录）
os.environ['WANDB_API_KEY'] = "your_key"
os.environ['SWANLAB_API_KEY'] = "your_key"

# 配置
SEED = 2022
EPOCH = 10
LR = 1e-3
BATCH_SIZE = 2048

# 选择要使用的 logger（可多选）
loggers = []

# 本地 TensorBoard
loggers.append(TensorBoardXLogger(log_dir=f"./runs/deepfm-{SEED}"))

# 云端 WandB
loggers.append(WandbLogger(
    project="ctr-experiment",
    name=f"deepfm-{SEED}",
    config={"seed": SEED, "lr": LR, "batch_size": BATCH_SIZE},
    tags=["deepfm", "criteo"]
))

# 模型和数据准备（略）
model = DeepFM(...)
train_dl, val_dl, test_dl = ...

# 训练
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": LR, "weight_decay": 1e-5},
    n_epoch=EPOCH,
    earlystop_patience=3,
    device="cuda:0",
    model_logger=loggers,  # 传入 logger 列表
)

trainer.fit(train_dl, val_dl)
# 训练结束，所有 logger 自动关闭

# 评估测试集
test_auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {test_auc}")
```

## 常见问题

**1. 不想使用跟踪功能？**

不传 `model_logger` 参数或传 `None`（默认值）。

**2. 某个库未安装怎么办？**

创建对应 logger 时会抛出友好的 `ImportError` 提示，不影响其他功能。

**3. 为什么在 `fit()` 后调用 `log_metrics()` 报错？**

因为 `fit()` 结束时会自动调用 `finish()` 关闭 logger。如需在训练后记录指标，请参考上文「在 fit() 后记录额外指标」章节。

**4. 能否关闭自动记录的超参数？**

目前不支持，Trainer 会在训练开始时自动记录一次超参数。如需完全自定义，可不传 `model_logger`，手动管理。

**5. TensorBoard 如何查看？**

```bash
tensorboard --logdir ./runs
```

## 最佳实践

| 场景               | 推荐方案                         |
| ------------------ | -------------------------------- |
| 本地快速调试       | `TensorBoardXLogger`             |
| 团队协作/云端记录  | `WandbLogger` 或 `SwanLabLogger` |
| 同时本地+云端      | 传入多个 logger 组成的列表       |
| 不需要跟踪         | 不传 `model_logger`（零开销）    |
| 需要记录测试集指标 | 手动管理 logger 或创建新 logger  |
