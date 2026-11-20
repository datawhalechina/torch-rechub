---
title: 实验追踪
description: Torch-RecHub 实验记录与对比
---

# 实验追踪

想比较两次 DeepFM 训练，可以先把每轮的损失和验证 AUC 记下来。自己在本机看曲线，用 TensorBoardX 把记录写到文件。团队已经用 W&B 或 SwanLab 时，把对应的 logger 传给同一个 `model_logger` 参数就行。只看终端的训练进度无需设置这个参数。

## 保存一条本地训练曲线

在运行训练的 Python 环境里安装 `tensorboardX`，再安装用来查看曲线的 `tensorboard`。下面接在快速入门的 DeepFM 示例后，沿用已经定义好的 `model`、`train_dl` 和 `val_dl`。[DeepFM 快速入门](/zh/guide/quick_start)

```python
from torch_rechub.basic.tracking import TensorBoardXLogger
from torch_rechub.trainers import CTRTrainer

logger = TensorBoardXLogger(log_dir="./runs/deepfm")
trainer = CTRTrainer(model, n_epoch=2, device="cpu", model_logger=logger)
trainer.fit(train_dl, val_dl)
```

训练结束后，`./runs/deepfm` 目录里会有事件文件。从同一工作目录启动 TensorBoard，打开终端打印的地址即可看到 `train/loss`、`learning_rate` 和 `val/auc`。做第二次实验时换一个日志目录，比如 `./runs/deepfm-lr2`，两条曲线就能分开看。

```bash
tensorboard --logdir ./runs
```

## 用 W&B 或 SwanLab 记录实验

先在当前 Python 环境安装要用的包，W&B 对应 `wandb`，SwanLab 对应 `swanlab`。需要同时使用三个后端时，也可以运行 `pip install "torch-rechub[tracking]"`。使用平台账号登录，或在运行训练的终端设置对应平台的密钥。

```bash
# bash/zsh：仅在当前 shell 中设置
export WANDB_API_KEY="..."
export SWANLAB_API_KEY="..."
```

```powershell
# PowerShell：仅在当前会话中设置
$env:WANDB_API_KEY = "..."
$env:SWANLAB_API_KEY = "..."
```

下面把训练记录写入 W&B 的 `my-ctr` 项目，这次实验名为 `exp1`。使用 SwanLab 时，把 logger 换成 `SwanLabLogger(project="my-ctr", experiment_name="exp1")`。

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

如果既想保留本地曲线，又要把指标发给团队，把需要的 logger 放进列表。下面展示三个后端一起记录的写法，可按需取其中几个。

```python
from torch_rechub.basic.tracking import TensorBoardXLogger, WandbLogger, SwanLabLogger

tb = TensorBoardXLogger(log_dir="./runs/exp1")
wb = WandbLogger(project="my-ctr", name="exp1")
sb = SwanLabLogger(project="my-ctr")

trainer = CTRTrainer(model, model_logger=[tb, wb, sb])
```

## 支持的 Trainer

训练开始时会记录超参数，训练指标按轮记录。传入验证集才会看到对应的验证指标。多任务的 `i` 从 0 开始，与传给训练器的任务顺序一致。

| 训练器 | 常用记录项 |
| --- | --- |
| `CTRTrainer`、`MatchTrainer` | `train/loss`、`learning_rate`、`val/auc` |
| `MTLTrainer` | `train/task_i_loss`、`val/task_i_score`、`learning_rate`。使用自适应任务权重时还会记录 `loss_weight/task_i` |
| `SeqTrainer` | `train/loss`、`learning_rate`、`val/loss`、`val/accuracy` |
| `RQVAETrainer` | `train/loss`、`train/recon_loss` 和评估时的 `val/collision_rate` |

### 在 fit() 后记录额外指标

交给 Trainer 的 logger 会在 `fit()` 正常结束时关闭。如果还要记录测试 AUC，可以新开一条实验记录，名字和训练用的区分开。

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

如果只关心最终的测试 AUC，可以自己管理 logger，在测试结束后写一次结果并关闭。

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
