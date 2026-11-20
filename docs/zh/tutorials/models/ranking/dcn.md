---
title: DCN / DCNv2 使用示例
description: 用相同的 Criteo 特征比较 DCN 和 DCNv2 的交叉层
---

# DCN / DCNv2 使用示例

## 1. 模型简介与适用场景

DCN 用交叉层学习输入字段的组合关系，同时用多层网络处理同一批字段。例如，一条曝光里的两个字段可以单独起作用，也可以在同时出现时影响点击分数。下面用同一份 Criteo 样本训练 DCN 或 DCNv2。

**论文**:
- DCN: [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123)
- DCNv2: [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)

### 核心区别

DCN 的每个交叉层使用一个权重向量。本项目的 DCNv2 默认把多个低秩专家的结果组合起来。传入 `use_low_rank_mixture=False`，才会使用完整的矩阵交叉层。两种模型都接受同一组输入特征。

---

## 2. 数据准备与预处理

先在同一个 Python 会话中完成[准备 Criteo 输入](/zh/guide/quick_start#criteo-input)。下面会用到两组特征 `dense_feas`、`sparse_feas` 和三个 DataLoader。若刚刚训练过另一个模型，可以复用数据切分，创建模型前重新构造特征对象。

## 3. 模型配置与参数说明

选择一个模型构造并训练，将 `DCN` 换为 `DCNv2` 可运行另一种交叉层。独立对比时从数据准备重新开始，避免共用已训练的嵌入参数。

```python
from torch_rechub.models.ranking import DCN, DCNv2

model = DCN(
    features=dense_feas + sparse_feas,
    n_cross_layers=3,
    mlp_params={"dims": [256, 128]},
)
```

### 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `features` | `list[Feature]` | 全部特征列表 (Dense + Sparse) | 所有特征 |
| `n_cross_layers` | `int` | Cross Network 的层数 | 2 ~ 4 |
| `mlp_params.dims` | `list[int]` | Deep Network 各层维度 | `[256, 128]` |
| `mlp_params.dropout` | `float` | Dropout 比率 | 0.1 ~ 0.3 |

---

## 4. 训练过程与代码示例

```python
import os
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/dcn", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=2,
    earlystop_patience=10,
    device="cpu",
    model_path="./saved/dcn"
)

trainer.fit(train_dl, val_dl)
```

---

## 5. 模型评估与结果分析

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

仓库随附的 Criteo 样本只有 115 行，适合检查模型能否训练和评估。要比较不同模型的点击预测效果，请使用更大的数据集，并为各模型保留同样的训练、验证和测试样本。[基准测试](/zh/tools/benchmark)

## 6. 参数调优建议

示例先用 3 个交叉层。要判断层数是否有帮助，可以固定数据切分和 MLP 的宽度，只改 `n_cross_layers`，比较验证集 AUC。换成 DCNv2 时记下是否使用低秩专家，这也会影响参数量。

## 7. 常见问题与解决方案

### 需要自己准备交叉特征吗？

直接传入 Criteo 的原始字段即可，交叉层会在训练中学习组合。如果业务数据里已经有手工组合字段，可以在相同切分下比较加上这一列前后的结果。

---

## 运行仓库示例

```bash
cd examples/ranking
uv run python run_criteo.py --model_name dcn --epoch 1 --device cpu
```
