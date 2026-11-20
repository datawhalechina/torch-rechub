---
title: Wide&Deep 使用示例
description: 把 Criteo 的连续字段和类别字段分别交给 WideDeep 的线性层与多层网络
---

# Wide&Deep 使用示例

## 1. 模型简介与适用场景

Wide&Deep 为一条广告曝光算两次分数。这个 Criteo 示例把连续字段交给线性层，把类别字段的嵌入交给多层网络，然后合并两个分数得到点击概率。

**论文**: [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792)

### 模型结构

`wide_features` 决定哪些字段直接进入线性层，`deep_features` 决定哪些字段经过多层网络。输入分组由你决定，两组允许重叠。

---

## 2. 数据准备与预处理

先在同一个 Python 会话中完成[准备 Criteo 输入](/zh/guide/quick_start#criteo-input)。下面会用到两组特征 `dense_feas`、`sparse_feas` 和三个 DataLoader。若刚刚训练过另一个模型，可以复用数据切分，创建模型前重新构造特征对象。

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.ranking import WideDeep

model = WideDeep(
    wide_features=dense_feas,      # Wide 部分更偏“记忆”，适合直接吃连续特征或手工交叉特征
    deep_features=sparse_feas,     # Deep 部分更偏“泛化”，适合学习稀疏类别特征的组合关系
    mlp_params={
        "dims": [256, 128],        # MLP 隐藏层维度
        "dropout": 0.2,
        "activation": "relu"
    }
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `wide_features` | `list[Feature]` | Wide 部分的特征列表 | 连续特征 / 交叉特征 |
| `deep_features` | `list[Feature]` | Deep 部分的特征列表 | 类别特征 |
| `mlp_params.dims` | `list[int]` | MLP 各层维度 | `[256, 128]` |
| `mlp_params.dropout` | `float` | Dropout 比率 | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | 激活函数 | `"relu"` |

如果有手工组合字段，也可以把它放到 Wide 分支。例如把“城市 × 时段”编码为一个类别字段，线性层就能直接为这个组合打分。

---

## 4. 训练过程与代码示例

```python
import os
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/widedeep", exist_ok=True)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=2,
    earlystop_patience=10,
    device="cpu",                   # GPU: "cuda:0"
    model_path="./saved/widedeep"
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

要检验一个新字段是否有用，可以保持训练和验证数据不变，只把这个字段加进 `wide_features` 或 `deep_features`，再比较验证 AUC。

## 7. 常见问题与解决方案

### 什么时候值得试 DeepFM？

Wide 分支直接使用你提供的字段。如果想让模型自动计算多个类别字段间的两两交互，可以在相同数据上比较 DeepFM，再同时查看验证 AUC 和训练耗时。[DeepFM 教程](/zh/tutorials/models/ranking/deepfm)

---

## 运行仓库示例

```bash
cd examples/ranking
uv run python run_criteo.py --model_name widedeep --epoch 1 --device cpu
```
