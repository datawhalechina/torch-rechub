---
title: DeepFM 使用示例
description: 用 DeepFM 预测广告点击，检查 FM 分支的嵌入维度和验证 AUC
---

# DeepFM 使用示例

## 1. 模型简介与适用场景

DeepFM 给每条广告曝光预测一个点击概率。本例把 Criteo 的类别字段同时交给 FM 和多层网络。FM 计算两两字段的交互，多层网络还能结合连续字段学习更复杂的组合。

**论文**: [DeepFM: A Factorization-Machine based Neural Network for CTR Prediction](https://arxiv.org/abs/1703.04247)

### 模型结构

<div align="center">
  <img src="/img/models/deepfm_arch.png" alt="DeepFM Model Architecture" width="600"/>
</div>

同一个类别字段进入 FM 和多层网络时，两边查的是该字段的同一张嵌入表。模型把线性项、FM 交互项和多层网络的分数相加，再用 sigmoid 得到点击概率。

---

## 2. 数据准备与预处理

先在同一个 Python 会话中完成[准备 Criteo 输入](/zh/guide/quick_start#criteo-input)。下面会用到两组特征 `dense_feas`、`sparse_feas` 和三个 DataLoader。若刚刚训练过另一个模型，可以复用数据切分，创建模型前重新构造特征对象。

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.ranking import DeepFM

model = DeepFM(
    deep_features=dense_feas + sparse_feas,  # Deep 部分负责学习高阶非线性关系
    fm_features=sparse_feas,       # FM 部分只对稀疏特征做二阶交互
    mlp_params={
        "dims": [256, 128],        # MLP 隐藏层维度
        "dropout": 0.2,            # dropout 比率
        "activation": "relu"       # 激活函数
    }
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `deep_features` | `list[Feature]` | Deep 部分的特征列表，通常包含全部特征 | 连续特征 + 类别特征 |
| `fm_features` | `list[Feature]` | FM 部分的特征列表，通常包含 SparseFeature | 全部类别特征 |
| `mlp_params.dims` | `list[int]` | MLP 每层神经元数量 | `[256, 128]` 或 `[256, 128, 64]` |
| `mlp_params.dropout` | `float` | Dropout 比率，防止过拟合 | 0.1 ~ 0.3 |
| `mlp_params.activation` | `str` | 激活函数 (`relu`, `prelu`, `sigmoid`) | `"relu"` |

---

## 4. 训练过程与代码示例

### 4.1 创建训练器并训练

```python
import os
import torch
from torch_rechub.trainers import CTRTrainer

torch.manual_seed(2022)
os.makedirs("./saved/deepfm", exist_ok=True)

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={
        "lr": 1e-3,               # 学习率
        "weight_decay": 1e-3       # L2 正则化
    },
    n_epoch=2,                    # 最大训练轮数
    earlystop_patience=10,         # 早停耐心值
    device="cpu",                  # 使用 "cuda:0" 进行 GPU 训练
    model_path="./saved/deepfm"    # 模型保存路径
)

# 开始训练
ctr_trainer.fit(train_dl, val_dl)
```

### 4.2 训练日志说明

这里 `fit` 收到了验证集，每轮训练后都会计算验证 AUC。例子最多跑两轮，之后用测试集评估验证 AUC 最好的一轮。仓库样本很小，换个随机种子就可能看到不同的分数。

---

## 5. 模型评估与结果分析

### 5.1 测试集评估

```python
auc = ctr_trainer.evaluate(ctr_trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

仓库随附的 Criteo 样本只有 115 行，适合检查模型能否训练和评估。要比较不同模型的点击预测效果，请使用更大的数据集，并为各模型保留同样的训练、验证和测试样本。[基准测试](/zh/tools/benchmark)

## 6. 参数调优建议

### 6.1 超参数调优优先级

可以从示例中的学习率 `1e-3` 开始，每次只改一个参数。改变 `embed_dim` 时，所有参与 FM 交互的类别字段要使用相同维度，才能把向量堆在一起计算交互。

### 6.2 调优技巧

下面是一种可选的训练配置，使用 CUDA，每两轮把学习率乘以 0.8。要使用它，在第 4 节创建训练器时替换原配置即可。

```python
import os

# 使用学习率调度器
os.makedirs("./saved/deepfm", exist_ok=True)

ctr_trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=100,
    earlystop_patience=10,
    device="cuda:0",
    model_path="./saved/deepfm",
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params={"step_size": 2, "gamma": 0.8}  # 每2个epoch衰减
)
```

---

## 7. 常见问题与解决方案

### Q1: 训练损失不下降？
- 检查学习率是否过大或过小
- 确认数据预处理是否正确（缺失值处理、归一化）
- 检查标签分布是否极度不平衡

### Q2: 过拟合（训练 AUC 高，验证 AUC 低）？
- 增加 dropout 比率（0.2 → 0.5）
- 增大 weight_decay（1e-3 → 1e-2）
- 减少 MLP 层数或维度
- 使用更多训练数据

### Q3: 如何选择 deep_features 和 fm_features?
本例把连续字段和类别字段都给 `deep_features`，把要做两两交互的类别字段给 `fm_features`。两份列表可以引用同一个类别特征对象。

### Q4: GPU 内存不足？
- 减小 `batch_size`（如 2048 → 512）
- 减小 `embed_dim`（如 16 → 8）
- 减小 MLP 维度

---

## 8. 模型可视化

可以画出当前配置，检查 FM 和多层网络分别接收哪些输入。

### 安装依赖

```bash
pip install torch-rechub[visualization]
# 系统级依赖:
# Ubuntu: sudo apt-get install graphviz
# macOS: brew install graphviz
# Windows: choco install graphviz
```

### 可视化 DeepFM 模型

```python
from torch_rechub.utils.visualization import visualize_model

# 自动生成输入并可视化（在 Jupyter 中直接显示内嵌图像）
graph = visualize_model(model, depth=4)

# 保存为高清 PNG（适合论文/文档）
visualize_model(model, save_path="deepfm_architecture.png", dpi=300)

# 保存为 PDF
visualize_model(model, save_path="deepfm_architecture.pdf")
```

### DeepFM 架构图

![DeepFM 模型架构图](/img/models/deepfm_arch.png)

`visualize_model` 会按特征定义生成一批输入。图太密时，可以减小 `depth`，先看模块之间的连接。

---

## 9. ONNX 导出

下面用 ONNX Runtime 加载导出的模型并检查输出形状。示例传入全零张量，只用来检查接口。要比对真实样本的预测值，请沿用训练时的编码方式。[部署示例](/zh/serving/demo)

### 导出模型

```python
from torch_rechub.utils.onnx_export import ONNXExporter

exporter = ONNXExporter(model, device="cpu")

# 导出 DeepFM 模型
exporter.export("deepfm.onnx", verbose=True)

# 查看输入信息
info = exporter.get_input_info()
print(info)
```

### 使用 ONNX Runtime 推理

```python
import onnxruntime as ort
import numpy as np

session = ort.InferenceSession("deepfm.onnx")

# 查看模型输入
for inp in session.get_inputs():
    print(f"  {inp.name}: shape={inp.shape}, type={inp.type}")

# 构造输入并推理
input_feed = {}
for inp in session.get_inputs():
    if "int" in inp.type.lower():
        input_feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.int64)
    else:
        input_feed[inp.name] = np.zeros([d if isinstance(d, int) else 1 for d in inp.shape], dtype=np.float32)

output = session.run(None, input_feed)
print(f"Output shape: {output[0].shape}")
```

> 排序模型（DeepFM / WideDeep / DCN）的 ONNX 导出是完整模型导出，不涉及 Tower 分离。双塔模型（DSSM / YoutubeDNN）支持 `mode="user"/"item"` 分别导出。

---

## 运行仓库示例

```bash
cd examples/ranking
uv run python run_criteo.py --model_name deepfm --epoch 1 --device cpu
```

