---
title: 快速开始
description: Torch-RecHub 快速入门指南，5分钟跑通第一个推荐模型
---

# 快速开始

本教程将帮助你在 **5 分钟内** 跑通一个完整的推荐模型训练流程。

## 安装

```bash
pip install torch-rechub
```

> 💡 建议同时安装可选依赖以获得完整功能：
> ```bash
> pip install torch-rechub[all]  # 包含 ONNX 导出、实验追踪等功能
> ```

---

## 示例 1：CTR 预测（精排模型）

这是一个完整的、可直接运行的 DeepFM 模型训练示例：

```python
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# ========== 1. 准备数据 ==========
# 使用内置的 Criteo 样本数据（100条记录，用于演示）
# 完整数据下载：https://www.kaggle.com/c/criteo-display-ad-challenge
data_url = "https://raw.githubusercontent.com/datawhalechina/torch-rechub/main/examples/ranking/data/criteo/criteo_sample.csv"
data = pd.read_csv(data_url)
print(f"数据集大小: {len(data)} 条记录")

# ========== 2. 特征处理 ==========
# Criteo 数据集包含 13 个数值特征 (I1-I13) 和 26 个类别特征 (C1-C26)
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# 填充缺失值
data[sparse_features] = data[sparse_features].fillna("-996")
data[dense_features] = data[dense_features].fillna(0)

# 数值特征归一化
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# 类别特征编码
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat].astype(str))

# ========== 3. 定义特征类型 ==========
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

# ========== 4. 创建 DataLoader ==========
x = data.drop(columns=["label"])
y = data["label"]

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],
    batch_size=256
)

# ========== 5. 定义模型 ==========
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"}
)

# ========== 6. 训练模型 ==========
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=5,
    device="cpu",  # 使用 GPU: "cuda:0"
)

trainer.fit(train_dl, val_dl)

# ========== 7. 评估模型 ==========
auc = trainer.evaluate(model, test_dl)
print(f"测试集 AUC: {auc:.4f}")
```

**预期输出：**

```
train: 100%|██████████| 1/1 [00:00<00:00,  4.47it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 333.15it/s]
epoch: 0 validation: auc: 0.3666666666666667
epoch: 1
train: 100%|██████████| 1/1 [00:00<00:00, 111.11it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 399.08it/s]
epoch: 1 validation: auc: 0.3666666666666667
epoch: 2
train: 100%|██████████| 1/1 [00:00<00:00, 95.60it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 492.75it/s]
epoch: 2 validation: auc: 0.33333333333333337
epoch: 3
train: 100%|██████████| 1/1 [00:00<00:00, 90.90it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 499.20it/s]
epoch: 3 validation: auc: 0.3
epoch: 4
train: 100%|██████████| 1/1 [00:00<00:00, 79.91it/s]
validation: 100%|██████████| 1/1 [00:00<00:00, 500.27it/s]
epoch: 4 validation: auc: 0.3333333333333333
validation: 100%|██████████| 1/1 [00:00<00:00, 249.71it/s]测试集 AUC: 0.9545
```

---

## 示例 2：召回模型（双塔 DSSM）

这是一个完整的、可直接运行的 DSSM 双塔模型训练示例：

```python
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import df_to_dict, MatchDataGenerator
from torch_rechub.utils.match import generate_seq_feature_match, gen_model_input

torch.manual_seed(2022)

# ========== 1. 准备数据 ==========
# 使用内置的 MovieLens 样本数据
data_url = "https://raw.githubusercontent.com/datawhalechina/torch-rechub/main/examples/matching/data/ml-1m/ml-1m_sample.csv"
data = pd.read_csv(data_url)
print(f"数据集大小: {len(data)} 条记录")

# 处理 genres 特征
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])

# ========== 2. 特征编码 ==========
user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]

feature_max_idx = {}
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat]) + 1  # +1 为 padding 预留 0
    feature_max_idx[feat] = data[feat].max() + 1

# ========== 3. 定义用户塔和物品塔特征 ==========
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_profile = data[user_cols].drop_duplicates("user_id")
item_profile = data[item_cols].drop_duplicates("movie_id")

# ========== 4. 生成序列特征和训练数据 ==========
df_train, df_test = generate_seq_feature_match(
    data,
    user_col,
    item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,  # point-wise
    neg_ratio=3,
    min_item=0
)

x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train["label"]
x_train = {k: v for k, v in x_train.items() if k != "label"}
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)

# ========== 5. 定义特征类型 ==========
user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
user_features += [
    SequenceFeature(
        "hist_movie_id",
        vocab_size=feature_max_idx["movie_id"],
        embed_dim=16,
        pooling="mean",
        shared_with="movie_id"
    )
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in item_cols
]

# ========== 6. 创建 DataLoader ==========
all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=256)

# ========== 7. 定义模型 ==========
model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [128, 64], "activation": "prelu"},
    item_params={"dims": [128, 64], "activation": "prelu"},
)

# ========== 8. 训练模型 ==========
trainer = MatchTrainer(
    model,
    mode=0,  # point-wise
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",
    model_path="./",
)

trainer.fit(train_dl)

# ========== 9. 导出嵌入向量 ==========
user_embedding = trainer.inference_embedding(model, mode="user", data_loader=test_dl, model_path="./")
item_embedding = trainer.inference_embedding(model, mode="item", data_loader=item_dl, model_path="./")

print(f"用户嵌入维度: {user_embedding.shape}")
print(f"物品嵌入维度: {item_embedding.shape}")
```

**预期输出：**

```
n_train: 384, n_test: 2
0 cold start user dropped 
epoch: 0
train: 100%|██████████| 2/2 [00:19<00:00,  9.81s/it]
epoch: 1
train: 100%|██████████| 2/2 [00:19<00:00,  9.65s/it]
user inference: 100%|██████████| 1/1 [00:04<00:00,  4.90s/it]
item inference: 100%|██████████| 1/1 [00:05<00:00,  5.23s/it]用户嵌入维度: torch.Size([2, 64])
物品嵌入维度: torch.Size([93, 64])
```

---

## 下一步

🎉 恭喜！你已经成功运行了第一个推荐模型。接下来可以：

- 📚 查看 [模型文档](../models/intro.md) 了解更多模型（DCN、MMOE、YoutubeDNN 等）
- 🔧 查看 [完整示例](https://github.com/datawhalechina/torch-rechub/tree/main/examples) 了解更多数据集和训练技巧
- 🚀 查看 [模型部署](../serving/intro.md) 了解如何导出 ONNX 和构建向量索引
- 📊 查看 [实验追踪](../tools/tracking.md) 了解如何使用 MLflow/TensorBoard 记录实验

---

## 常见问题

### Q: 如何使用 GPU 训练？

将 `device="cpu"` 改为 `device="cuda:0"`：

```python
trainer = CTRTrainer(model, device="cuda:0", ...)
```

### Q: 如何保存和加载模型？

```python
# 保存
torch.save(model.state_dict(), "model.pth")

# 加载
model.load_state_dict(torch.load("model.pth"))
```

### Q: 如何导出 ONNX 模型？

```python
trainer.export_onnx("model.onnx")

# 双塔模型可分别导出
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```
