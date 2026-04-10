---
title: 快速开始
description: Torch-RecHub 快速入门指南，5 分钟跑通第一个推荐模型
---

# 快速开始

本页提供两条可以直接跑通的最小链路：

- CTR 排序模型：`DeepFM`
- 双塔召回模型：`DSSM`

以下命令和代码默认在**仓库根目录**执行，并优先使用仓库内置的样本数据，不依赖外网下载。

## 安装

```bash
pip install torch-rechub
```

如果需要 ONNX 导出、可视化、实验追踪等功能，建议安装可选依赖：

```bash
pip install "torch-rechub[all]"
```

---

## 示例 1：CTR 预测（DeepFM）

这是一条完整的排序训练链路，使用仓库内置的 `Criteo` 样本数据。

```python
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# ========== 1. 加载样本数据 ==========
data_path = "examples/ranking/data/criteo/criteo_sample.csv"
data = pd.read_csv(data_path)
print(f"数据集大小: {len(data)}")

# ========== 2. 特征处理 ==========
# Criteo 的 I1-I13 是连续特征，C1-C26 是离散特征
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# 稀疏特征填字符串，连续特征填 0，和仓库 examples 保持一致
data[sparse_features] = data[sparse_features].fillna("-996")
data[dense_features] = data[dense_features].fillna(0)

# 连续特征做归一化，方便 MLP 更稳定地训练
scaler = MinMaxScaler()
data[dense_features] = scaler.fit_transform(data[dense_features])

# 稀疏特征转成从 0 开始的类别索引，后面会交给 SparseFeature 做 embedding
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat].astype(str))

# DenseFeature / SparseFeature 是 Torch-RecHub 的统一特征抽象
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=data[name].nunique(), embed_dim=16)
    for name in sparse_features
]

# ========== 3. 构建 DataLoader ==========
# DataGenerator 负责把 pandas 风格输入包装成可直接训练的 DataLoader
x = data.drop(columns=["label"])
y = data["label"]

dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    split_ratio=[0.7, 0.1],
    batch_size=256,
)

# ========== 4. 定义模型 ==========
# DeepFM 同时建模低阶特征交互（FM）和高阶非线性关系（MLP）
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

# ========== 5. 训练与评估 ==========
# CTRTrainer 封装了训练、验证、早停和评估逻辑
trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/quick_start_deepfm",
)

# 当前 Trainer 不会自动创建保存目录，所以这里先手动创建
os.makedirs("./saved/quick_start_deepfm", exist_ok=True)
trainer.fit(train_dl, val_dl)
# evaluate 建议传 trainer.model，确保使用的是验证后保存的最优权重
auc = trainer.evaluate(trainer.model, test_dl)
print(f"测试集 AUC: {auc:.4f}")
```

如果你想直接复用完整脚本，也可以运行：

```bash
cd examples/ranking
python run_criteo.py --model_name deepfm --epoch 2 --device cuda:0
```

---

## 示例 2：召回模型（DSSM）

这是一条完整的双塔召回链路，使用仓库内置的 `MovieLens-1M` 样本数据。

```python
import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

torch.manual_seed(2022)

# ========== 1. 加载样本数据 ==========
data_path = "examples/matching/data/ml-1m/ml-1m_sample.csv"
data = pd.read_csv(data_path)
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
print(f"数据集大小: {len(data)}")

# ========== 2. 编码离散特征 ==========
# MovieLens 这里全部按稀疏离散特征处理，后续统一走 embedding
user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]

# +1 是给 padding / OOV 位置预留 0
feature_max_idx = {}
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat]) + 1
    feature_max_idx[feat] = data[feat].max() + 1

# user_profile / item_profile 会在生成模型输入时和序列样本拼接
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_profile = data[user_cols].drop_duplicates("user_id")
item_profile = data[item_cols].drop_duplicates("movie_id")

# ========== 3. 构建序列样本 ==========
# point-wise 召回训练：正负样本分别带 0/1 标签
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,
    neg_ratio=3,
    min_item=0,
)

# gen_model_input 会把 profile 特征、目标物品、历史序列整理成模型可直接消费的字典
x_train = gen_model_input(
    df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50
)
y_train = x_train.pop("label")
x_test = gen_model_input(
    df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50
)

# ========== 4. 定义特征 ==========
# DSSM 的用户侧通常由画像特征 + 历史行为聚合向量组成
user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
# 这里用 mean pooling 把历史电影序列压成一个固定向量
user_features += [
    SequenceFeature(
        "hist_movie_id",
        vocab_size=feature_max_idx["movie_id"],
        embed_dim=16,
        pooling="mean",
        shared_with="movie_id",
    )
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in item_cols
]

# ========== 5. DataLoader ==========
# 召回场景会同时返回训练、用户推理、物品推理三套 DataLoader
all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,  # Windows / Notebook 环境下多进程 worker 容易出问题
)

# ========== 6. 定义模型 ==========
# DSSM 是最基础的双塔召回模型：user tower / item tower 映射到同一向量空间
model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# ========== 7. 训练与导出向量 ==========
# MatchTrainer 会负责召回场景下的训练和 embedding 导出
trainer = MatchTrainer(
    model,
    mode=0,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/quick_start_dssm",
)

# 先创建模型保存目录，后续 inference_embedding 也会复用这里的权重
os.makedirs("./saved/quick_start_dssm", exist_ok=True)
trainer.fit(train_dl)

# 召回评估通常不是直接看分类分数，而是先导出 user / item embedding
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path="./saved/quick_start_dssm",
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path="./saved/quick_start_dssm",
)

print(f"用户向量形状: {user_embedding.shape}")
print(f"物品向量形状: {item_embedding.shape}")
```

如果你更希望直接跑示例脚本，也可以执行：

```bash
cd examples/matching
python run_ml_dssm.py --epoch 2 --device cuda:0
```

如果你在 Windows / Notebook 环境下运行脚本时遇到 DataLoader 多进程问题，更推荐直接执行上面的 Python 代码块，因为当前脚本没有暴露 `num_workers` 参数，而召回场景通常需要显式设置 `num_workers=0`。

---

## 运行提示

- GPU 训练：将 `device="cpu"` 改成 `device="cuda:0"`
- 运行目录：文档里的路径默认以**仓库根目录**为基准
- 保存目录：当前 Trainer 不会自动创建 `model_path`，请先执行一次 `os.makedirs(path, exist_ok=True)`
- Windows 环境：召回场景推荐显式设置 `num_workers=0`

---

## 下一步

- 查看 [排序模型教程](../tutorials/ctr.md)，学习 `WideDeep / DeepFM / DCN / DIN`
- 查看 [召回模型教程](../tutorials/retrieval.md)，学习 `DSSM / YoutubeDNN / MIND`
- 查看 [多任务教程](../tutorials/pipeline.md)，学习 `MMOE / PLE / ESMM`
- 查看 [模型部署](../serving/intro.md)，了解 ONNX 导出和向量索引

---

## 常见问题

### Q: 如何保存和加载模型？

```python
import torch

torch.save(model.state_dict(), "model.pth")
model.load_state_dict(torch.load("model.pth"))
```

### Q: 如何导出 ONNX 模型？

```python
trainer.export_onnx("model.onnx")

# 双塔模型可分别导出
trainer.export_onnx("user_tower.onnx", mode="user")
trainer.export_onnx("item_tower.onnx", mode="item")
```

### Q: 我只想快速确认环境没问题，有没有最小命令？

```bash
cd examples/ranking
python run_criteo.py --model_name deepfm --epoch 1 --device cuda:0
```
