---
title: 排序模型教程
description: Torch-RecHub 排序模型教程，覆盖 WideDeep、DeepFM、DCN 以及序列排序模型入口
---

# 排序模型教程

本教程聚焦排序场景的共性训练流程：数据准备、特征定义、训练器使用、评估与常见扩展。文中的基础示例使用仓库内置的 `Criteo` 样本数据；序列模型部分使用 `Amazon Electronics` 样本数据。

## 一、基础排序链路（Criteo）

### 1. 数据准备与特征处理

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.utils.data import DataGenerator

df = pd.read_csv("examples/ranking/data/criteo/criteo_sample.csv")

# 排序类基线模型一般都遵循“连续特征 + 类别特征”这套输入范式
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# 缺失值处理方式尽量和仓库 examples 保持一致，避免复现差异
df[sparse_features] = df[sparse_features].fillna("-996")
df[dense_features] = df[dense_features].fillna(0)

# 连续特征归一化，类别特征编码成离散 id
scaler = MinMaxScaler()
df[dense_features] = scaler.fit_transform(df[dense_features])

for feat in sparse_features:
    encoder = LabelEncoder()
    df[feat] = encoder.fit_transform(df[feat].astype(str))

# 这些 Feature 对象描述“这一列该怎么喂给模型”
dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [SparseFeature(name, vocab_size=df[name].nunique(), embed_dim=16) for name in sparse_features]

x = df.drop(columns=["label"])
y = df["label"]

# DataGenerator 会自动按 split_ratio 划分训练 / 验证 / 测试集
dg = DataGenerator(x, y)
train_dl, val_dl, test_dl = dg.generate_dataloader(split_ratio=[0.7, 0.1], batch_size=256)
```

### 2. WideDeep / DeepFM / DCN 的统一训练方式

```python
import os
from torch_rechub.models.ranking import WideDeep, DeepFM, DCN
from torch_rechub.trainers import CTRTrainer

# 任选一种模型
# DeepFM 适合做第一条排序链路；注释掉的 WideDeep / DCN 只是切换模型，不需要改数据流
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

# model = WideDeep(
#     wide_features=sparse_feas,
#     deep_features=sparse_feas + dense_feas,
#     mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
# )

# model = DCN(
#     features=dense_feas + sparse_feas,
#     n_cross_layers=3,
#     mlp_params={"dims": [256, 128]},
# )

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/ctr_basic",
)

# 训练前先创建保存目录，避免 fit 结束后保存最优权重时报错
os.makedirs("./saved/ctr_basic", exist_ok=True)
trainer.fit(train_dl, val_dl)
# evaluate 传 trainer.model，拿到的是当前 trainer 持有的最优模型
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 3. 这一页适合什么模型？

- `WideDeep`：快速验证 wide + deep 结构
- `DeepFM`：类别特征交互 + MLP 的经典基线
- `DCN / DCNv2`：显式特征交叉

这些模型都使用同一套 `DenseFeature + SparseFeature + DataGenerator + CTRTrainer` 训练范式。

## 二、序列排序链路（DIN / DIEN / BST）

序列模型和基础排序模型最大的区别在于：需要额外生成历史行为序列，并且 `history_features` 与 `target_features` 要严格对应。

### 1. 使用 Amazon Electronics 样本数据构建序列

```python
import pandas as pd

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import DataGenerator, df_to_dict, generate_seq_feature

data = pd.read_csv("examples/ranking/data/amazon-electronics/amazon_electronics_sample.csv")

# generate_seq_feature 会按时间排序，为每条样本生成历史物品 / 历史类目序列
train, val, test = generate_seq_feature(
    data=data,
    user_col="user_id",
    item_col="item_id",
    time_col="time",
    item_attribute_cols=["cate_id"],
)

n_users = data["user_id"].max()
n_items = data["item_id"].max()
n_cates = data["cate_id"].max()

# target_features 和 history_features 后面要一一对应做注意力计算
features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8),
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8),
]
target_features = features

history_features = [
    # 序列模型这里必须保留完整序列张量，所以用 concat 而不是 mean / sum
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat", shared_with="target_item_id"),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat", shared_with="target_cate_id"),
]

# 先把 DataFrame 转成 dict，再交给 DataGenerator，和 examples/ranking 的写法一致
train_dict, val_dict, test_dict = df_to_dict(train), df_to_dict(val), df_to_dict(test)
train_y = train_dict.pop("label")
val_y = val_dict.pop("label")
test_y = test_dict.pop("label")

dg = DataGenerator(train_dict, train_y)
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=val_dict,
    y_val=val_y,
    x_test=test_dict,
    y_test=test_y,
    batch_size=4096,
)
```

### 2. DIN / DIEN / BST 的创建方式

```python
import os
from torch_rechub.models.ranking import DIN, DIEN, BST
from torch_rechub.trainers import CTRTrainer

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    attention_mlp_params={"dims": [256, 128]},
)

# model = DIEN(
#     features=features,
#     history_features=history_features,
#     target_features=target_features,
#     mlp_params={"dims": [256, 128]},
#     attention_mlp_params={"dims": [256, 128]},
# )

# model = BST(
#     features=features,
#     history_features=history_features,
#     target_features=target_features,
#     mlp_params={"dims": [256, 128]},
#     nhead=8,
#     dropout=0.2,
#     num_layers=1,
# )

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-3},
    n_epoch=2,
    earlystop_patience=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/ctr_sequence",
)

# 同样先创建保存目录，避免训练结束保存权重失败
os.makedirs("./saved/ctr_sequence", exist_ok=True)
trainer.fit(train_dl, val_dl)
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 3. 序列模型的关键约束

- `SequenceFeature` 必须使用 `pooling="concat"`，因为 DIN / DIEN / BST 需要拿到完整序列张量。
- `history_features` 和 `target_features` 必须一一对应，并通过 `shared_with` 共享 embedding。
- `BST` 的 `embed_dim` 必须能被 `nhead` 整除。

## 三、评估、导出与可视化

### 1. 评估

```python
auc = trainer.evaluate(trainer.model, test_dl)
print(f"Test AUC: {auc:.4f}")
```

### 2. ONNX 导出

```python
trainer.export_onnx("model.onnx", dynamic_batch=True)
```

### 3. 结构可视化

```python
from torch_rechub.utils.visualization import visualize_model

visualize_model(model, save_path="model_architecture.png", dpi=300)
```

> 可视化功能需要额外安装：`pip install "torch-rechub[visualization]"`

## 四、常见问题

### Q1：为什么这页不直接覆盖所有排序模型的完整代码？

因为排序模型可以分成两类：

- 基础排序：`WideDeep / DeepFM / DCN`
- 序列排序：`DIN / DIEN / BST`

它们的数据准备方式不同。这一页保留共性流程，具体参数和调优建议放在各模型页面中展开。

### Q2：如何切换到 GPU？

将 `device="cpu"` 改成 `device="cuda:0"` 即可。

### Q3：为什么示例里先执行 `os.makedirs`？

当前 `CTRTrainer` 会直接把权重保存到 `model_path`，不会自动创建目录。为了保证示例可以直接运行，建议在训练前先创建保存目录。

### Q4：还想继续看更细的模型说明？

- [DeepFM 教程](/zh/tutorials/models/ranking/deepfm)
- [WideDeep 教程](/zh/tutorials/models/ranking/widedeep)
- [DCN 教程](/zh/tutorials/models/ranking/dcn)
- [DIN 教程](/zh/tutorials/models/ranking/din)
- [DIEN 教程](/zh/tutorials/models/ranking/dien)
- [BST 教程](/zh/tutorials/models/ranking/bst)
