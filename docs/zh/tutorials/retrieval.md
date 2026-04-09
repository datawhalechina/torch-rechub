---
title: 召回模型教程
description: Torch-RecHub 召回模型教程，覆盖 DSSM、GRU4Rec、MIND 与向量检索流程
---

# 召回模型教程

本教程聚焦召回场景的通用流程：构建用户/物品特征、生成序列样本、训练双塔或序列召回模型、导出向量并做 ANN 检索。示例默认在**仓库根目录**执行，使用 `MovieLens-1M` 样本数据。

## 一、数据准备

### 1. 加载样本数据

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

data = pd.read_csv("examples/matching/data/ml-1m/ml-1m_sample.csv")
# 这里简单取 genres 的第一个类目，目的是构造一个最小可运行示例
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])

user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]
```

### 2. 编码特征并构建 profile

```python
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    # +1 给 padding 预留 0，和框架里 embedding 的常见约定保持一致
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

# profile 表记录静态画像，后面会和序列特征拼接成模型输入
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates("user_id")
item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")
```

### 3. 生成序列样本

#### DSSM（point-wise）

```python
# point-wise：每条 user-item 样本独立打 0/1 标签，适合 DSSM 这类基础双塔
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

# gen_model_input 会把用户画像、目标物品、历史序列等字段拼成模型输入字典
x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train.pop("label")
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
```

#### GRU4Rec / MIND（list-wise）

```python
# list-wise：一个正样本配多个负样本，适合序列召回模型
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=2,
    neg_ratio=3,
    min_item=0,
)

x_train = gen_model_input(
    df_train,
    user_profile,
    user_col,
    item_profile,
    item_col,
    seq_max_len=50,
    padding="post",
    truncating="post",
)
y_train = [0] * len(df_train)  # list-wise 下标签恒为 0，表示列表第一个位置是正样本
x_test = gen_model_input(
    df_test,
    user_profile,
    user_col,
    item_profile,
    item_col,
    seq_max_len=50,
    padding="post",
    truncating="post",
)
```

## 二、DSSM：基础双塔召回

### 1. 定义特征

```python
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]
user_features += [
    # 历史序列在 DSSM 里只需要被压成一个固定向量，因此可以用 mean pooling
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="mean", shared_with="movie_id")
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in item_cols
]

all_item = df_to_dict(item_profile)
test_user = x_test
```

### 2. 训练模型

```python
import os
import torch

from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

torch.manual_seed(2022)

# MatchDataGenerator 会同时生成训练、用户推理、物品推理三路 DataLoader
dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,  # Windows / Notebook 下这里显式设 0 更稳
)

# DSSM 是最标准的双塔召回基线，建议先跑通它再看更复杂的序列召回
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# Trainer 不会自动建目录，所以这里先创建
os.makedirs("./saved/dssm", exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=0,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/dssm",
)

trainer.fit(train_dl)
```

### 3. 导出向量

```python
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path="./saved/dssm",
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path="./saved/dssm",
)

print(user_embedding.shape, item_embedding.shape)
```

## 三、序列召回：GRU4Rec / MIND

这两类模型和 DSSM 的主要区别是：

- 训练样本需要 `mode=2` 的 list-wise 数据
- 历史行为和负样本都使用 `SequenceFeature(..., pooling="concat")`

### 1. 定义序列特征

```python
user_cols = ["user_id", "gender", "age", "occupation", "zip"]

user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
history_features = [
    # 这里必须保留完整序列，交给 GRU4Rec / MIND 在模型内部处理
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="concat", shared_with="movie_id")
]
item_features = [SparseFeature("movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16)]
neg_item_feature = [
    # list-wise 训练里的负样本序列同样要保留完整张量
    SequenceFeature("neg_items", vocab_size=feature_max_idx["movie_id"], embed_dim=16,
                    pooling="concat", shared_with="movie_id")
]
```

### 2. GRU4Rec

```python
from torch_rechub.models.matching import GRU4Rec

# GRU4Rec 用 RNN 对历史兴趣做时序建模，结构上比 DSSM 更适合顺序敏感场景
model = GRU4Rec(
    user_features,
    history_features,
    item_features,
    neg_item_feature,
    user_params={"dims": [128, 64, 16]},
    temperature=0.02,
)
```

### 3. MIND

```python
from torch_rechub.models.matching import MIND

# MIND 会把一个用户拆成多个兴趣 capsule，适合兴趣分散的场景
model = MIND(
    user_features,
    history_features,
    item_features,
    neg_item_feature,
    max_length=50,
    temperature=0.02,
)
```

### 4. 训练方式

```python
import os

# 序列召回训练时仍然建议显式设置 num_workers=0，尤其是 Windows 环境
dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(
    test_user,
    all_item,
    batch_size=256,
    num_workers=0,
)

os.makedirs("./saved/matching_sequence", exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=2,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path="./saved/matching_sequence",
)

trainer.fit(train_dl)
```

## 四、向量检索

Torch-RecHub 内置了 `Annoy / Faiss / Milvus` 三种向量检索封装。最小示例：

```python
from torch_rechub.utils.match import Annoy

annoy = Annoy(n_trees=10)
annoy.fit(item_embedding)
similar_items, scores = annoy.query(user_embedding[0], topk=10)
print(similar_items)
```

> 如果缺少依赖，请先安装对应包，例如 `pip install annoy`。

## 五、常见问题

### Q1：为什么 DSSM 用 `pooling="mean"`，GRU4Rec / MIND 用 `pooling="concat"`？

- DSSM 直接把历史行为聚合成一个固定向量，因此可以使用 `mean`
- GRU4Rec / MIND 需要完整序列张量，必须使用 `concat`

### Q2：为什么训练前要先创建 `./saved/...`？

当前 `MatchTrainer` 不会自动创建 `model_path`，所以示例中需要先执行一次 `os.makedirs(path, exist_ok=True)`。

### Q3：为什么 Windows 下建议 `num_workers=0`？

召回场景会同时创建训练、用户推理、物品推理三个 DataLoader。Windows / Notebook 环境下多进程 worker 更容易触发权限或句柄问题，文档示例默认显式设置为 0。

### Q4：还想看具体模型页？

- [DSSM 教程](/zh/tutorials/models/matching/dssm)
- [YoutubeDNN 教程](/zh/tutorials/models/matching/youtube_dnn)
- [MIND 教程](/zh/tutorials/models/matching/mind)
