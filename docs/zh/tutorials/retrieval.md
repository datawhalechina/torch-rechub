---
title: 召回模型教程
description: Torch-RecHub 召回模型教程，覆盖 DSSM、GRU4Rec、MIND 与向量检索流程
---

# 召回模型教程

给定一个用户看过的电影，怎样从电影库里再找出 10 部候选？下面用 MovieLens 样本训练 DSSM。训练完分别计算用户和电影的向量，再检查推荐列表里有没有用户最后一次交互的那部电影。从仓库根目录运行即可。

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

### 2. 编码 ID，整理用户表和电影表

这里把样本中的电影当作一份已知片单。`user_profile` 每位用户一行，`item_profile` 每部电影一行。训练使用编码后的 ID，最后展示结果时还要靠 `raw_movie_ids` 查回原始电影编号。换成业务日志时，词表和片单应使用当时已经可用的版本。

```python
feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    # +1 给 padding 预留 0，和框架里 embedding 的常见约定保持一致
    data[feature] = lbe.fit_transform(data[feature]) + 1
    if feature == item_col:
        raw_movie_ids = lbe.classes_
    feature_max_idx[feature] = data[feature].max() + 1

# profile 表记录静态画像，后面会和序列特征拼接成模型输入
user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates("user_id")
item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")
```

### 3. 生成序列样本

默认设置下，每条正交互会配 3 条负样本，标签分别是 1 和 0。要试批内负采样，只需把下一段的 `in_batch_neg` 改成 `True`，然后从这里向下重新运行。构造样本时会只保留正交互，训练器从同一批的其他电影里取负例。

```python
in_batch_neg = False
df_train, df_test = generate_seq_feature_match(
    data,
    user_col=user_col,
    item_col=item_col,
    time_col="timestamp",
    item_attribute_cols=[],
    sample_method=1,
    mode=0,
    in_batch_neg=in_batch_neg,
    neg_ratio=3,
    min_item=0,
)

# gen_model_input 会把用户画像、目标物品、历史序列等字段拼成模型输入字典
x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = x_train.pop("label")
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
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
                    pooling="mean", shared_with="movie_id", padding_idx=0)
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16, padding_idx=0)
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

# 两座塔都输出 64 维向量
model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# Trainer 不会自动建目录，所以这里先创建
save_dir = "./saved/dssm_inbatch" if in_batch_neg else "./saved/dssm"
os.makedirs(save_dir, exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=0,
    in_batch_neg=in_batch_neg,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # GPU 改成 "cuda:0"
    model_path=save_dir,
)

trainer.fit(train_dl)
```

例如一批有 3 对正交互，3 个用户分别和 3 部电影打分，就得到一个 3 × 3 的矩阵。第一个用户对应第一部电影，第二个用户对应第二部，第三个用户对应第三部，因此正例都在对角线上。使用这种方式每批至少需要两对正交互。

### 3. 导出向量

`test_dl` 用每个用户留出的那条记录构造用户输入，`item_dl` 遍历整份电影表。下面打印的两个形状，第一维分别是用户记录数和电影数，第二维都应为 64，即两座塔的最后一层宽度。

```python
user_embedding = trainer.inference_embedding(
    model=model,
    mode="user",
    data_loader=test_dl,
    model_path=save_dir,
)
item_embedding = trainer.inference_embedding(
    model=model,
    mode="item",
    data_loader=item_dl,
    model_path=save_dir,
)

print(user_embedding.shape, item_embedding.shape)
```

## 三、查出 10 部电影，检查是否命中

先安装 `annoy` 可选依赖。Annoy 返回的是电影在向量表里的行号。先用行号取出 `movie_id` 的编码，再按编码查回原始电影 ID。若测试用户最后看过的电影出现在前 10 部里，下面打印的 `Hit@10` 就是 1，否则是 0。这里检查的是一个用户。

```python
from torch_rechub.utils.match import Annoy

index = Annoy(n_trees=10)
index.fit(item_embedding)
rows, distances = index.query(user_embedding[0], n=10)
recommended = raw_movie_ids[all_item[item_col][rows] - 1]
held_out = raw_movie_ids[test_user[item_col][0] - 1]
print("推荐电影", recommended.tolist())
print("留出的电影", held_out, "Hit@10", int(held_out in recommended))
```

## 四、显式候选列表

YoutubeDNN、GRU4Rec 和 MIND 的 list-wise 示例为每行准备一个正物品和一组负物品，打分形状为 `[B, 1 + K]`。正物品放第一列，所以交叉熵的位置标签全是 0。以下命令各自构建样本、训练和评估，可从仓库根目录选择一条运行。

```bash
cd examples/matching
uv run --extra annoy python run_ml_youtube_dnn.py --epoch 1 --device cpu
uv run --extra annoy python run_ml_gru4rec.py --epoch 1 --device cpu
uv run --extra annoy python run_ml_mind.py --epoch 1 --device cpu
```

## 五、常见问题

### Q1：为什么 DSSM 用 `pooling="mean"`，GRU4Rec / MIND 用 `pooling="concat"`？

- DSSM 直接把历史行为聚合成一个固定向量，因此可以使用 `mean`
- GRU4Rec / MIND 需要完整序列张量，必须使用 `concat`

### Q2：为什么训练前要先创建 `./saved/...`？

当前 `MatchTrainer` 不会自动创建 `model_path`，所以示例中需要先执行一次 `os.makedirs(path, exist_ok=True)`。

### Q3：为什么 Windows 下建议 `num_workers=0`？

召回场景会同时创建训练、用户推理、物品推理三个 DataLoader。Windows / Notebook 环境下多进程 worker 更容易触发权限或句柄问题，文档示例默认显式设置为 0。

### 各模型的训练和检索示例

- [DSSM 教程](/zh/tutorials/models/matching/dssm)
- [YoutubeDNN 教程](/zh/tutorials/models/matching/youtube_dnn)
- [MIND 教程](/zh/tutorials/models/matching/mind)
