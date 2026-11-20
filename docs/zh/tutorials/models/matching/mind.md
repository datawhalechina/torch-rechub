---
title: MIND 使用示例
description: 用 MIND 为每个用户生成多个兴趣向量，并合并各个向量找到的电影
---

# MIND 使用示例

## 1. 模型简介与适用场景

一个人可能喜欢科幻片，也可能常看喜剧。MIND 会从同一个人的观影记录里提取多个兴趣向量，每个向量都能单独到片库里找电影。这些向量是从数据里学出来的，训练时无需给兴趣起名或打标签。

**论文**: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030v1)

### 模型结构

> **注意**: 由于 MIND 内部使用动态路由胶囊网络，torchview 暂时无法自动追踪其计算图，因此未提供架构可视化图。

- **Embedding Layer**: 编码用户属性和历史行为序列
- **Capsule Network (Dynamic Routing)**: 从行为序列中提取多个兴趣向量
- **User Representation**: 多个兴趣向量（而非一个），维度为 `[batch_size, interest_num, embed_dim]`
- **训练方式**: List-wise (Softmax)，与 YoutubeDNN 类似

### List-wise 前向输出

在 `mode=2` 的 list-wise 训练中，`neg_item_feature` 会提供每条样本的负样本集合，`item_tower` 返回候选物品向量：

```text
item_embedding: [batch_size, 1 + n_neg_items, embed_dim]
```

MIND 会先使用正样本物品从多个用户兴趣向量中选择最相关的 `best_interest_emb`：

```text
best_interest_emb: [batch_size, 1, embed_dim]
```

然后对每个候选物品计算内积，输出 sampled softmax 所需的 logits：

```text
y = (best_interest_emb * item_embedding).sum(dim=-1)
y: [batch_size, 1 + n_neg_items]
```

这里必须沿 embedding 维度 `dim=-1` 求和，不能沿候选物品维度求和。`MatchTrainer(mode=2)` 使用 `CrossEntropyLoss`，因此 `y_train = 0` 表示第 0 个候选物品是正样本。

训练时，模型用正样本电影选出一个兴趣向量来打分。为用户推荐时还没有目标电影，所以要分别用各个兴趣向量检索候选。第 5 节给出了合并结果的写法。

---

## 2. 数据准备与预处理

使用 **MovieLens-1M** 数据集，与 DSSM/YoutubeDNN 的数据处理方式一致，采用 `mode=2` (list-wise) 构建训练数据。

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match

data = pd.read_csv("examples/matching/data/ml-1m/ml-1m_sample.csv")
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])

sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip', 'cate_id']
user_col, item_col = "user_id", "movie_id"

feature_max_idx = {}
for feature in sparse_features:
    lbe = LabelEncoder()
    data[feature] = lbe.fit_transform(data[feature]) + 1
    feature_max_idx[feature] = data[feature].max() + 1

user_profile = data[["user_id", "gender", "age", "occupation", "zip"]].drop_duplicates("user_id")
item_profile = data[["movie_id", "cate_id"]].drop_duplicates("movie_id")

# mode=2: list-wise 训练
df_train, df_test = generate_seq_feature_match(
    data, user_col, item_col, time_col="timestamp",
    item_attribute_cols=[], sample_method=1, mode=2, neg_ratio=3, min_item=0
)

x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50, padding="post", truncating="post")
y_train = np.array([0] * df_train.shape[0])
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50, padding="post", truncating="post")
```

### 定义特征

```python
user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']

user_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16)
    for name in user_cols
]

# 历史行为序列
history_features = [
    SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"],
                    embed_dim=16, pooling="concat", shared_with="movie_id")
]

# 正样本物品特征
item_features = [
    SparseFeature("movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16)
]

# 负样本物品特征
neg_item_feature = [
    SequenceFeature("neg_items", vocab_size=feature_max_idx["movie_id"],
                    embed_dim=16, pooling="concat", shared_with="movie_id")
]

all_item = df_to_dict(item_profile)
test_user = x_test

dg = MatchDataGenerator(x=x_train, y=y_train)
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=2048, num_workers=0)
```

---

## 3. 模型配置与参数说明

### 3.1 创建模型

```python
from torch_rechub.models.matching import MIND

model = MIND(
    user_features=user_features,
    history_features=history_features,
    item_features=item_features,
    neg_item_feature=neg_item_feature,
    max_length=50,          # 最大序列长度
    temperature=0.02,       # 仅批内交叉熵使用
    interest_num=4           # 兴趣向量数量
)
```

### 3.2 参数详解

| 参数 | 类型 | 说明 | 建议值 |
|------|------|------|--------|
| `user_features` | `list[Feature]` | 用户侧特征 | 用户属性 |
| `history_features` | `list[Feature]` | 用户历史行为序列（`pooling="concat"`） | |
| `item_features` | `list[Feature]` | 正样本物品特征 | |
| `neg_item_feature` | `list[Feature]` | 负样本物品特征 | |
| `max_length` | `int` | 最大序列长度 | 50 |
| `temperature` | `float` | 批内交叉熵使用的温度。本页 `mode=2` 的前向计算直接使用内积 | 批内采样时使用 |
| `interest_num` | `int` | 提取的兴趣向量数量 | 4 ~ 8 |

本例设 `interest_num=4`。一位用户会得到 4 个向量，每个向量的长度与电影向量一致。

---

## 4. 训练过程与代码示例

```python
import os
from torch_rechub.trainers import MatchTrainer

torch.manual_seed(2022)
save_dir = "./saved/mind/"
os.makedirs(save_dir, exist_ok=True)

trainer = MatchTrainer(
    model,
    mode=2,                          # list-wise
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=10,
    device="cpu",
    model_path=save_dir
)

trainer.fit(train_dl)
```

---

## 5. 模型评估与结果分析

```python
# 生成向量
# MIND 的用户侧不是单个向量，而是多个兴趣向量
user_embedding = trainer.inference_embedding(
    model=model, mode="user", data_loader=test_dl, model_path=save_dir
)
item_embedding = trainer.inference_embedding(
    model=model, mode="item", data_loader=item_dl, model_path=save_dir
)

# MIND 的 user_embedding shape: [n_users, interest_num, embed_dim]
print(f"User Embedding shape: {user_embedding.shape}")
print(f"Item Embedding shape: {item_embedding.shape}")
```

> **注意**: MIND 的 User Embedding 是 3D 的 `[n_users, interest_num, embed_dim]`，在向量检索时需要对每个兴趣向量分别检索，然后合并去重。

### 向量检索

对每个有效兴趣向量分别检索，按 Annoy 距离从小到大合并。同一电影只保留最近的位置。下面用第一个用户的留出电影检查 Hit@10。

```python
from torch_rechub.utils.match import Annoy

index = Annoy(n_trees=10)
index.fit(item_embedding)
candidates = []
for interest in user_embedding[0]:
    if interest.norm().item() > 0:
        rows, distances = index.query(interest, n=10)
        candidates.extend(zip(rows, distances))

candidates.sort(key=lambda pair: pair[1])
rows = list(dict.fromkeys(row for row, _ in candidates))[:10]
movie_ids = all_item["movie_id"][rows]
print("Hit@10", int(test_user["movie_id"][0] in movie_ids))
```

仓库脚本对全部用户评估，并用同版词表还原原始物品 ID。

## 6. 参数调优建议

调大 `interest_num` 时，每位用户需要查询的向量也会增多。可先保留 4 个，用同一份留出的电影记录比较 Hit@10 和每位用户的查询时间，再决定是否增加。若改历史长度，数据处理的 `seq_max_len` 和模型的 `max_length` 要一起改。

## 7. 常见问题与解决方案

### 为什么同一部电影会被找到两次？

两个兴趣向量可能查到同一部电影。第 5 节先按距离合并结果，再为每部电影保留距离最近的一次。这样一部电影只会占用一个推荐位。

---

## 8. 模型可视化限制

项目的可视化工具需要先安装 `pip install "torch-rechub[visualization]"` 和系统 Graphviz。不过，当前 MIND 的动态路由循环不能被 `torchview` 稳定追踪，因此不要直接对 MIND 调用 `visualize_model()`；本页只保留前文的结构与张量形状说明。

---

## 9. ONNX 导出

先安装 ONNX 可选依赖：

```bash
pip install "torch-rechub[onnx]"
```

动态路由会按示例输入的序列长度参与追踪，导出后务必用真实输入验证输出。最小导出方式如下：

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("mind_user_tower.onnx", mode="user")
exporter.export("mind_item_tower.onnx", mode="item")
```

---

## 运行仓库示例

从仓库根目录执行。脚本会保存映射和权重，并评估召回列表。

```bash
cd examples/matching
uv run --extra annoy python run_ml_mind.py --epoch 1 --device cpu
```
