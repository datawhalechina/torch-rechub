---
title: MIND 使用示例
description: Multi-Interest Network with Dynamic Routing (MIND) 模型完整使用教程 —— 多兴趣召回模型
---

# MIND 使用示例

## 1. 模型简介与适用场景

MIND (Multi-Interest Network with Dynamic Routing) 是阿里妈妈在 CIKM'2019 提出的多兴趣召回模型。不同于 DSSM 为用户生成**单一向量**的做法，MIND 使用**动态路由的胶囊网络 (Capsule Network)** 从用户行为序列中提取**多个兴趣向量**，更好地表示用户兴趣的多样性。

**论文**: [Multi-Interest Network with Dynamic Routing for Recommendation at Tmall](https://arxiv.org/pdf/1904.08030v1)

### 模型结构

> **注意**: 由于 MIND 内部使用动态路由胶囊网络，torchview 暂时无法自动追踪其计算图，因此未提供架构可视化图。

- **Embedding Layer**: 编码用户属性和历史行为序列
- **Capsule Network (Dynamic Routing)**: 从行为序列中提取多个兴趣向量
- **User Representation**: 多个兴趣向量（而非一个），维度为 `[batch_size, interest_num, embed_dim]`
- **训练方式**: List-wise (Softmax)，与 YoutubeDNN 类似

### 适用场景

- 推荐系统**召回阶段**
- 用户兴趣具有**多样性**的场景（如掏宝用户同时对手机、服装、食品感兴趣）
- 大规模候选集的 ANN 检索

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

x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
y_train = np.array([0] * df_train.shape[0])
x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)
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
train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=2048)
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
    temperature=0.02,       # 温度系数
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
| `temperature` | `float` | Softmax 温度系数 | 0.02 |
| `interest_num` | `int` | 提取的兴趣向量数量 | 4 ~ 8 |

> **interest_num** 是 MIND 最重要的超参数，决定了用多少个向量表示一个用户。通常 4~8 之间效果最好。

---

## 4. 训练过程与代码示例

```python
from torch_rechub.trainers import MatchTrainer

torch.manual_seed(2022)

trainer = MatchTrainer(
    model,
    mode=2,                          # list-wise
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=10,
    device="cpu",
    model_path="./saved/mind/"
)

trainer.fit(train_dl)
```

---

## 5. 模型评估与结果分析

```python
# 生成向量
user_embedding = trainer.inference_embedding(
    model=model, mode="user", data_loader=test_dl, model_path="./saved/mind/"
)
item_embedding = trainer.inference_embedding(
    model=model, mode="item", data_loader=item_dl, model_path="./saved/mind/"
)

# MIND 的 user_embedding shape: [n_users, interest_num, embed_dim]
print(f"User Embedding shape: {user_embedding.shape}")
print(f"Item Embedding shape: {item_embedding.shape}")
```

> **注意**: MIND 的 User Embedding 是 3D 的 `[n_users, interest_num, embed_dim]`，在向量检索时需要对每个兴趣向量分别检索，然后合并去重。

### 向量检索

```python
from torch_rechub.utils.match import Annoy

# 为每个兴趣向量分别检索，合并结果
annoy = Annoy(n_trees=10)
annoy.fit(item_embedding)

# 对每个用户的每个兴趣向量检索
for i in range(min(3, len(user_embedding))):
    all_indices = set()
    for k in range(user_embedding.shape[1]):  # interest_num
        indices, _ = annoy.query(user_embedding[i, k], n=10)
        all_indices.update(indices)
    print(f"User {i} -> Total unique items: {len(all_indices)}")
```

---

## 6. 参数调优建议

1. **interest_num**: 关键超参数。值越大，能捕捉越多样的兴趣，但检索成本也成倍增加
2. **max_length**: 序列越长，胶囊网络能捕获的信息越丰富，但计算量增大
3. **温度系数**: 对于 MIND，`temperature=0.02` 是推荐值

---

## 7. 常见问题与解决方案

### Q1: MIND 和 DSSM 在线上部署的区别？
DSSM 每个用户只有 1 个向量，MIND 有 `interest_num` 个向量。线上需要对每个兴趣向量分别查 ANN，然后合并 Top-K 结果。

### Q2: interest_num 设多大比较好？
取决于业务场景中用户兴趣的多样性程度。电商通常 4~8，新闻/视频因兴趣更分散可以 8~16。

---

## 8. 模型可视化

```python
from torch_rechub.utils.visualization import visualize_model
visualize_model(model, save_path="mind_architecture.png", dpi=300)
```

---

## 9. ONNX 导出

```python
from torch_rechub.utils.onnx_export import ONNXExporter
exporter = ONNXExporter(model, device="cpu")
exporter.export("mind_user_tower.onnx", mode="user")
exporter.export("mind_item_tower.onnx", mode="item")
```

---

## 完整代码

```python
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from torch_rechub.basic.features import SparseFeature, SequenceFeature
from torch_rechub.models.matching import MIND
from torch_rechub.trainers import MatchTrainer
from torch_rechub.utils.data import MatchDataGenerator, df_to_dict
from torch_rechub.utils.match import gen_model_input, generate_seq_feature_match, Annoy


def main():
    torch.manual_seed(2022)

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

    df_train, df_test = generate_seq_feature_match(
        data, user_col, item_col, time_col="timestamp",
        item_attribute_cols=[], sample_method=1, mode=2, neg_ratio=3, min_item=0
    )
    x_train = gen_model_input(df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50)
    y_train = np.array([0] * df_train.shape[0])
    x_test = gen_model_input(df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50)

    user_cols = ['user_id', 'gender', 'age', 'occupation', 'zip']
    user_features = [SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16) for name in user_cols]
    history_features = [SequenceFeature("hist_movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16, pooling="concat", shared_with="movie_id")]
    item_features = [SparseFeature("movie_id", vocab_size=feature_max_idx["movie_id"], embed_dim=16)]
    neg_item_feature = [SequenceFeature("neg_items", vocab_size=feature_max_idx["movie_id"], embed_dim=16, pooling="concat", shared_with="movie_id")]

    all_item = df_to_dict(item_profile)
    test_user = x_test

    dg = MatchDataGenerator(x=x_train, y=y_train)
    train_dl, test_dl, item_dl = dg.generate_dataloader(test_user, all_item, batch_size=2048)

    model = MIND(user_features, history_features, item_features, neg_item_feature,
                 max_length=50, temperature=0.02, interest_num=4)

    trainer = MatchTrainer(model, mode=2, optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
                           n_epoch=10, device="cpu", model_path="./saved/mind/")
    trainer.fit(train_dl)

    user_embedding = trainer.inference_embedding(model=model, mode="user", data_loader=test_dl, model_path="./saved/mind/")
    item_embedding = trainer.inference_embedding(model=model, mode="item", data_loader=item_dl, model_path="./saved/mind/")
    print(f"User Embedding: {user_embedding.shape}, Item Embedding: {item_embedding.shape}")

    # 向量召回
    annoy = Annoy(n_trees=10)
    annoy.fit(item_embedding)
    for i in range(min(3, len(user_embedding))):
        all_indices = set()
        for k in range(user_embedding.shape[1]):
            indices, _ = annoy.query(user_embedding[i, k], n=10)
            all_indices.update(indices)
        print(f"User {i} -> Total unique items: {len(all_indices)}")


if __name__ == "__main__":
    main()
```
