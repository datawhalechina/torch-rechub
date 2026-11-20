---
title: 快速开始
description: 用 Criteo 样本训练点击率模型，用 MovieLens 样本训练双塔召回模型
---

# 快速开始

想知道一条广告被点击的概率，先看 DeepFM。想根据一个用户看过的电影再找几部可能喜欢的，直接看 DSSM。两个例子都用仓库里的小样本，可以单独运行。

## 安装

在仓库根目录打开终端，运行 `uv sync`。下文的 Python 代码从这个目录启动，用 `uv run python` 或同一环境的 Jupyter 内核运行。两个例子默认使用 CPU，装 NVIDIA、AMD 或昇腾设备环境可参照[安装指南](/zh/guide/install)。

## 示例 1：CTR 预测（DeepFM）

Criteo 的一行代表一次广告展示。`label=1` 表示用户点击了广告，`label=0` 表示没有点击。DeepFM 读取这一行的数值和类别特征，为这次展示预测点击概率。

### 准备 Criteo 输入 {#criteo-input}

先划分训练、验证和测试集，再用训练集建立类别词表、计算数值的缩放范围。这样测试集中第一次出现的类别会得到编号 `0`，和线上遇到新类别时的处理一致。

```python
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from torch_rechub.basic.features import DenseFeature, SparseFeature
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.utils.data import DataGenerator

# 读取仓库自带的样本
data_path = "examples/ranking/data/criteo/criteo_sample.csv"
data = pd.read_csv(data_path)
print(f"数据集大小: {len(data)}")

# Criteo 的 I1-I13 是连续特征，C1-C26 是离散特征
dense_features = [f"I{i}" for i in range(1, 14)]
sparse_features = [f"C{i}" for i in range(1, 27)]

# 先切分再拟合预处理。样本较小，按 label 分层让三份数据都保留正负例。
train, holdout = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=2022)
val, test = train_test_split(holdout, test_size=2/3, stratify=holdout["label"], random_state=2022)

scaler = MinMaxScaler().fit(train[dense_features].fillna(0))
vocabs = {
    name: {key: i for i, key in enumerate(train[name].dropna().astype(str).unique(), start=1)}
    for name in sparse_features
}

def encode_features(frame):
    dense = scaler.transform(frame[dense_features].fillna(0)).astype("float32")
    x = {name: dense[:, i] for i, name in enumerate(dense_features)}
    for name in sparse_features:
        x[name] = frame[name].astype("string").map(vocabs[name]).fillna(0).to_numpy(dtype="int64")
    return x

dense_feas = [DenseFeature(name) for name in dense_features]
sparse_feas = [
    SparseFeature(name, vocab_size=len(vocabs[name]) + 1, embed_dim=16, padding_idx=0)
    for name in sparse_features
]

# 验证、测试和预测复用训练集的词表与缩放参数，未知类别和缺失类别都映射到 0。
dg = DataGenerator(encode_features(train), train["label"].to_numpy())
train_dl, val_dl, test_dl = dg.generate_dataloader(
    x_val=encode_features(val), y_val=val["label"].to_numpy(),
    x_test=encode_features(test), y_test=test["label"].to_numpy(),
    batch_size=256,
)

```

### 训练和评估 DeepFM

在同一个 Python 会话里继续运行。模型训练两轮，每轮打印验证 AUC，最后一行是用留出的测试集算出的 AUC。

```python
# DeepFM 同时建模低阶特征交互（FM）和高阶非线性关系（MLP）
model = DeepFM(
    deep_features=dense_feas + sparse_feas,
    fm_features=sparse_feas,
    mlp_params={"dims": [256, 128], "dropout": 0.2, "activation": "relu"},
)

trainer = CTRTrainer(
    model,
    optimizer_params={"lr": 1e-3, "weight_decay": 1e-5},
    n_epoch=2,
    device="cpu",  # NVIDIA/ROCm GPU 改成 "cuda:0"；NPU 可按环境改成 "npu:0"
    model_path="./saved/quick_start_deepfm",
)

# 当前 Trainer 不会自动创建保存目录，所以这里先手动创建
os.makedirs("./saved/quick_start_deepfm", exist_ok=True)
trainer.fit(train_dl, val_dl)
# evaluate 需要显式传入待评估的模型
auc = trainer.evaluate(trainer.model, test_dl)
print(f"测试集 AUC: {auc:.4f}")
```

换成有时间戳的曝光日志时，把较晚的记录留给验证和测试。做线上预测时沿用训练得到的 `vocabs` 和 `scaler`。[用同一条记录核对 PyTorch 与 ONNX 输出](/zh/serving/demo)

如果只想看训练输出，可以在仓库根目录打开终端，直接运行自带的脚本。

```bash
cd examples/ranking
uv run python run_criteo.py --model_name deepfm --epoch 2 --device cpu
```

---

## 示例 2：召回模型（DSSM）

DSSM 分别给用户和电影算一个向量。用户塔读取用户资料和观影历史，物品塔读取电影 ID 和类别。训练后可以把所有电影的向量存起来，为用户找相近的电影。

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

# 读取仓库自带的样本
data_path = "examples/matching/data/ml-1m/ml-1m_sample.csv"
data = pd.read_csv(data_path)
data["cate_id"] = data["genres"].apply(lambda x: x.split("|")[0])
print(f"数据集大小: {len(data)}")

# 编码离散特征
# 这个小样本假定目录已知，先为用户画像和物品编索引，再按用户留出最后的交互。
user_col, item_col = "user_id", "movie_id"
sparse_features = ["user_id", "movie_id", "gender", "age", "occupation", "zip", "cate_id"]

# +1 是给 padding / OOV 位置预留 0
feature_max_idx = {}
for feat in sparse_features:
    encoder = LabelEncoder()
    data[feat] = encoder.fit_transform(data[feat]) + 1
    if feat == item_col:
        raw_movie_ids = encoder.classes_  # 编码 i 对应原始电影 raw_movie_ids[i - 1]
    feature_max_idx[feat] = data[feat].max() + 1

# user_profile / item_profile 会在生成模型输入时和序列样本拼接
user_cols = ["user_id", "gender", "age", "occupation", "zip"]
item_cols = ["movie_id", "cate_id"]

user_profile = data[user_cols].drop_duplicates("user_id")
item_profile = data[item_cols].drop_duplicates("movie_id")

# 构建序列样本
# 每条正样本（标签 1）配三条负样本（标签 0）
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

# 按用户和电影 ID 补上各自的属性，把观影历史补到同一长度
x_train = gen_model_input(
    df_train, user_profile, user_col, item_profile, item_col, seq_max_len=50
)
y_train = x_train.pop("label")
x_test = gen_model_input(
    df_test, user_profile, user_col, item_profile, item_col, seq_max_len=50
)

# 定义特征
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
        shared_with="movie_id", padding_idx=0,
    )
]

item_features = [
    SparseFeature(name, vocab_size=feature_max_idx[name], embed_dim=16, padding_idx=0)
    for name in item_cols
]

# DataLoader
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

# 定义模型
# DSSM 是最基础的双塔召回模型：user tower / item tower 映射到同一向量空间
model = DSSM(
    user_features,
    item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64], "activation": "prelu"},
    item_params={"dims": [256, 128, 64], "activation": "prelu"},
)

# 训练与导出向量
trainer = MatchTrainer(
    model,
    mode=0,
    optimizer_params={"lr": 1e-4, "weight_decay": 1e-6},
    n_epoch=2,
    device="cpu",  # NVIDIA/ROCm GPU 改成 "cuda:0"；NPU 可按环境改成 "npu:0"
    model_path="./saved/quick_start_dssm",
)

# 先创建模型保存目录，后续 inference_embedding 也会复用这里的权重
os.makedirs("./saved/quick_start_dssm", exist_ok=True)
trainer.fit(train_dl)

# 按测试用户和电影目录的顺序，分别取出两张向量表
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

最后两行会打印两张向量表的形状。用户表每行对应一个测试用户，电影表每行对应一部候选电影，两者都有 64 列。想看这些向量能否找回用户最后看过的电影，可以在仓库根目录另开终端，运行带 Annoy 检索的脚本。

```bash
cd examples/matching
uv run --extra annoy python run_ml_dssm.py --epoch 1 --device cpu
```

这个脚本把权重写到 `examples/matching/data/ml-1m/saved/model.pth`。上面的 Python 示例则把权重写到 `saved/quick_start_dssm/model.pth`。

---

## 换设备运行

上面的 `device="cpu"` 可以换成第一张 NVIDIA 或 AMD GPU（`"cuda:0"`），配好昇腾环境后也可以用 `"npu:0"`。Windows 或 Notebook 中先用 `num_workers=0`，让当前进程读取数据。

## 下一步

- 查看 [排序模型教程](../tutorials/ctr.md)，学习 `WideDeep / DeepFM / DCN / DIN`
- 查看 [召回模型教程](../tutorials/retrieval.md)，学习 `DSSM / YoutubeDNN / MIND`
- 查看 [多任务教程](../tutorials/pipeline.md)，学习 `MMOE / PLE / ESMM`
- 查看 [模型部署](../serving/intro.md)，了解 ONNX 导出和向量索引

---

## 常见问题

### 训练好的权重在哪里？

两个示例的 `fit` 都会在各自的 `model_path` 下写入 `model.pth`。下次加载时，先用相同的特征定义和参数创建模型，再用 `model.load_state_dict(...)` 恢复权重。原始类别到整数的词表也要保留，同一个编号在两次运行中应指向同一部电影或同一个类别。

### 怎样导出 ONNX？

排序模型可以导出一个文件。DSSM 可以分别导出用户塔和物品塔，以便按不同频率更新两侧向量。[排序模型导出与预测比对](/zh/serving/demo) · [双塔导出](/zh/serving/onnx)
