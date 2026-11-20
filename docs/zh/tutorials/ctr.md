---
title: 排序模型教程
description: Torch-RecHub 排序模型教程，覆盖 WideDeep、DeepFM、DCN 以及序列排序模型入口
---

# 排序模型教程

Criteo 样本每行是一条广告曝光记录，标签表示用户有没有点击。下面先用 DeepFM 为每条记录预测点击概率。第二个例子换成 Amazon 商品数据，加入用户看过的商品，演示 DIN 和 BST 如何给当前候选商品打分。

## 一、用 Criteo 曝光记录预测点击

### 1. 数据准备与特征处理

先在同一个 Python 会话中运行[准备 Criteo 输入](/zh/guide/quick_start#criteo-input)，保留两组特征 `dense_feas`、`sparse_feas` 和训练、验证、测试三个 DataLoader。对比多个模型时可以复用这三个 DataLoader。每次创建模型前重新构造特征对象，避免带入上一个模型训练过的嵌入参数。

### 2. WideDeep / DeepFM / DCN 的统一训练方式

```python
import os
from torch_rechub.models.ranking import WideDeep, DeepFM, DCN
from torch_rechub.trainers import CTRTrainer

# 任选一种模型
# 下面先训练 DeepFM。对比其他模型时，用后面对应的构造代码替换它。
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

### 3. 如何比较这三个模型？

保持训练集、验证集和测试集不变，每次只运行一个模型的构造代码。WideDeep 将线性层和多层网络的分数相加，DeepFM 多算一项两两特征的交互，DCN 则用交叉层学习特征组合。三者都可以交给同一个 `CTRTrainer` 接口训练。

## 二、用 DIN 或 BST 加入用户的浏览历史

这份样本带有交互时间。预测用户会不会点击某件商品时，只能把它之前的记录作为历史。例如，历史里的商品和类目要按同一顺序排列，才能知道每件商品属于哪个类目。

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

# features 只放用户画像/上下文；target_features 与 history_features 一一对应
features = [
    SparseFeature("user_id", vocab_size=n_users + 1, embed_dim=8),
]
target_features = [
    SparseFeature("target_item_id", vocab_size=n_items + 1, embed_dim=8, padding_idx=0),
    SparseFeature("target_cate_id", vocab_size=n_cates + 1, embed_dim=8, padding_idx=0),
]

history_features = [
    # 序列模型这里必须保留完整序列张量，所以用 concat 而不是 mean / sum
    SequenceFeature("hist_item_id", vocab_size=n_items + 1, embed_dim=8, pooling="concat", shared_with="target_item_id", padding_idx=0),
    SequenceFeature("hist_cate_id", vocab_size=n_cates + 1, embed_dim=8, pooling="concat", shared_with="target_cate_id", padding_idx=0),
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

### 2. DIN / BST 的创建方式

```python
import os
from torch_rechub.models.ranking import DIN, BST
from torch_rechub.trainers import CTRTrainer

model = DIN(
    features=features,
    history_features=history_features,
    target_features=target_features,
    mlp_params={"dims": [256, 128]},
    attention_mlp_params={"dims": [256, 128]},
)

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
- `features` 是用户画像/上下文特征，不要再把 `target_features` 整体放进去，否则会重复输入；BST 还会因历史与目标维度不等而报错。
- `BST` 的 `embed_dim` 必须能被 `nhead` 整除。

DIEN 还需要逐时刻负样本序列 `neg_history_features`，且 `CTRTrainer` 要设置 `loss_mode=False`，不能只替换上面的模型类。请直接参考 [DIEN 教程](/zh/tutorials/models/ranking/dien)。

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



### 如何用 NVIDIA 显卡训练？

在已装好 CUDA 版 PyTorch 的环境中，把创建 `CTRTrainer` 时的 `device` 改成 `"cuda:0"`。

### 权重会保存到哪里？

`CTRTrainer` 把权重写入 `model_path` 指定的目录。示例训练前用 `os.makedirs` 建好这个目录。

### 各模型的配置示例

- [DeepFM 教程](/zh/tutorials/models/ranking/deepfm)
- [WideDeep 教程](/zh/tutorials/models/ranking/widedeep)
- [DCN 教程](/zh/tutorials/models/ranking/dcn)
- [DIN 教程](/zh/tutorials/models/ranking/din)
- [DIEN 教程](/zh/tutorials/models/ranking/dien)
- [BST 教程](/zh/tutorials/models/ranking/bst)
