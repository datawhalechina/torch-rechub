---
title: 生成式推荐模型
description: Torch-RecHub 生成式推荐模型详细介绍
---

# 生成式推荐模型

给定一个用户看过的电影，下面的模型都围绕“下一部可能是哪部”来训练。HSTU 和 HLLM 给候选电影打分。TIGER 则逐段生成下一部电影的编码，这串编码叫语义 ID，可以先用 RQ-VAE 从电影向量中学出来。

## 1. HSTUModel

HSTU 读取按时间排列的物品 ID。训练时，前两部电影用来预测第三部，前三部用来预测第四部，以此类推。不同长度的记录用 `0` 补齐，损失会忽略这些补位。

如果有交互时间，可以把每个位置的时间差一并传给模型。注意力得分会加上由相对位置和时间差查出的偏置，本实现的各个头共享这组偏置。核心层与论文公式的对应见 [HSTU 复现说明](/zh/blog/hstu_reproduction)。

### 使用方法

```python
import torch

from torch_rechub.models.generative import HSTUModel

model = HSTUModel(
    vocab_size=128,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=16,
    num_time_buckets=128,
    time_bucket_unit="seconds",
)

seq_tokens = torch.randint(1, 128, (2, 16))
time_diffs = torch.zeros_like(seq_tokens)  # 相对查询时间的秒级时间差
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # torch.Size([2, 16, 128])
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| vocab_size | int | item 词表大小，`0` 保留为 PAD | 必填 |
| d_model | int | 隐藏维度 | 512 |
| n_heads | int | 注意力头数 | 8 |
| n_layers | int | 堆叠 HSTU 层数 | 4 |
| dqk | int | 每个 head 的 Query/Key 维度 | 64 |
| dv | int | 每个 head 的 Value/U 维度 | 64 |
| max_seq_len | int | 最大支持序列长度 | 256 |
| dropout | float | Dropout 比例 | 0.1 |
| use_time_embedding | bool | 是否加入输入侧时间桶 embedding；`time_diffs` 仍会用于 `rab^{p,t}` | True |
| num_time_buckets | int | 时间 embedding 和 attention bias 使用的桶数 | 128 |
| time_bucket_fn | {"sqrt", "log"} | 时间差桶化函数 | "sqrt" |
| time_bucket_divisor | float | 桶化后再除以该值，用于调节 bucket 范围 | 1.0 |
| tie_embeddings | bool | 输出投影是否与 token embedding 共享权重 | True |

## 2. HLLMModel

HLLM 也读用户的观看顺序，但每部电影的初始向量来自片名、类型等文本。先在训练前算好这些向量，再用冻结的向量表训练用户侧的 Transformer。

词表把某部电影编码成 42 时，`item_embeddings[42]` 就必须存这部电影的向量。训练后，模型用用户历史的表示和候选电影表示计算余弦相似度。

### 使用方法

```python
from torch_rechub.models.generative import HLLMModel

import torch

# 必须是 [vocab_size, d_model]，且第 i 行对应 token i
item_embeddings = torch.load("item_embeddings_tinyllama.pt", map_location="cpu")
model = HLLMModel(
    item_embeddings=item_embeddings,
    vocab_size=item_embeddings.shape[0],
    d_model=item_embeddings.shape[1],
    n_heads=8,
    n_layers=2,
    max_seq_len=50,
    dropout=0.1,
    temperature=0.07,
)

seq_tokens = torch.randint(1, item_embeddings.shape[0], (32, 50))
time_diffs = torch.zeros_like(seq_tokens)  # 单位：秒
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # [32, 50, vocab_size]
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| item_embeddings | Tensor or str | 按物品 ID 排列的二维向量表，或 `torch.load` 可读的文件路径 | 必填 |
| vocab_size | int | 包含 PAD=0 的词表大小，必须等于 embedding 行数 | 必填 |
| d_model | int | 用户编码器宽度。与原始物品向量宽度不同时需设置 `project_items=True` | 512 |
| n_heads / n_layers | int | Transformer 头数 / 层数 | 8 / 4 |
| max_seq_len | int | 最大序列长度 | 256 |
| use_rel_pos_bias | bool | 是否使用相对位置 bias | True |
| use_time_embedding | bool | 是否使用时间差 embedding | True |
| temperature | float | 余弦 logits 温度 | 0.07 |

## 3. RQVAEModel

RQ-VAE 接收已有的物品向量，为每个物品找出一组离散编号。第一级码本选出最接近的向量，下一级再编码剩下的误差。这些编号可以写成 `<a_12><b_7><c_91>`，作为 TIGER 训练时的语义 ID。

### 用 RQ-VAE 为每个物品生成语义 ID

```python
import os
import torch
from torch.utils.data import DataLoader

from torch_rechub.models.generative import RQVAEModel
from torch_rechub.trainers.rqvae_trainer import Trainer
from torch_rechub.utils.data import EmbDataset

dataset = EmbDataset("examples/generative/data/amazon-books/processed/item_embeddings_tinyllama.pt")
model = RQVAEModel(
    in_dim=dataset.dim,
    num_emb_list=[256, 256, 256],
    e_dim=32,
    layers=[512, 256, 128, 64],
    sk_epsilons=[0.0, 0.0, 0.0],
)

os.makedirs("saved/rqvae", exist_ok=True)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
trainer = Trainer(model, n_epoch=50, device="cpu", model_path="saved/rqvae", eval_step=5)
trainer.fit(train_loader)

# 生成 semantic ID 时必须保持 dataset 原始顺序，否则字典中的 item index 会错位
semantic_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
state = torch.load("saved/rqvae/model_best_collision_rate.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
semantic_ids = model.generate_semantic_ids(
    dataset,
    semantic_loader,
    prefix=["<a_{}>", "<b_{}>", "<c_{}>"],
    use_sk=True,
    device="cpu",
)
```

`semantic_ids` 的 key 是 `EmbDataset` 中的行号，因此 embedding 文件的行顺序必须与后续 TIGER 数据中的 item ID 映射一致。不要在生成阶段使用 `shuffle=True`。

### 主要参数

| 参数 | 说明 |
| --- | --- |
| in_dim | 原始 item embedding 维度 |
| num_emb_list | 每一级残差量化器的 codebook 大小，列表长度就是 semantic ID 段数 |
| e_dim | 量化空间维度 |
| layers | Encoder 隐藏层，Decoder 按逆序对称构建 |
| loss_type | `mse` 或 `l1` 重建损失 |
| sk_epsilons / sk_iters | Sinkhorn 分配参数，列表长度需与 codebook 级数一致 |

## 4. TIGERModel

TIGER（Transformer Index for GEnerative Recommenders）把用户看过的电影转换成语义 ID，送给 T5，再逐个预测下一部电影的 ID 片段。

假设片库中以 `<a_1>` 开头的 ID 只接 `<b_3>` 或 `<b_7>`，生成第一段后就只允许从这两段中选。示例把所有电影的语义 ID 放进一棵前缀树，在每步解码时查出可以接在后面的 token。

### 使用方法

下面用三个语义 ID 片段构造一个小型 T5，看看模型需要怎样的词表和配置。要训练和评估这个模型，可以顺着 [TIGER 复现说明](/zh/blog/tiger_reproduction) 先跑合成数据，再换成 MovieLens 或 Amazon Books。

```python
from transformers import T5Config
from torch_rechub.models.generative.tiger import TIGERModel

tokenizer = TIGERModel.build_tokenizer(["<a_1>", "<b_3>", "<c_5>"])
config = T5Config(vocab_size=len(tokenizer), d_model=128, d_ff=512, num_layers=2,
                  num_heads=4, pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id)
model = TIGERModel(config)
model.set_hyper(temperature=1.0)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| config | T5Config | T5 配置，其中 `vocab_size` 需为加入语义 ID token 之后的词表大小 | 必填 |
| temperature | float | 通过 `set_hyper` 设置，用于缩放 logits 的温度 | 1.0 |

## 5. 模型比较

| 手头的数据 | 训练目标 | 得到什么 |
| --- | --- | --- |
| 用户交互序列，可附带时间戳 | HSTU 根据历史预测下一个物品 | 候选物品的分数 |
| 交互序列和按物品 ID 排好的文本向量 | HLLM 学习怎样利用这些向量预测下一步 | 用户历史与候选物品的匹配分数 |
| 物品向量 | RQ-VAE 学习重建这些向量 | 物品 ID 到语义 ID 的对照表 |
| 交互序列和语义 ID 对照表 | TIGER 预测下一件物品的语义 ID | 可以在对照表里找到的候选编码 |

## 用 SeqTrainer 训练 HSTU

```python
import os
import pickle
import torch

from torch_rechub.models.generative import HSTUModel
from torch_rechub.trainers import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator

with open("examples/generative/data/ml-1m/processed/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("examples/generative/data/ml-1m/processed/val_data.pkl", "rb") as f:
    val_data = pickle.load(f)
with open("examples/generative/data/ml-1m/processed/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)
with open("examples/generative/data/ml-1m/processed/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

train_gen = SequenceDataGenerator(
    train_data["seq_tokens"],
    train_data["seq_positions"],
    train_data["targets"],
    train_data["seq_time_diffs"],
)
val_gen = SequenceDataGenerator(
    val_data["seq_tokens"],
    val_data["seq_positions"],
    val_data["targets"],
    val_data["seq_time_diffs"],
)
test_gen = SequenceDataGenerator(
    test_data["seq_tokens"],
    test_data["seq_positions"],
    test_data["targets"],
    test_data["seq_time_diffs"],
)

train_dl = train_gen.generate_dataloader(batch_size=512, num_workers=0)[0]
val_dl = val_gen.generate_dataloader(batch_size=512, num_workers=0)[0]
test_dl = test_gen.generate_dataloader(batch_size=512, num_workers=0)[0]

item_to_idx = vocab["item_to_idx"] if "item_to_idx" in vocab else vocab
vocab_size = max(item_to_idx.values()) + 1  # token 0 为 PAD，不能只用 len(...)
model = HSTUModel(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=200,
    dropout=0.1,
    time_bucket_unit="seconds",
)

os.makedirs("saved/hstu", exist_ok=True)
trainer = SeqTrainer(
    model,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=10,
    earlystop_patience=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_path="saved/hstu",
)

trainer.fit(train_dl, val_dl)
test_loss, top1_acc = trainer.evaluate(test_dl)
print(f"test_loss={test_loss:.4f}, top1_acc={top1_acc:.4f}")
```

## 显存与评估

HSTU 和 HLLM 对全词表打分时，分数张量的形状是 `[批大小, 序列长度, 词表大小]`。如果词表占用太多显存，单卡训练可以用 `loss_type="sampled_softmax"`，例如 `loss_params={"num_negatives": 32}` 表示每个目标采 32 个负物品。CUDA 上还可以开启 `amp=True`，或用 `accumulation_steps` 分几批累积梯度再更新。

`SeqTrainer.evaluate()` 返回损失和留出目标的 Top-1 准确率。要看前十个推荐里有哪些物品，可以用 `predict(data_loader, top_k=10)` 分块检索。数据准备、单卡参数以及脚本打印的 Hit@K 和 NDCG@K 见 [HSTU](/zh/blog/hstu_reproduction)、[HLLM](/zh/blog/hllm_reproduction) 和 [TIGER](/zh/blog/tiger_reproduction) 的运行说明。
