---
title: 生成式推荐模型
description: Torch-RecHub 生成式推荐模型详细介绍
---

# 生成式推荐模型

本页的“生成式推荐”指用序列模型预测或生成下一个 item token / 语义 ID。当前实现面向 next-item recommendation，并不生成推荐理由、商品文案或自然语言对话。

## 1. HSTUModel

### 功能描述

HSTU（Hierarchical Sequential Transduction Units）是面向 next-item prediction 的自回归序列推荐模型。Torch-RecHub 中的 `HSTUModel` 接收 padding 后的 item token 序列，以及可选的逐位置时间差特征，并在每个序列位置输出 item 词表上的 logits。

### 核心原理

- **Eq. 2 UVQK 投影**：对联合 `UVQK` 投影先整体做一次 `SiLU`，再 split，因此 `U`、`V`、`Q`、`K` 都经过同一个非线性。
- **Eq. 3 注意力偏置**：将 per-head 的桶化相对位置/时间偏置 `rab^{p,t}` 加到 attention scores，再做 `silu(scores) / max_seq_len`。
- **Eq. 4 门控输出**：使用 `LayerNorm(A V) * U` 后接一个输出线性层，不再使用 concat-u/x 旁路，也没有额外 FFN。
- **外部残差**：`HSTUBlock` 中每层按 `x = x + HSTULayer(x)` 包裹。
- **生成式训练**：按 next-token 目标训练，并在 loss 中忽略 PAD token `0`。

### 使用方法

```python
import torch

from torch_rechub.models.generative import HSTUModel

model = HSTUModel(
    vocab_size=100000,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=200,
    num_time_buckets=128,
    time_bucket_unit="seconds",
)

seq_tokens = torch.randint(1, 100000, (32, 200))
time_diffs = torch.zeros_like(seq_tokens)  # 相对查询时间的秒级时间差
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # torch.Size([32, 200, 100000])
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

### 适用场景

- 大规模序列推荐
- 长序列建模
- next-item prediction

## 2. HLLMModel

### 功能描述

`HLLMModel` 是一个轻量化的序列推荐实现：先在模型外用 LLM 生成 item embedding，然后冻结这张表，只训练用户序列 Transformer。它不会在 `forward` 中加载 BERT/LLM，也不接收 `user_features` / `item_features` 特征列表。

### 核心原理

- **离线 item 语义表**：`item_embeddings[token_id]` 必须与训练数据的 token ID 对齐，row 0 为 PAD
- **用户序列模型**：位置编码、可选时间桶 embedding 和相对位置 bias
- **余弦打分**：Transformer 输出与冻结 item embedding 归一化后做矩阵乘，返回全词表 logits

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
| item_embeddings | Tensor or str | `[vocab_size, d_model]` 的预计算 embedding，或 `torch.load` 可读的文件路径 | 必填 |
| vocab_size | int | 包含 PAD=0 的词表大小，必须等于 embedding 行数 | 必填 |
| d_model | int | Transformer 维度，必须等于 embedding 列数 | 512 |
| n_heads / n_layers | int | Transformer 头数 / 层数 | 8 / 4 |
| max_seq_len | int | 最大序列长度 | 256 |
| use_rel_pos_bias | bool | 是否使用相对位置 bias | True |
| use_time_embedding | bool | 是否使用时间差 embedding | True |
| temperature | float | 余弦 logits 温度 | 0.07 |

### 适用场景

- 已能离线生成且按 token ID 对齐 item embedding 的场景
- next-item 序列预测
- 希望固定 item 语义空间、只训练用户序列塔的场景

## 3. RQVAEModel

### 功能描述

RQ-VAE 将每个 item 的连续 embedding 经过多级残差向量量化，转换为离散 codebook 索引。这些索引再格式化为 `<a_12><b_7><c_91>` 一类语义 ID，是 TIGER 的 item token 来源。

### 最小训练与语义 ID 链路

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

### 功能描述

TIGER（Transformer Index for GEnerative Recommenders）把推荐建模成"生成下一个 item 的语义 ID"的序列到序列任务。每个 item 先由 RQ-VAE 量化成一串 codebook token（语义 ID，如 `<a_1><b_3><c_5>`），TIGER 基于 T5 自回归地生成下一个 item 的语义 ID，再通过前缀受限的 beam search 约束到合法 item 上。`TIGERModel` 继承自 `transformers` 的 `T5ForConditionalGeneration`。

### 核心原理

- **语义 ID**：用 RQ-VAE 对 item embedding 做多级残差量化，得到每个 item 的 codebook token 序列，相似 item 共享前缀，天然带有层次结构。
- **序列到序列**：输入是用户历史 item 的语义 ID 拼接，标签是下一个 item 的语义 ID，按 T5 的 teacher-forcing 交叉熵训练。
- **新增 token**：训练前把所有语义 ID token 加入 tokenizer 并调用 `resize_token_embeddings`，否则 `<a_1>` 这类 token 会被 T5 切成子词。
- **受限生成**：推理时用 `Trie` 构建 `prefix_allowed_tokens_fn`，保证 beam search 只生成语义 ID 表中合法的 item。

### 使用方法

完整工作流（生成 toy 数据 / 训练 / 测试，以及真实数据的 RQ-VAE → TIGER 流水线）见 [TIGER 复现说明](/zh/blog/tiger_reproduction) 与示例脚本 `examples/generative/run_tiger_movielens.py`、`run_tiger_amazon_books.py`。模型最小用法：

```python
from transformers import T5Config, T5Tokenizer

from torch_rechub.models.generative.tiger import TIGERModel

tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.add_tokens(["<a_1>", "<b_3>", "<c_5>"])  # 语义 ID token

config = T5Config.from_pretrained("t5-small")
config.vocab_size = len(tokenizer)
model = TIGERModel(config)
model.set_hyper(temperature=1.0)
model.resize_token_embeddings(len(tokenizer))
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| config | T5Config | T5 配置，其中 `vocab_size` 需为加入语义 ID token 之后的词表大小 | 必填 |
| temperature | float | 通过 `set_hyper` 设置，用于缩放 logits 的温度 | 1.0 |

### 适用场景

- 基于语义 ID 的生成式检索
- item 数量极大、需要压缩 item 表示的场景
- 希望用多段离散 token 表示 item 的序列推荐实验

## 5. RPGModel

### 功能描述

RPG（Recommendation with Parallel Generation）沿用 TIGER 的语义 ID 思路，但去掉了 digit 之间的顺序。每个 item 由 OPQ 量化成 `n_digit` 个**无序**码字，任何一位都不是前一位的细化，因此所有位可以在一次前向中并行预测。省掉自回归解码后，语义 ID 才得以从 4 位扩展到 32 甚至 64 位，效果提升正来自于此。

### 核心原理

- **无序语义 ID**：`OPQTokenizer` 先对 item embedding 做白化 PCA，再用 FAISS `OPQ{m},IVF1,PQ{m}x{bits}` 量化。第 `j` 位量化自己那块子空间，各位相互独立而非残差关系。
- **一个 item 占一个位置**：item 表示是它 `n_digit` 个 token embedding 的均值池化，所以长度为 `L` 的历史只占 `L` 个位置，而不是 `L * n_digit`。
- **多 token 预测**：`n_digit` 个独立的 `ResBlock` 头（初始化时是恒等映射）各产生一个 query，与自己那一位的 codebook 做余弦相似度并除以 `temperature`，每一位算一个交叉熵后取平均。
- **图约束解码**：item 相似度由学到的 codebook 导出，每个 item 只保留 `n_edges` 个最近邻。检索从随机 beam 出发，在这张图上走 `propagation_steps` 轮，因此只需给一小部分候选打分。全量打分依然可用，验证阶段就走这条路径。

### 使用方法

完整流程（预处理 / 分词 / 训练 / 测试）见示例脚本 `examples/generative/run_rpg_amazon_2014.py`。模型最小用法：

```python
import numpy as np

from torch_rechub.models.generative import RPGModel
from torch_rechub.utils.opq import OPQTokenizer

item_embeddings = np.load("item_embeddings.npy")  # (n_items - 1, dim)
tokenizer = OPQTokenizer(n_codebook=32, codebook_size=256, pca_dim=128)
tokenizer.fit(item_embeddings)

model = RPGModel(tokenizer.item_tokens(), codebook_size=256, temperature=0.03)
states, loss = model(input_ids, attention_mask, labels)

model.build_decoding_graph(n_edges=200)
preds = model.generate(input_ids, attention_mask, seq_lens, topk=10, use_graph=True)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| item_tokens | LongTensor | `(n_items, n_digit)` token 表，来自 `OPQTokenizer.item_tokens()`，第 0 行为 PAD | 必填 |
| codebook_size | int | 每一位的码字数 | 256 |
| n_embd | int | 隐藏维度 | 448 |
| n_layer | int | Transformer 层数 | 2 |
| n_head | int | 注意力头数 | 4 |
| n_inner | int | 前馈层维度 | 1024 |
| max_seq_len | int | 单条序列最多包含的 item 数 | 50 |
| embd_pdrop / attn_pdrop | float | embedding / attention dropout；论文靠大 dropout 正则化这个小 backbone | 0.5 |
| temperature | float | 用于缩放余弦 logits | 0.07 |

### 适用场景

- item 规模极大、TIGER 的逐位 beam search 成为推理瓶颈的场景
- 需要长语义 ID（32~64 位）而非常见 4 位的场景
- 对延迟或显存敏感的检索场景，可借助图约束解码

## 6. 模型比较

| 模型 | 复杂度 | 表达能力 | 计算效率 | 适用场景 |
| --- | --- | --- | --- | --- |
| HSTUModel | 中 | 高 | 中 | next-item 预测、长序列建模 |
| HLLMModel | 中 | 中 | 中 | 使用预计算 LLM item embedding 的序列推荐 |
| RQVAEModel | 中 | 中 | 中 | 将连续 item embedding 量化为 TIGER 语义 ID |
| TIGERModel | 高 | 高 | 中 | 基于语义 ID 的生成式检索、超大 item 空间 |
| RPGModel | 中 | 高 | 高 | 长语义 ID、低延迟生成式检索 |

## 7. 使用建议

1. 直接对 item token 做 next-item 预测时，使用 HSTUModel。
2. 已有按 token ID 对齐的 item embedding，且希望冻结 item 语义空间时，使用 HLLMModel。
3. 使用 TIGER 前，先用 RQVAEModel 生成语义 ID；量化和生成两个阶段必须复用同一份 item 行号映射。
4. HSTU/HLLM 的输出是 `[batch, seq_len, vocab_size]`，词表较大时需先估算 logits 的显存占用。
5. 关注解码延迟时优先选择 RPGModel：所有 digit 在一次前向中并行预测，语义 ID 变长不会增加解码步数。

## 8. 代码示例：完整的生成式推荐模型训练流程

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

## 9. 当前实现边界

- `SeqTrainer` 训练 HSTU/HLLM，并报告 loss 与 token-level top-1 accuracy；它没有内置 Recall@K、NDCG@K、BLEU 或 ROUGE 评估。
- RQ-VAE 与 TIGER 是两阶段流程：先离线量化 item embedding，再训练 TIGER。项目不会自动生成原始 item embedding。
- RPG 同样是两阶段流程：先离线用 `OPQTokenizer` 拟合 item embedding，再用得到的 token 训练模型。其中 OPQ 步骤依赖 `faiss`，且 `RPGTrainer` 报告的是 Recall@K 与 NDCG@K，而非 token-level accuracy。
- 当前训练器可选用单机 `DataParallel`，但没有 DDP、多机分布式、流水线并行或自动混合精度训练流程。
- 本模块没有内置生产服务、TensorRT、边缘部署、自然语言内容生成或多模态推荐能力。需要这些能力时，应在项目外自行实现并验证。
