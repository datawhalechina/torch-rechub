---
title: 生成式推荐模型
description: Torch-RecHub 生成式推荐模型详细介绍
---

# 生成式推荐模型

生成式推荐模型是一种利用生成式AI技术（如大语言模型）进行推荐的新兴方法，能够生成个性化的推荐内容，提供更丰富、更自然的推荐体验。Torch-RecHub 提供了多种先进的生成式推荐模型，结合了推荐系统和生成式AI的优势。

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
- 实时推荐场景
- 万亿参数级推荐系统

## 2. HLLMModel

### 功能描述

HLLM（Hybrid Large Language Model）是一种融合了大语言模型（LLM）能力的混合推荐模型，能够结合推荐系统的协同过滤能力和LLM的语义理解能力。

### 核心原理

- **混合架构**：结合了传统推荐模型和大语言模型的优势
- **语义理解**：利用LLM的强大语义理解能力，处理文本信息
- **协同过滤**：保留传统推荐模型的协同过滤能力，利用用户-物品交互数据
- **多模态融合**：支持文本、图像等多种模态的融合

### 使用方法

```python
from torch_rechub.models.generative import HLLMModel

# 创建模型
model = HLLMModel(
    user_features=user_features,
    item_features=item_features,
    llm_params={
        "model_name": "bert-base-uncased",
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1
    },
    fusion_params={
        "fusion_type": "concat",
        "fusion_dims": [512, 256, 128],
        "dropout": 0.2
    }
)
```

### 参数说明

| 参数 | 类型 | 描述 | 默认值 |
| --- | --- | --- | --- |
| user_features | list | 用户特征列表 | None |
| item_features | list | 物品特征列表 | None |
| llm_params | dict | 大语言模型参数 | None |
| fusion_params | dict | 特征融合参数 | None |

### 适用场景

- 融合LLM能力的推荐场景
- 文本信息丰富的推荐场景
- 需要生成式推荐的场景
- 多模态推荐场景

## 3. TIGERModel

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
- 希望相似 item 共享前缀、提升冷启动与泛化的场景

## 4. 模型比较

| 模型 | 复杂度 | 表达能力 | 计算效率 | 适用场景 |
| --- | --- | --- | --- | --- |
| HSTUModel | 高 | 高 | 中 | 大规模序列推荐、长序列建模 |
| HLLMModel | 高 | 高 | 低 | 融合LLM能力、文本信息丰富的场景 |
| TIGERModel | 高 | 高 | 中 | 基于语义 ID 的生成式检索、超大 item 空间 |

## 5. 使用建议

1. **根据业务需求选择模型**：
   - 大规模序列推荐场景推荐使用 HSTUModel
   - 需要融合LLM能力的场景推荐使用 HLLMModel
   - 文本信息丰富的推荐场景推荐使用 HLLMModel

2. **根据计算资源选择模型**：
   - 计算资源有限时推荐使用 HSTUModel
   - 计算资源充足时可以尝试 HLLMModel
   - 考虑使用模型压缩技术，如知识蒸馏、量化等

3. **模型训练建议**：
   - 采用预训练+微调的方式，提高模型效果和训练效率
   - 使用混合精度训练，加速模型训练
   - 采用分布式训练，处理大规模数据

4. **模型部署建议**：
   - 考虑使用模型压缩技术，减少模型大小和推理延迟
   - 采用服务化部署，支持高并发请求
   - 考虑使用边缘计算，将模型部署到边缘设备

## 6. 代码示例：完整的生成式推荐模型训练流程

```python
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

vocab_size = len(vocab["item_to_idx"]) if "item_to_idx" in vocab else len(vocab)
model = HSTUModel(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=200,
    dropout=0.1,
)

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

## 7. 常见问题与解决方案

### Q: 如何处理大规模数据？
A: 可以尝试以下方法：
- 采用分布式训练，利用多GPU或多机器并行训练
- 使用数据采样技术，如负采样、分层采样等
- 采用模型并行或流水线并行，处理超大模型
- 考虑使用混合精度训练，加速训练过程

### Q: 如何提高生成式推荐模型的推理效率？
A: 可以尝试以下方法：
- 使用模型压缩技术，如知识蒸馏、量化、剪枝等
- 采用模型部署优化，如TensorRT、ONNX Runtime等
- 考虑使用边缘计算，将模型部署到边缘设备
- 采用异步推理或批处理，提高并发处理能力

### Q: 如何评估生成式推荐模型的效果？
A: 可以尝试以下方法：
- 传统推荐评估指标：AUC、Precision@K、Recall@K、NDCG@K等
- 生成式评估指标：BLEU、ROUGE、METEOR、Perplexity等
- 人类评估：通过用户调研或A/B测试评估模型效果
- 业务指标：点击率、转化率、用户留存率等

### Q: 如何处理冷启动问题？
A: 可以尝试以下方法：
- 对于新用户，使用基于内容的推荐或流行度推荐
- 对于新物品，利用LLM的语义理解能力，基于物品描述进行推荐
- 使用迁移学习，从其他相关领域迁移知识
- 采用元学习，快速适应新用户或新物品

## 8. 生成式推荐的应用场景

1. **个性化内容生成**：
   - 生成个性化的推荐理由
   - 生成个性化的商品描述
   - 生成个性化的营销文案

2. **多模态推荐**：
   - 结合文本、图像、音频等多种模态
   - 生成多模态的推荐内容
   - 支持跨模态的推荐

3. **交互式推荐**：
   - 支持用户与推荐系统的自然语言交互
   - 基于用户反馈动态调整推荐
   - 生成对话式推荐

4. **场景化推荐**：
   - 基于用户当前场景生成推荐
   - 生成场景化的推荐内容
   - 支持复杂场景的推荐

## 9. 未来发展趋势

1. **大语言模型与推荐系统的深度融合**：
   - 更紧密地结合LLM和推荐系统的优势
   - 开发专门针对推荐场景优化的LLM
   - 利用LLM的上下文理解能力，提供更个性化的推荐

2. **多模态生成式推荐**：
   - 结合多种模态的生成式推荐
   - 支持跨模态的内容生成和推荐
   - 开发更高效的多模态融合方法

3. **实时生成式推荐**：
   - 实现低延迟的生成式推荐
   - 支持实时的用户交互和反馈
   - 开发更高效的推理架构

4. **可控生成式推荐**：
   - 支持用户对推荐结果的控制和调整
   - 实现可解释、可信赖的生成式推荐
   - 开发更安全、更可靠的生成式推荐系统

5. **大规模生成式推荐**：
   - 支持数十亿用户和物品的大规模推荐
   - 开发更高效的模型训练和推理方法
   - 实现分布式、可扩展的生成式推荐系统

生成式推荐是推荐系统的一个重要发展方向，能够提供更丰富、更自然、更个性化的推荐体验。Torch-RecHub 提供了多种先进的生成式推荐模型，方便开发者根据业务需求选择和使用。随着大语言模型和生成式AI技术的不断发展，生成式推荐将在更多场景中得到应用，为用户提供更好的推荐体验。
