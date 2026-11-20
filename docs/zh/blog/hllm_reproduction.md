---
title: HLLM 模型复现说明
description: torch-rechub 中轻量 HLLM 实现的数据准备、训练命令、损失语义与实现边界
---

# HLLM 模型在 torch-rechub 中的复现说明

MovieLens 里既有观看顺序，也有电影名称和类型。HLLM 把这两份信息分两步用起来：先让语言模型为每部电影算一个文本向量，保存到磁盘。训练时按观看顺序查出这些向量，交给用户侧的 Transformer，预测接下来会看哪部电影。

这样只需在生成物品向量时运行语言模型，换一组用户侧参数也能复用同一份向量。这里的单卡示例固定物品向量，只训练用户侧。论文的端到端方案还会联合训练物品和用户两侧的模型。下面先跑 MovieLens，使用图书数据时再换到 Amazon Books 一节。

## 1. 代码入口

- 模型：`torch_rechub/models/generative/hllm.py`
  - `HLLMTransformerBlock`
  - `HLLMModel`
- 通用序列训练器：`torch_rechub/trainers/seq_trainer.py`
- MovieLens-1M：
  - `examples/generative/data/ml-1m/preprocess_ml_hstu.py`
  - `examples/generative/data/ml-1m/preprocess_hllm_data.py`
  - `examples/generative/run_hllm_movielens.py`
- Amazon Books：
  - `examples/generative/data/amazon-books/preprocess_amazon_books.py`
  - `examples/generative/data/amazon-books/preprocess_amazon_books_hllm.py`
  - `examples/generative/run_hllm_amazon_books.py`

`examples/generative/data/*/processed/` 是运行预处理后生成的目录，仓库不提交其中的 `pkl` 或 embedding 文件。

## 2. 从物品文本到下一次观看

### 2.1 为每个物品保存一行向量

预处理器支持两种 `--model_type`：

| 参数 | HuggingFace 模型 | 隐藏维度 |
| --- | --- | ---: |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 |
| `baichuan2` | `baichuan-inc/Baichuan2-7B-Chat` | 4096 |

文本格式为：

```text
Compress the following sentence into embedding: title: {title}genres: {genres}
```

Amazon Books 的输入文本取自 `title` 和 `description`。脚本取最后一个 token 的隐藏状态，把向量写到 `item_embeddings_<model_type>.pt`。第 0 行留给补位，其余行号由训练序列使用的词表决定。

例如，假设原始电影 ID 42 在 `vocab.pkl` 中查到的 `item_to_idx[42]` 等于 17。第 17 行就应该来自电影 42 的文本。交换两部电影的行号，矩阵形状依然正确，模型却会给用户读错电影。所以要把生成向量时的词表和向量文件一起保存。日志还会打印多少个物品有文本，缺少文本的行会保留为零向量。

### 2.2 用观看历史训练用户编码器

前向流程为：

```text
seq_tokens [B, L]
  -> 冻结 item embedding lookup
  + 可学习绝对位置 embedding
  + 可选时间 bucket embedding
  -> causal Transformer blocks
  -> L2 normalize hidden states
  -> hidden @ normalized_item_embeddings.T / 0.07
  -> logits [B, L, V]
```

每个 block 使用 pre-norm 多头自注意力、前馈网络和残差连接。causal mask 与 padding mask 分别屏蔽未来 token 和 PAD key，并将 PAD 位置输出清零。

### 2.3 一段历史怎样产生训练目标

训练时，历史中的每个有效位置都用来预测下一个物品。例如：

```text
logits[:, i, :]  -> seq_tokens[:, i + 1]
logits[:, -1, :] -> held-out targets
```

训练器跳过补位标签 `0`。脚本默认的 `--loss_type nce` 仍会计算全词表得分，再对目标类别算负对数似然。得分已在模型内按温度缩放，所以脚本给损失函数传入的温度是 1。需要减少训练时的候选数时，选择 `--loss_type sampled_softmax`，用 `--num_negatives` 指定每个目标采多少个负例。`cross_entropy` 也使用全词表。

## 3. 安装与模型缓存

从仓库根目录安装：

```bash
pip install -e ".[generative]"
```

`generative` extra 提供 `transformers` 与 `accelerate`。若所选模型的 tokenizer 报缺少 SentencePiece（Baichuan2 环境较常见），还需执行 `pip install sentencepiece`；该包当前未包含在 extra 中。

MovieLens 的 HLLM 预处理脚本会先以 `local_files_only=True` 检查 HuggingFace 缓存；目标 LLM 未缓存时会直接退出。请先在可联网环境下载相应模型，或预先把缓存复制到运行环境。`--no_download` 只控制数据集文件，不会让未缓存的 LLM 自动变为可用。

## 4. MovieLens-1M 复现命令

以下命令均从仓库根目录运行：

```bash
# 1. 生成序列数据；缺少 ratings.dat/movies.dat/users.dat 时自动下载
python examples/generative/data/ml-1m/preprocess_ml_hstu.py

# 2. 生成文本映射与 token-id 对齐的 item embeddings
python examples/generative/data/ml-1m/preprocess_hllm_data.py \
    --model_type tinyllama \
    --device cuda

# 3. 单卡训练。每批 8 条、序列最多 32 个物品，每个目标采 32 个负物品
mkdir -p outputs/hllm_ml
python examples/generative/run_hllm_movielens.py --model_type tinyllama --device cuda --epoch 1 --batch_size 8 --max_seq_len 32 --d_model 128 --loss_type sampled_softmax --num_negatives 32 --accumulation_steps 2 --amp --save_dir outputs/hllm_ml
```

在 RTX 5080（16 GB）上，用 MovieLens-1M 的用户 1–32、每人最近最多 24 条交互验证了上述训练参数。子集有 578 部电影，按物品 ID 取对应的 2048 维 TinyLlama 向量。训练和评估的 PyTorch 峰值张量显存约 38 MiB。此数值限于小样本的用户编码器阶段，生成文本向量需要另外测量。

默认数据目录是 `examples/generative/data/ml-1m/processed/`，应包含：

```text
vocab.pkl
train_data.pkl
val_data.pkl
test_data.pkl
movie_text_map.pkl
item_embeddings_tinyllama.pt
```

MovieLens 按每个用户的时间顺序留出样本。最后一次观看用来测试，倒数第二次用来验证，更早的观看记录用于训练。

自定义目录时，两个预处理脚本必须使用同一个 `--output_dir`，训练时再通过 `--dataset_path` 指向该目录。训练脚本对显式相对 `--dataset_path` 的处理与脚本目录有关，自动化任务中建议传绝对路径。

## 5. Amazon Books 复现命令

序列数据与 item metadata 必须选择一致的数据源。默认 `bytedance` 会下载 ByteDance 处理后的文件；`raw` 使用 Stanford SNAP 原始文件。

```bash
# 1. 生成序列数据
python examples/generative/data/amazon-books/preprocess_amazon_books.py \
    --data_source bytedance \
    --max_seq_len 200 \
    --min_seq_len 5

# 2. 生成文本映射与 item embeddings
python examples/generative/data/amazon-books/preprocess_amazon_books_hllm.py \
    --data_source bytedance \
    --model_type tinyllama \
    --device cuda

# 3. 训练与评估
python examples/generative/run_hllm_amazon_books.py \
    --data_dir examples/generative/data/amazon-books/processed \
    --model_type tinyllama \
    --batch_size 64 \
    --epochs 5 \
    --learning_rate 1e-3 \
    --n_layers 2 \
    --dropout 0.1 \
    --max_seq_len 200 \
    --loss_type nce \
    --device cuda
```

Amazon 预处理输出为：

```text
vocab.pkl
train_data.pkl
val_data.pkl
test_data.pkl
item_text_map.pkl
item_embeddings_tinyllama.pt
```

重新运行预处理会重写输出目录中的映射、split 和 embedding 文件。`--overwrite` 只描述下载文件的覆盖行为；要保留已有实验产物，请先备份或换一个 `--output_dir`。

## 6. 训练结束后看哪些数字

日志里的 loss 衡量历史中各有效位置的预测误差，Top-1 准确率看的是每条验证或测试记录留出的那个目标。脚本还会计算 HR 和 NDCG（K 取 10、50、200）。前者看目标物品是否被找回，后者还看它排第几。比较两次实验时，候选片库和过滤已看物品的设置要一致。

训练中显存紧张时，先看词表和用户侧模型各占多少。全词表打分需要一个 `[批大小, 序列长度, 词表大小]` 的张量，采样训练能减少最后一维的开销。`--d_model 128` 则用一个可训练的投影，把缓存的宽向量映射进更小的用户编码器。评估时 `SeqTrainer.predict()` 可以分块检索全库。

## 7. 与论文实验怎样比较

这套脚本适合回答“同一份冻结的文本向量，能否帮助预测下一次交互”。若要比较 ByteDance 官方 HLLM 的指标，还需要对齐两侧模型的架构和联合训练方式、数据切分、负采样和评估候选集。仓库中的小样本成绩只对应上面列出的配置。
