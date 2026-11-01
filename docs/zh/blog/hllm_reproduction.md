---
title: HLLM 模型复现说明
description: torch-rechub 中轻量 HLLM 实现的数据准备、训练命令、损失语义与实现边界
---

# HLLM 模型在 torch-rechub 中的复现说明

本文说明仓库当前 HLLM（Hierarchical Large Language Model for Recommendation）示例的可运行路径。这里实现的是“离线预计算 item 文本 embedding + 轻量 User Transformer”的版本，不是 ByteDance 官方端到端 HLLM 训练栈，也不应把本文结果当作论文指标复现。

默认路径适合单卡实验：离线冻结 item 向量，训练小型 User Transformer；可用 `--d_model` 投影降维，并启用 `--loss_type sampled_softmax --amp`。模型接口也支持传入预训练 `user_model`，但示例默认不会加载它，更不包含两侧 LLM 的联合训练。16/24 GB 显存能容纳的规模取决于词表、序列长度和模型配置。

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

## 2. 当前模型做了什么

### 2.1 Item 侧（离线且冻结）

预处理器支持两种 `--model_type`：

| 参数 | HuggingFace 模型 | 隐藏维度 |
| --- | --- | ---: |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 |
| `baichuan2` | `baichuan-inc/Baichuan2-7B-Chat` | 4096 |

文本格式为：

```text
Compress the following sentence into embedding: title: {title}genres: {genres}
```

Amazon Books 使用 `title` 与 `description`。脚本取最后一个 token 的隐藏状态，按 token id 行号写入 `item_embeddings_<model_type>.pt`；第 0 行保留给 PAD。`HLLMModel` 检查 token 行号对齐并冻结 item 向量；默认维度一致时直接归一化，设置较小的 `--d_model` 时用可训练投影衔接 User 编码器。

缺少文本的 token 会保留为零向量。预处理日志会打印覆盖数量，正式实验应检查该数字，不能只检查文件是否存在。

### 2.2 User 侧（训练）

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

### 2.3 训练目标与 `NCELoss` 的真实语义

`SeqTrainer` 训练整段 next-token 目标，而不是只训练最后一个位置：

```text
logits[:, i, :]  -> seq_tokens[:, i + 1]
logits[:, -1, :] -> held-out targets
```

PAD 标签 `0` 被忽略。`--loss_type cross_entropy` 使用 `CrossEntropyLoss`；`--loss_type nce` 使用项目的 `NCELoss`。

需要注意：当前 `NCELoss` 对全量词表 logits 做 temperature 缩放、`log_softmax` 和目标类负对数似然，并未采样噪声，也未自动构造 in-batch negatives。因此不能据此宣称采样 NCE 的计算加速或额外指标提升。两个 HLLM 训练脚本都给 `NCELoss` 传入 `temperature=1.0`，因为模型输出已经按 `0.07` 缩放，避免重复 temperature。

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

# 3. 训练与评估
mkdir -p outputs/hllm_ml
python examples/generative/run_hllm_movielens.py \
    --model_type tinyllama \
    --epoch 5 \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --max_seq_len 200 \
    --loss_type nce \
    --device cuda \
    --save_dir outputs/hllm_ml \
    --seed 2022
```

默认数据目录是 `examples/generative/data/ml-1m/processed/`，应包含：

```text
vocab.pkl
train_data.pkl
val_data.pkl
test_data.pkl
movie_text_map.pkl
item_embeddings_tinyllama.pt
```

MovieLens 序列预处理采用按用户 leave-last-out：最后一次交互为 test target，倒数第二次为 validation target，之前的前缀生成训练样本；不是 70/10/20 用户随机划分。

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

## 6. 资源与评估注意事项

- 默认全词表训练会生成 `[B, L, V]` logits；`sampled_softmax` 则只计算正例与采样负例。大词表下可先选择采样训练，再降低 `--batch_size`、`--max_seq_len` 和 `--d_model`。
- TinyLlama/Baichuan2 只在离线 embedding 生成阶段运行；训练阶段使用预计算 embedding，但 2048/4096 维 User Transformer 本身仍然较大。
- 训练脚本报告 `SeqTrainer.evaluate()` 的 full-sequence loss、held-out top-1 accuracy，并额外计算 HR/NDCG@10/50/200。
- HLLM 示例使用 `SeqTrainer.predict()` 分块检索候选，排除 PAD；是否过滤历史由 `filter_seen` 控制。比较论文或线上指标前，应统一该选项与候选集合。
- 时间和显存受硬件、词表规模、序列长度与缓存状态影响；本文不提供未经基准脚本验证的固定耗时或百分比提升。

## 7. 与官方端到端 HLLM 的边界

当前仓库可以验证的能力：

- item 文本 embedding 按 token id 对齐并冻结；
- causal User Transformer 输出全词表 cosine logits；
- 支持 MovieLens-1M 与 Amazon Books 的预处理、训练和 top-k 评估示例；
- 单机 `SeqTrainer` 支持全词表 CE、兼容旧接口的全词表 `NCELoss`，以及只计算正例和采样负例得分的 `sampled_softmax`。

当前未实现或未对齐的部分：

- Item LLM 与 User LLM 的端到端联合训练；
- 官方大模型架构、可学习 item embedding token 与训练配置的逐项复刻；
- 官方完整的负采样策略与训练协议；
- DeepSpeed 或分布式训练；
- 统一的历史候选过滤协议和论文指标复现实验；
- 多步自回归解码与生产推理服务。

现有测试覆盖 item embedding 行号/维度校验和 cosine logits 范围，但不证明与官方实现的指标一致。因此更准确的定位是轻量研究示例，而不是“97% 对齐”或可直接生产部署的官方复现。
