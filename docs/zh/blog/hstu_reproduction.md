---
title: HSTU 模型复现说明
description: Meta HSTU 模型在 torch-rechub 中的复现说明，包括架构设计和实现细节
---

# HSTU 模型在 torch-rechub 中的复现说明

HSTU 读入用户按时间排列的观看记录，预测下一部电影。它既利用顺序，也能利用两次观看隔了多久。第一次运行可先看下面的 MovieLens 命令，检查训练后是否打印了 Hit@K 和 NDCG@K。想对照论文读源码，再从第 2 节的三条公式看起。

这里用单任务的下一物品预测来复现 HSTU 核心层。位置和时间的处理、数据切分、训练预算与原论文都有差异。早期版本用过另一种核心层输出结构，升级这类旧实验时需要重新训练。

## 1. 模块划分

与 HSTU 相关的主要模块如下：

- **模型主体**：`torch_rechub/models/generative/hstu.py`
  - `HSTUModel`：item token embedding、位置/时间输入 embedding、`HSTUBlock`、输出投影。
- **核心层与 Block**：`torch_rechub/basic/layers.py`
  - `HSTULayer`：实现 HSTU Eq. 2-4 的单层 sequential transduction unit。
  - `HSTUBlock`：按 `x = x + HSTULayer(x)` 堆叠多层。
- **HSTU / HLLM 工具**：`torch_rechub/utils/hstu_utils.py`
  - `RelativeBucketedTimeAndPositionBias`：HSTU 使用的 `rab^{p,t}` attention-score bias。
  - `RelPosBias`：legacy 相对位置偏置，保留给 HLLM 和兼容实验；当前 `HSTUModel` 不使用它。
  - `VocabMask`：推理/排序时屏蔽 PAD 或非法 item。
- **数据与训练**：
  - `torch_rechub/utils/data.py`：`SeqDataset`、`SequenceDataGenerator`。
  - `torch_rechub/trainers/seq_trainer.py`：`SeqTrainer`。
  - `examples/generative/run_hstu_movielens.py`、`examples/generative/run_hstu_amazon_books.py`：训练与 ranking 评估示例。

### 在 MovieLens 上跑一轮训练

从仓库根目录先运行预处理，生成 `processed/` 中的词表和训练数据，再启动训练。

```bash
pip install -e .

# 缺少 MovieLens 原始文件时会自动下载；已有文件可加 --no_download
python examples/generative/data/ml-1m/preprocess_ml_hstu.py

# 在 CPU 上跑一个真实数据 epoch
mkdir -p outputs/hstu_ml
python examples/generative/run_hstu_movielens.py \
    --device cpu --epoch 1 --batch_size 16 \
    --save_dir outputs/hstu_ml
```

使用单张 CUDA 卡时，可以在相同的 MovieLens 预处理结果上运行下面的起步配置。每批 8 条历史，保留最近 32 次交互，每个目标采 32 个负物品，每两批更新一次参数。训练结束后脚本还会在留出的物品上打印 Hit@K 和 NDCG@K。

```bash
python examples/generative/run_hstu_movielens.py --device cuda --epoch 1 --batch_size 8 --max_seq_len 32 --loss_type sampled_softmax --num_negatives 32 --accumulation_steps 2 --amp --save_dir outputs/hstu_ml
```

在 RTX 5080（16 GB）上，用 MovieLens-1M 的用户 1–32、每人最近最多 24 条交互验证了上述参数。子集含 578 部电影，训练、验证、测试各 32 条用户序列。PyTorch 统计的峰值张量显存约 20 MiB，包含训练和评估，不含驱动、CUDA 上下文与缓存保留量。完整数据集需按实际词表和序列长度测量。

Amazon Books 使用 `examples/generative/data/amazon-books/preprocess_amazon_books.py` 生成输入，训练脚本为 `examples/generative/run_hstu_amazon_books.py`。Amazon Books 的训练脚本也支持上述单卡采样参数。采用默认交叉熵时，训练开销随 `[batch, 序列长度, 词表大小]` 增长。预处理会写入词表和三个数据切分，新实验请选独立输出目录。

---

## 2. HSTULayer：Eq. 2-4

### 2.1 Eq. 2：联合 UVQK 投影

`HSTULayer` 先对输入做 `LayerNorm`，再通过一个线性层同时生成 `Q/K/U/V`：

```python
proj_out = F.silu(self.proj1(self.norm_in(x)))
```

注意两步的顺序：先对完整投影做 `SiLU`，再拆成四组。因此 `U`、`V`、`Q`、`K` 都经过了同一个非线性变换。

### 2.2 Eq. 3：attention score 加 `rab^{p,t}`

当前 attention 路径为：

```python
scores = Q @ K.transpose(-2, -1)
scores = scores + rab(time_diffs, seq_len)
attn_weights = silu(scores) / max_seq_len
AV = attn_weights @ V
```

同样两个物品，隔一天出现和隔一年出现，模型可以学到不同的偏置。`rab` 把相对位置差和时间差映射到对应的偏置值，加到注意力得分上，所有注意力头共用这张表。接着用 `silu(scores) / N` 计算权重。掩码让每个位置只看当前及更早的有效物品，跳过补位。

`RelativeBucketedTimeAndPositionBias` 的输入 `time_diffs` 约定为 `query_time - timestamp[i]`。做 pairwise 差时 anchor 会抵消，得到两两事件之间的相对时间差。

### 2.3 Eq. 4：门控输出与投影

当前输出路径为：

```python
gated = LayerNorm(AV) * U
output = f2(gated)
```

`U` 在生成时已经经过 `SiLU`，可以直接用它给归一化后的注意力输出加门控。`proj2` 只接收这个乘积。

### 2.4 外部残差

`HSTUBlock` 在层外执行残差：

```python
for layer in self.layers:
    x = x + layer(x, padding_mask=padding_mask, time_diffs=time_diffs)
```

这样，上一层的输入可以直接传给下一层，`HSTULayer` 只需学出要加上的变化。

---

## 3. `rab^{p,t}` 与时间特征

### 3.1 `RelativeBucketedTimeAndPositionBias`

`RelativeBucketedTimeAndPositionBias` 维护两组可学习参数：

- `pos_w`：相对位置差 `i - j` 的 per-head bias，表大小为 `2 * max_seq_len - 1`。
- `ts_w`：相对时间差 bucket 的 per-head bias，表大小为 `num_time_buckets + 1`。

时间差 bucket 化流程：

```text
dt = abs(time_diffs[i] - time_diffs[j]) / 60.0
bucket = sqrt(dt) 或 log(dt)
bucket = clamp(bucket / time_bucket_divisor, 0, num_time_buckets)
```

当 `time_diffs=None` 时，`rab` 退化为 position-only bias，输出形状为 `(1, H, L, L)`；传入 `time_diffs` 时输出形状为 `(B, H, L, L)`。当前 `HSTULayer` 为此模块设置 `n_heads=1`，再广播到各 attention head，共享同一组 bias。

### 3.2 源码中的另一种位置偏置

`hstu_utils.py` 中还保留了 `RelPosBias`，供 HLLM 和旧实验使用。阅读 HSTU 的注意力计算时，请看 `RelativeBucketedTimeAndPositionBias`。

## 4. 从观看序列到候选物品分数

`HSTUModel` 先把物品 ID 序列转成向量，送入 `HSTUBlock` 后，再为每个位置算出词表中各物品的分数。

从输入端到输出端，依次经过下面几步：

- `token_embedding(vocab_size, d_model, padding_idx=0)`。
- 绝对 `position_embedding(max_seq_len, d_model)`。
- 可选输入侧 `time_embedding(num_time_buckets, d_model)`，由 `use_time_embedding` 控制。
- `HSTUBlock(..., num_time_buckets, time_bucket_fn, time_bucket_divisor)`。
- 默认 tied embedding 输出投影：`F.linear(hstu_output, token_embedding.weight, output_bias)`。

需要区分两条时间/位置路径：

- HSTU 核心 Eq. 3 的相对位置/时间建模发生在 attention scores 上，即 `rab^{p,t}`。
- 当前 `HSTUModel` 包装层仍保留绝对位置 embedding 和可选输入侧时间 embedding，用于兼容现有数据接口与实验设置。

PAD token 约定为 `0`。模型会在输入 embedding 后和 HSTU 输出后显式清零 PAD 行，避免位置/时间 embedding 从 PAD 位置泄漏信号。

---

## 5. 数据与训练约定

### 5.1 每个位置的时间差怎样算

预处理脚本生成的 `seq_time_diffs` 采用以下语义：

```text
time_diffs[i] = query_time - timestamp[i]
```

其中 `query_time` 通常是当前 history 中最后一个行为的时间戳。例子：

```text
timestamps  = [100, 200, 300, 400]
query_time  = 400
time_diffs  = [300, 200, 100, 0]
```

两项都减过同一个查询时间，所以再做差时，这个公共时间会抵消。模型仍然能算出两次行为相隔多久。

### 5.2 数据集格式

`SequenceDataGenerator` 使用四元组格式：

```text
(seq_tokens, seq_positions, seq_time_diffs, targets)
```

其中 `seq_positions` 为兼容数据接口保留，当前 `HSTUModel` 会在 forward 中根据序列长度内部生成 position index，`SeqTrainer` 不使用 batch 中的 `seq_positions`。

### MovieLens 每用户序列采样

MovieLens 预处理器默认使用 `user_sequence`：每个符合长度要求的用户保留一条最近训练序列，对所有有效位置计算 next-token 损失。

每个用户的最后一次交互留作测试、倒数第二次留作验证。对于序列 `1 2 3 4 5 6 7 8`，`user_sequence` 的训练输入是 `1 2 3 4 5`，逐位置标签是 `2 3 4 5 6`；验证目标为 `7`，测试目标为 `8`。最大输入长度为 200 时，训练窗口使用最后至多 201 个训练交互，更早的交互不会进入该窗口。它借鉴每用户采样的思路，但保留本项目的独立验证集、左 padding 和当前事件时间偏置，不等同于官方完整训练协议。

从仓库根目录运行。训练示例默认训练 101 轮：

```bash
python examples/generative/data/ml-1m/preprocess_ml_hstu.py \
    --training_sampling user_sequence \
    --output_dir outputs/hstu_ml_user_sequence/data
mkdir -p outputs/hstu_ml_user_sequence/model
python examples/generative/run_hstu_movielens.py \
    --dataset_path "$PWD/outputs/hstu_ml_user_sequence/data" \
    --save_dir outputs/hstu_ml_user_sequence/model \
    --device cuda --epoch 101 --batch_size 128
```

`preprocessing_config.json` 记录采样模式、最大长度及留出数量。

每轮遍历所有符合条件的用户，每个用户贡献一条训练样本。训练时使用验证集选择模型权重，并记录更新步数和实际耗时；测试集用于最终效果评估。

### 5.3 训练目标

`SeqTrainer` 使用整段 next-token 交叉熵：

```text
logits[:, i, :] -> seq_tokens[:, i + 1]
logits[:, -1, :] -> targets
```

PAD token `0` 通过 `ignore_index=0` 从 loss 中排除。验证/测试中的 `evaluate` 返回平均 loss 和最后一个 held-out target 的 top-1 accuracy；示例脚本中的 ranking 评估额外计算 HR@K 和 NDCG@K。

---

## 6. 与 Meta 实现的对照

核对核心层时可以按这三个位置读：联合投影后的 `SiLU` 对应论文公式 2，得分上的相对位置和时间偏置对应公式 3，`LayerNorm(AV) * U` 后接输出投影对应公式 4。残差由外层的 `HSTUBlock` 相加。

影响实验可比性的差异主要在核心层之外。本项目额外保留了输入端的绝对位置嵌入和可选时间嵌入。注意力偏置取当前交互事件之间的时间差。MovieLens 示例按用户留一条训练序列，单独划出验证目标，训练和评估只做下一物品预测。复用实验结果时，应同时记录这些约定和训练预算。
