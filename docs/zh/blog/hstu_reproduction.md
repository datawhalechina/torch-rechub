---
title: HSTU 模型复现说明
description: Meta HSTU 模型在 torch-rechub 中的复现说明，包括架构设计和实现细节
---

## HSTU 模型在 torch-rechub 中的复现说明

本文件说明当前 `torch-rechub` 中 HSTU（Hierarchical Sequential Transduction Units）的实现状态。核心层已经按论文 Eq. 2-4 和 Meta reference 的主要路径对齐：联合 UVQK 投影先做 SiLU、attention scores 加 `rab^{p,t}`、输出侧只保留 `f_2(Norm(AV) * U)`，层间残差放在 `HSTUBlock` 外部。

---

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

---

## 2. HSTULayer：Eq. 2-4

### 2.1 Eq. 2：联合 UVQK 投影

`HSTULayer` 先对输入做 `LayerNorm`，再通过一个线性层同时生成 `Q/K/U/V`：

```python
proj_out = F.silu(self.proj1(self.norm_in(x)))
```

这里的 `SiLU` 作用在完整 `UVQK` 投影上，然后再 split。因此 `U`、`V`、`Q`、`K` 四路都经过同一个非线性，避免只对 gate 侧单独激活。

### 2.2 Eq. 3：attention score 加 `rab^{p,t}`

当前 attention 路径为：

```python
scores = (Q @ K.transpose(-2, -1)) * (1.0 / sqrt(dqk))
scores = scores + rab(time_diffs, seq_len)
attn_weights = silu(scores) / max_seq_len
AV = attn_weights @ V
```

关键点：

- `rab^{p,t}` 是每个 head 一份的可学习 bias，来自相对位置差和相对时间差的桶化结果。
- bias 加在 attention scores 上，而不是加到输入 token embedding 上。
- HSTU 使用 `silu(scores) / N`，不是 Transformer 标准 softmax。
- causal mask 会阻止当前位置看未来 token；padding mask 会屏蔽 PAD 作为 key 的位置。

`RelativeBucketedTimeAndPositionBias` 的输入 `time_diffs` 约定为 `query_time - timestamp[i]`。做 pairwise 差时 anchor 会抵消，得到两两事件之间的相对时间差。

### 2.3 Eq. 4：门控输出与投影

当前输出路径为：

```python
gated = LayerNorm(AV) * U
output = f2(gated)
```

注意：

- `U` 已经在 Eq. 2 的联合投影里经过 `SiLU`，这里不再二次 `SiLU(U)`。
- `proj2` 只接收 gated attention output。
- 不再拼接 `[U, x, gated]`，也没有单独的 position-wise FFN。

### 2.4 外部残差

`HSTUBlock` 在层外执行残差：

```python
for layer in self.layers:
    x = x + layer(x, padding_mask=padding_mask, time_diffs=time_diffs)
```

这避免让 `proj2` 自己学习恒等映射，也与 HSTU 论文/reference 的层间残差形式一致。

---

## 3. `rab^{p,t}` 与时间特征

### 3.1 `RelativeBucketedTimeAndPositionBias`

`RelativeBucketedTimeAndPositionBias` 维护两组可学习参数：

- `pos_w`：相对位置差 `i - j` 的 per-head bias，表大小为 `2 * max_seq_len - 1`。
- `ts_w`：相对时间差 bucket 的 per-head bias，表大小为 `num_time_buckets + 1`。

时间差 bucket 化流程：

```python
dt = abs(time_diffs[i] - time_diffs[j]) / 60.0
bucket = sqrt(dt) 或 log(dt)
bucket = clamp(bucket / time_bucket_divisor, 0, num_time_buckets)
```

当 `time_diffs=None` 时，`rab` 退化为 position-only bias，输出形状为 `(1, H, L, L)`；传入 `time_diffs` 时输出形状为 `(B, H, L, L)`。

### 3.2 `RelPosBias` 的定位

`RelPosBias` 仍保留在 `hstu_utils.py`，但它是 legacy 的相对位置 bias：

- HLLM 仍会复用它。
- 兼容旧实验时可以单独使用。
- 当前 HSTU 主路径不再使用它，HSTU attention score 上的相对位置/时间建模由 `RelativeBucketedTimeAndPositionBias` 负责。

---

## 4. HSTUModel 包装层

`HSTUModel` 负责把 item token 序列转换成 hidden states，调用 `HSTUBlock`，并输出 vocab logits。

当前包装层包含：

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

### 5.1 时间差语义

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

这个格式可以让 `rab` 通过两两相减恢复事件间相对时间差。

### 5.2 数据集格式

`SequenceDataGenerator` 使用四元组格式：

```text
(seq_tokens, seq_positions, seq_time_diffs, targets)
```

其中 `seq_positions` 为兼容数据接口保留，当前 `HSTUModel` 会在 forward 中根据序列长度内部生成 position index，`SeqTrainer` 不使用 batch 中的 `seq_positions`。

### 5.3 训练目标

`SeqTrainer` 使用整段 next-token 交叉熵：

```text
logits[:, i, :] -> seq_tokens[:, i + 1]
logits[:, -1, :] -> targets
```

PAD token `0` 通过 `ignore_index=0` 从 loss 中排除。验证/测试中的 `evaluate` 返回平均 loss 和最后一个 held-out target 的 top-1 accuracy；示例脚本中的 ranking 评估额外计算 HR@K 和 NDCG@K。

---

## 6. 与 Meta reference 的一致性和差异

主要一致点：

- Eq. 2：完整 `UVQK` 投影先 `SiLU` 再 split。
- Eq. 3：attention score 加 per-head `rab^{p,t}`，再做 `silu(scores) / N`。
- Eq. 4：`f_2(LayerNorm(AV) * U)`，无 concat-u/x 旁路、无额外 FFN。
- 层间残差：由 `HSTUBlock` 在层外执行。
- 时间差：推荐使用 `query_time - timestamp[i]` 的 anchor-delta 形式。

仍保留的工程差异：

- 未包含 DLRM、多任务头和复杂特征交叉，本实现聚焦单任务 next-item prediction。
- `HSTUModel` 包装层保留绝对位置 embedding 和可选输入侧时间 embedding；核心 HSTU layer 已通过 `rab^{p,t}` 注入相对位置/时间 bias。
- 当前没有封装多步自回归解码、beam search 等生成接口。
- 初始化和部分训练配置不追求 bit-level 复现。

---

## 7. 近期 HSTU 相关改动摘要

- 修正 Eq. 2：`SiLU` 从只作用于 gate 侧改为作用于完整 `UVQK` 投影。
- 修正 Eq. 3：新增 `RelativeBucketedTimeAndPositionBias`，并在 attention score 上接入 `rab^{p,t}`。
- 修正 Eq. 4：移除 `[U, x, gated]` concat 输出旁路，`proj2` 只投影 gated attention output。
- 修正层间结构：`HSTUBlock` 改为外部残差 `x = x + layer(x)`。
- 更新文档与 API 说明：明确 `RelPosBias` 是 legacy/HLLM 路径，HSTU 主路径使用 `rab^{p,t}`。
