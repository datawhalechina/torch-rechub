---
title: HSTU 模型复现说明
description: Meta HSTU 模型在 torch-rechub 中的复现说明，包括架构设计和实现细节
---

## HSTU 模型在 torch-rechub 中的复现说明

本文件总结 torch-rechub 中对 Meta HSTU（Hierarchical Sequential Transduction Units）模型的复现情况，重点说明：

- 当前实现的整体架构与关键设计细节；
- 与 Meta 官方开源实现/论文的一致之处；
- 有意简化或仍然存在差异的部分。

---

## 1. 整体架构概览

### 1.1 模块划分

与 HSTU 相关的主要模块如下：

- **模型主体**：`torch_rechub/models/generative/hstu.py`
  - `HSTUModel`：Embedding + HSTUBlock + 输出投影
- **核心层与 Block**：`torch_rechub/basic/layers.py`
  - `HSTULayer`：单层 HSTU 转导单元（多头注意力 + 门控 + FFN）
  - `HSTUBlock`：多层 HSTULayer 堆叠
- **相对位置偏置与词表工具**：`torch_rechub/utils/hstu_utils.py`
  - `RelPosBias`、`VocabMask`、`VocabMapper`
- **时间感知数据预处理**：`examples/generative/data/ml-1m/preprocess_ml_hstu.py`
- **数据集与数据生成器**：`torch_rechub/utils/data.py`
  - `SeqDataset`、`SequenceDataGenerator`
- **训练与评估**：
  - `torch_rechub/trainers/seq_trainer.py`：`SeqTrainer`
  - `examples/generative/run_hstu_movielens.py`：示例脚本、评估指标

### 1.2 数据与任务

- 数据集：MovieLens-1M `ratings.dat`（包含时间戳）
- 任务形式：**Next-item prediction**（给定历史序列，预测下一个 item）
- 训练目标：自回归式的 next-token 交叉熵损失（仅使用序列最后一个位置的 logits）
- 评估指标：HR@K、NDCG@K（K=10, 50, 200）

---

## 2. HSTULayer 与 HSTUBlock 实现细节

### 2.1 HSTULayer：核心转导单元

`torch_rechub/basic/layers.py::HSTULayer` 实现了论文中的“Sequential Transduction Unit”核心思想：

1. **输入与线性投影**
   - 输入形状：`(B, L, D)`
   - 通过 `proj1: Linear(D → 2·H·dqk + 2·H·dv)` 同时产生 Q / K / U / V：
     - Q, K 形状：`(B, H, L, dqk)`
     - U, V 形状：`(B, H, L, dv)`

2. **多头自注意力 + causal mask**
   - 注意力打分：`scores = (Q @ K^T) / sqrt(dqk)`，形状 `(B, H, L, L)`
   - 使用严格的 **causal mask**：位置 i 只能看到 `≤ i` 的 token，防止未来信息泄露。
   - 可选加上相对位置偏置 `RelPosBias`。
   - softmax 后得到 `attn_weights`，再与 V 相乘得到 `attn_output`。

3. **门控机制（Gated Attention）**
   - 将注意力输出 `attn_output` 与门控向量 U 进行逐元素门控：
   - `gated_output = attn_output * sigmoid(U)`，形状 `(B, L, H·dv)`。

4. **输出投影与残差 + FFN**
   - 使用 `proj2: Linear(H·dv → D)` 将多头输出还原到模型维度。
   - 两个残差块：
     1. 自注意力 + 门控 + 投影 + Dropout + 残差
     2. LayerNorm + FFN(4D) + Dropout + 残差
   - 使用 `LayerNorm` 做 pre-norm，提升深层训练稳定性。

### 2.2 HSTUBlock：多层堆叠

`HSTUBlock` 是多个 `HSTULayer` 的简单堆叠：

- 初始化时构建 `n_layers` 个 HSTULayer；
- 前向传播中按顺序依次传递；
- 未做层间不同窗口/不同参数共享的“显式层级结构”，这一点属于对论文中“Hierarchical”概念的工程化简化。

这一设计与 Meta 官方开源代码的风格一致：通过多层堆叠来实现逐层抽象的“层级”表示，而不是显式的多分辨率分支。

---

## 3. 时间戳建模与时间嵌入

### 3.1 数据预处理中的时间差计算

文件：`examples/generative/data/ml-1m/preprocess_ml_hstu.py`

核心设计：

- 对每个用户的交互序列，使用滑动窗口生成 `(history, target)` 样本：
  - history = 序列前缀；target = 当前 prefix 之后的一个 item；
- 对于每个 history，计算 **相对于查询时间的时间差**：
  - 查询时间 = history 中最后一个事件的时间戳 `query_timestamp`；
  - 对每个历史事件 `ts`，时间差为 `query_timestamp - ts`；
  - 例如时间戳 `[100, 200, 300, 400]` → 时间差 `[300, 200, 100, 0]`；
- 时间差以秒为单位保存为 `seq_time_diffs`，与 `seq_tokens` 同长；
- 所有序列截断/左侧 padding 到固定长度 `max_seq_len`，padding 的时间差为 0。

这与 Meta 官方 HSTU 代码中 `query_time - timestamps` 的处理方式保持一致，而不是相邻事件时间间隔的形式。

### 3.2 模型中的时间嵌入与 bucket 化

文件：`torch_rechub/models/generative/hstu.py`

1. **时间嵌入表**
   - `self.time_embedding = nn.Embedding(num_time_buckets + 1, d_model, padding_idx=0)`
   - 其中 bucket 0 作为 padding bucket。

2. **时间差 → bucket 的映射**

```python
# 伪代码
# 1) 秒 → 分钟
minutes = time_diffs.float() / 60.0
# 2) 避免 log(0)
minutes = clamp(minutes, min=1e-6)
# 3) 按 sqrt 或 log 映射到 bucket
if fn == 'sqrt':
    bucket = sqrt(minutes)
elif fn == 'log':
    bucket = log(minutes)
# 4) 截断到 [0, num_time_buckets-1]
```

3. **嵌入融合与 Alpha 缩放**

- Token Embedding 使用 Alpha 缩放：`token_emb = token_embedding(x) * sqrt(d_model)`；
- Position Embedding 为标准的绝对位置嵌入；
- Time Embedding 通过上述 bucket 索引查表得到；
- 最终序列表示：`embeddings = token_emb + pos_emb + time_emb`。

这部分在最近一次提交中完成了对 Meta 官方实现的细节对齐：

- 修复了时间差计算方式（由相邻间隔 → 与查询时间差）；
- 增加了 `/60.0` 的时间单位转换；
- 增加了 `alpha = sqrt(d_model)` 的缩放。

---

## 4. 训练与评估流水线

### 4.1 SeqDataset 与 SequenceDataGenerator

文件：`torch_rechub/utils/data.py`

- 近期提交中已**移除旧 3 元组格式的向后兼容逻辑**，统一为 4 元组：
  - `(seq_tokens, seq_positions, seq_time_diffs, targets)`；
- `SeqDataset` 负责将 NumPy 数组转换为 PyTorch 张量；
- `SequenceDataGenerator` 根据给定的 train/val/test 划分构造 DataLoader。

### 4.2 SeqTrainer：训练与评估

文件：`torch_rechub/trainers/seq_trainer.py`

- `train_one_epoch`：
  - 输入 batch 形如：`(seq_tokens, seq_positions, seq_time_diffs, targets)`；
  - 将张量移动到设备；
  - 调用 `model(seq_tokens, seq_time_diffs)` 得到 `(B, L, V)` logits；
  - 只取最后一个位置 `logits[:, -1, :]` 与 `targets` 做交叉熵损失；
- `evaluate`：
  - 与训练阶段类似，同样只使用序列最后一个位置；
  - 统计平均 loss 与 top-1 准确率，用于早停与模型选择。

### 4.3 示例脚本与推荐指标

文件：`examples/generative/run_hstu_movielens.py`

- 负责加载预处理好的 MovieLens 数据（真实数据），构造数据加载器与模型；
- 使用 `SeqTrainer` 进行训练与验证；
- `evaluate_ranking` 函数在测试集上计算 HR@K 与 NDCG@K：
  - 模型同样使用最后一个位置的 logits；
  - 对所有候选 item 排序，计算 top-K 命中率与折损累计增益。

近期在修复时间戳处理逻辑后，测试集指标相比旧实现有显著提升（以 K=10 为例）：

- HR@10：约从 0.17 提升到 0.21+
- NDCG@10：约从 0.08 提升到 0.11+

这表明时间衰减建模对生成式推荐效果有明显正向作用。

---

## 5. 与 Meta 官方实现的一致性与差异

### 5.1 主要一致点

与 Meta 官方 HSTU / DLRM-HSTU 实现相比，本框架在以下方面保持较高一致性：

- **核心层结构**：HSTULayer 采用 Q/K/V/U 四路线性投影、多头注意力、门控机制与两段残差 FFN，结构上与官方实现高度一致；
- **因果掩码**：在注意力打分阶段使用严格的 causal mask，保证生成式任务的因果性；
- **时间差定义**：使用 `query_time - timestamps` 形式的时间差，而非相邻事件间隔；
- **时间 bucket 化与嵌入**：支持 sqrt/log 两种 bucket 映射，配合时间嵌入表，与官方思路对齐；
- **Alpha 缩放**：对 token embedding 乘以 `sqrt(d_model)`，与官方实现中的缩放策略一致；
- **训练目标**：自回归式的 next-item 交叉熵目标，等价于语言模型式训练。

### 5.2 主要差异与简化

目前实现仍有以下差异或有意简化：

1. **未包含 DLRM 与多任务头**
   - 官方 DLRM-HSTU 实现支持复杂的特征交叉与多任务学习；
   - 本框架专注于单任务的 next-item prediction，未实现 DLRM 部分与多目标头。

2. **相对位置偏置为简化版本**
   - 当前的 `RelPosBias` 基于 `|i - j|` 距离做线性分桶；
   - 未显式区分方向（正/负距离）、也未使用更复杂的 log-scaling bucket 公式；
   - 这在工程上更简单，但与官方实现存在细节差异。

3. **仅提供单步 next-item 预测接口**
   - 训练和评估阶段都是“给定完整历史 → 预测下一个 item”；
   - 尚未封装多步自回归解码接口（如 beam search 生成未来 N 步序列）；
   - 对于大多数推荐 benchmark（只评估下一步）已经足够，但与“通用生成式序列模型”相比功能较少。

4. **部分初始化细节不同**
   - 当前使用 `xavier_uniform_` 初始化大部分线性层和嵌入；
   - 官方实现中某些嵌入可能使用基于维度的 `uniform(-sqrt(1/N), sqrt(1/N))`；
   - 这类初始化差异对最终收敛影响有限，但不是 100% bit-level 复现。

---

## 6. 近期提交总结

- 引入了 HSTU 模型、HSTULayer/HSTUBlock、SeqTrainer、SeqDataset 等完整骨架；
- 实现了基本的生成式 next-item 训练与评估流程；
- 时间戳处理、时间嵌入与部分细节尚处于初版实现阶段。

- 重构 MovieLens 预处理脚本：
  - 使用滑动窗口策略大幅增加训练样本；
  - 按用户划分 train/val/test，避免数据泄漏；
  - 正确使用 `query_time - timestamps` 形式的时间差；
- 修复时间嵌入实现：
  - 添加秒 → 分钟的时间单位转换；
  - 增加 `alpha = sqrt(d_model)` 缩放；
  - 与官方时间建模逻辑对齐；
- 清理向后兼容逻辑：
  - 移除 3 元组数据格式，统一为 4 元组 `(tokens, positions, time_diffs, targets)`；
  - 简化 SeqDataset、SequenceDataGenerator、SeqTrainer 代码结构；
- 训练与评估结果显示所有排名指标均有显著提升，验证了时间建模修复的必要性和有效性。

---

## 7. 小结

- 当前实现已经在 **HSTU 核心层结构、时间建模与训练目标** 上与 Meta 官方实现高度对齐；
- 同时刻意简化了 DLRM、多任务头、复杂特征工程等工程部分，使得该实现更适合作为研究和教学的参考版本；
- 如果后续需要进一步逼近“论文级完全复现”，推荐优先完善：
  1. RelPosBias 的 bucket 公式与方向建模；
  2. padding mask 的显式支持；
  3. 多步自回归解码接口与更复杂的下游任务场景。

