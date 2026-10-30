# 生成式推荐：单设备用法与实现边界

本分支提供 HSTU、HLLM、TIGER 的低资源训练入口，目标是便于学习和小规模实验，不宣称已经复现论文指标。保留原来的模型调用方式，不增加统一生成式模型基类。

## 1. 模型与训练器的分工

| 模块 | 本框架的实现 | 与论文或官方代码的差别 |
| --- | --- | --- |
| HSTU | dense 注意力；未缩放 QK 加 head-shared bias，再做 SiLU/N | 保留本框架的同位置时间差输入；不是官方下一事件时间语义，也不兼容官方 checkpoint |
| HLLM | 冻结离线 Item 向量；默认使用随机初始化的小 Transformer，也可传入已有 User 模型 | 默认只是 HLLM-inspired baseline；冻结 Item 的预训练 User 模型方案也不是双 LLM 联合训练 |
| RQ-VAE | 隐藏层 ReLU，latent 和重建末层 Linear；顺序导出唯一语义 ID | 特征文本、向量模型和训练日程由用户确定，不能仅凭结构相似声称复现 |
| TIGER | 离线语义 tokenizer、用户桶 token、HF T5、目录约束 beam search | 使用 Hugging Face 而非论文 T5X；小配置不代表参数量、评估协议或结果完全一致 |

依据：[Meta HSTU 实现](https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/research/modeling/sequential/hstu.py)、[ByteDance HLLM 实现](https://github.com/bytedance/HLLM/blob/main/code/REC/model/HLLM/hllm.py)、[TIGER 论文](https://arxiv.org/html/2305.05065v3)。

HSTU/HLLM 保留 `forward(history, time_diffs) -> [B,L,V]`，另提供 `encode` 和 `score_items`，使训练器不必先生成全词表分数。TIGER 仍使用 Transformers Trainer，不强行塞进 SeqTrainer。

## 2. 保留旧调用，按需启用采样

原调用继续可用，默认不启用采样、AMP 或梯度累积：

```python
trainer = SeqTrainer(model, device="cuda:0")
trainer.fit(train_loader, val_loader)
```

大词表可改用：

```python
trainer = SeqTrainer(
    model, device="cuda:0",
    loss_type="sampled_softmax",
    loss_params={"num_negatives": 128},
    amp=True, accumulation_steps=4, max_grad_norm=1.0,
)
trainer.fit(train_loader, val_loader)
item_ids = trainer.predict(test_loader, top_k=10)
```

这里是正例加随机目录负例的交叉熵，不是召回模块的 B×B in-batch，也不是带 log-Q 修正的全词表 softmax 估计器。有效序列位置共享一组负例，只屏蔽抽中该位置正例本身的项；未引入召回 ID 去重规则。

- 序列必须连续，可左或右 padding；例如 `[1,2,3,0,0]` 的下一项为 4，监督为 `[2,3,4,0,0]`。
- `evaluate` 的 accuracy 使用全目录 top-1；采样模式的验证 loss 使用固定抽样，不可与全词表 CE 数值直接比较。
- `predict` 分块计算全目录 top-k，排除 PAD；`filter_seen=False` 是默认值，开启后不足 K 个候选用 -1 填充。测试 loader 应设置 `shuffle=False`，便于与原始目标逐行对齐。
- 每次验证 accuracy 严格提升立即保存 `model.pth`，正常结束恢复最佳权重。同分不替换；没有验证集时保存最终权重。最佳快照只由 EarlyStopper 管理并放在 CPU。
- 梯度累积是等权微批平均，尾部不足一个累积组也会更新；不会自动扩大负例池。
- 新采样路径只支持单设备；完整 optimizer/scaler 断点恢复和跨卡负例汇集后置。

## 3. 简单的小显存入口

以下命令需先准备好对应数据。CPU 运行时改为 `--device cpu` 并去掉 `--amp`。

HSTU MovieLens：

```powershell
python examples/generative/run_hstu_movielens.py --dataset_path YOUR_DATA --device cuda:0 --max_seq_len 50 --batch_size 16 --loss_type sampled_softmax --amp --accumulation_steps 4
```

HLLM 小编码器与离线缓存：

```powershell
python examples/generative/run_hllm_movielens.py --dataset_path YOUR_DATA --device cuda --d_model 256 --batch_size 8 --max_seq_len 50 --loss_type sampled_softmax --amp --learn_temperature
```

MovieLens 示例会按预处理的左 padding 约定保留最近 `max_seq_len` 个位置，避免数据仍长 200、模型却只配置 50。HSTU 示例沿用过滤历史的评估方式，HLLM 示例不滤历史，比较效果前要统一协议。

这些是起步配置，不是“任意 16G/24G 数据都保证运行”的承诺。特别是 HLLM 缓存仍整体常驻设备，float32 内存约为 `物品数 × 原始向量维度 × 4 bytes`；投影到 256 维不会消除原缓存。

若希望使用已有预训练 User 模型，可明确传入本地 causal base model：

```python
from transformers import AutoModel

base = AutoModel.from_pretrained(local_checkpoint, local_files_only=True)
model = HLLMModel(
    cache, len(cache), d_model=base.config.hidden_size,
    user_model=base, learn_temperature=True, max_seq_len=50,
    project_items=cache.shape[1] != base.config.hidden_size,
)
```

传入的是输出 hidden states 的 base model，而不是带语言词表预测头的 CausalLM 包装。调用方负责选用因果模型和确定冻结策略。本框架不会重新初始化其权重，也不自动下载权重或配置 LoRA。

## 4. RQ-VAE 与 TIGER 的目录必须一致

训练 RQ-VAE 可 shuffle，导出 SID 必须按目录顺序。业务 item ID 必须与向量行号或显式 `item_ids` 对齐，PAD 行不应作为推荐物品导出。

例如两个物品的 RQ 编码均为 `[a7,b1,c4]`，导出为 `[a7,b1,c4,d0]` 和 `[a7,b1,c4,d1]`；无碰撞的物品也有 d0。这个后缀用于区分物品，应保留，不能与 in-batch 负采样混为一谈。

离线 toy 示例：

```powershell
python examples/generative/run_tiger_movielens.py --mode generate-toy-data --data_inter_path NEW_DIR/inter.json --data_indice_path NEW_DIR/semantic_ids.json
python examples/generative/run_tiger_movielens.py --mode all --data_inter_path NEW_DIR/inter.json --data_indice_path NEW_DIR/semantic_ids.json --output_dir NEW_DIR/ckpt --epochs 2 --per_device_batch_size 8
```

生成 toy 数据必须显式调用且拒绝覆盖；真实数据直接 train/test。GPU 可另加 `--fp16 --gradient_checkpointing`。

使用 `--resume_from_checkpoint CHECKPOINT_DIR` 时，在写出配置或加载权重之前检查 SID 指纹、用户桶数量和 tokenizer 映射。映射不同，即使词表大小相同也拒绝续训；缺少指纹的旧 checkpoint 不能使用这个安全续训入口。测试入口保留旧 checkpoint 的兼容读取，但缺少指纹时需自行确认原目录。

TIGER 默认 tokenizer 完全离线；如选用 `--base_model` 的 T5 tokenizer，需额外安装 `sentencepiece`。

## 5. 权重兼容与验证范围

- HSTU 的相对偏置从每个 head 一份改为共享，数学缩放也改变；旧模型请重新训练，不使用 `strict=False` 掩盖不匹配。
- RQ-VAE 线性末层改变了结构和参数键。请重新训练、重新导出 SID，再训练 TIGER，不能混用旧 tokenizer/权重。
- HLLM 未开启投影、可学习温度、外部 User 模型时保留原参数结构；PAD 修正会改变错误的旧数值行为。
- 通用 MLP 的 `batchnorm` 默认仍为 True；RQ-VAE 的 `bn=False` 现在真正生效，不改变召回/排序默认配置。
- 绘图继续复用公共可视化入口，ONNX 导出保留原 forward 输出；不保证任意外部 User LLM 都可导出。

本次使用小型 CPU 数据验证训练、预测和保存边界，不把它当成论文效果或真实 16G/24G 显存测试。完整真实数据集效果、双 LLM 联合训练和多卡采样留待独立工作。
