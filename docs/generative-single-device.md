# 生成式推荐：单设备训练闭环与论文差异说明

审查日期：2026-10-15。分支：`pr/generative-single-device`。基线：源仓库 `948d24f5`（已包含召回 in-batch PR #274）。本分支不包含 SENet 修复，也不替代多卡分支。

## 1. 先说边界

这次交付的是能在本框架内训练、验证、保存、重载和推荐的低资源实现，不是三篇论文结果的完整复现。接口默认保留旧调用方式；显存优化显式开启。没有下载大型预训练权重，没有进行完整 Amazon/MovieLens 数据集训练，也没有实测 24GB 卡。

特别注意：HLLM 的“小编码器＋缓存 Item LLM 向量”只能称为 HLLM-inspired baseline。传入真实预训练 User LLM 后，是冻结 Item LLM 的适配版本，仍不等同于官方双 LLM 联合训练。

## 2. 原有实现、此次改动、参考实现的对应关系

| 模块 | 原有问题 | 此次修改 | 仍存在的论文差异 |
| --- | --- | --- | --- |
| HSTU 输出 | 每次前向都生成 [B,L,V] | 保留 forward，增加 encode 和 score_items；采样训练不走全词表输出 | 不改原有 dense 注意力结构，不能称参数完全对齐 |
| SeqTrainer 监督 | 目标总放在数组最后一列；右 padding 会丢失最后目标 | 目标放到最后一个真实历史位置；左右 padding 均可；内部 PAD 间隙明确报错 | 历史必须是连续序列；不支持把 PAD 当缺失事件 |
| 序列损失 | 旧 nce 实际仍计算全词表 | 保留旧 nce；新增 sampled_softmax，默认 128 个目录负例，屏蔽同 ID 负例 | 使用批内位置共享的目录负例，不是逐位置独立抽样，也不是召回的 B×B in-batch |
| HLLM 序列编码 | PAD 被当成 key；User 编码器随机初始化；维度被缓存向量宽度绑死 | PAD mask；可选小投影；可传入 causal User base model，保留其权重 | 离线 Item 缓存冻结；没有联合更新 Item LLM |
| HLLM 对比目标 | 固定温度、没有相似负例过滤 | 可选 learn_temperature；采样损失可设相似度过滤阈值 0.99 | 投影版本在投影后的空间过滤；未做官方跨卡 gather |
| RQ-VAE→SID | shuffle 后按行编号会错配；重试仍可能碰撞；导出修改量化器参数 | 顺序导出、可传业务 item_ids、确定性碰撞后缀、导出无 sk_epsilon 修改 | 后缀取决于固定目录；新增/删除目录可能改变碰撞组编号 |
| RQ-VAE Trainer | 默认 10 轮、50 轮评估，可能不存任何可用文件；scheduler 不执行 | 最后轮也评估保存；scheduler 正常 step；非法 eval_step/空目录明确报错 | best_loss 是各 batch loss 之和，主要用于同一 loader 内比较 |
| TIGER 输入 | User ID 未使用；自然语言词表不必要 | 可选稳定用户 hash token；示例默认 2000 桶；离线语义词表 | 哈希规则是本框架固定 SHA256；不是原项目某一特定 hash 实现 |
| TIGER 运行 | all 隐式覆盖真实数据；测试时补随机 token 掩盖不匹配 | toy 必须显式生成且拒绝覆盖；检查词表、目录 fingerprint；默认不启动外部日志服务 | 安全修复有意改变 all 的数据准备行为 |
| TIGER 评估 | 重复 beam 可重复计算同一物品命中 | 对生成物品去重，保留合法目录约束 | Trie 约束是工程化选择；不能与无约束论文数字直接比较 |

### HSTU 的参考证据

Meta 的 rel_bias 路径使用未额外除以 sqrt(dqk) 的 QK，并以 SiLU/N 归一化；其时间偏置涉及下一事件时间与当前 key 时间。现有 Torch-Rechub 则保留缩放 QK、按 head 独立偏置以及同位置时间差。此次没有悄悄重写这些结构，也没有宣称已消除差异。[官方 HSTU 源码](https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/research/modeling/sequential/hstu.py)

官方 MovieLens 配置使用 sampled-softmax、128 负例、归一化打分和 0.05 温度。资源测量使用接近该量级的 50 维、8 层、2 heads、dqk=dv=25，但测量不是官方实验结果。[官方配置](https://github.com/meta-recsys/generative-recommenders/blob/main/configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin)

官方负采样器按监督位置抽样；本实现为了简单与显存可控共享一组负例。它们的负例分布和相关性不完全相同，不能把“也使用 128”当成复现对齐。[官方负采样代码](https://github.com/meta-recsys/generative-recommenders/blob/main/generative_recommenders/research/modeling/sequential/autoregressive_losses.py)

### HLLM 的参考证据

官方 HLLM 包含 Item LLM 与 User LLM，NCE 使用可学习 logit scale、相似负例过滤和跨设备负例汇集。本次只补齐在单设备、冻结 Item 缓存条件下可实现的对应逻辑。[ByteDance 官方实现](https://github.com/bytedance/HLLM/blob/main/code/REC/model/HLLM/hllm.py)

`user_model=None`：仍是小型 Transformer baseline。传入 `AutoModel` 而不是带词表预测头的 `AutoModelForCausalLM`，可直接用 inputs_embeds 输出用户序列状态；模型必须因果、hidden_size 必须匹配。小规模验证用的是随机 tiny Llama，只证明接口和权重保留，没有证明预训练效果或真实 LoRA 训练成功。

### TIGER 的参考证据

论文将 RQ 三段编码增加碰撞消歧段，并用用户桶 token 与语义历史预测下一个 SID。本文参考这些机制；使用 Hugging Face T5 而非论文的 T5X。按论文列出的 128 维、4 层、6 heads 等字段构造出的当前 T5 为约 485 万参数，而论文报告约 1300 万，因此只称“小配置适配”，不称参数精确复现。训练步数、优化器日程和数据规模也未复现。[TIGER 原论文](https://arxiv.org/html/2305.05065v3)

## 3. 对初学者而言，接口变在哪里

旧方式继续可用：

```python
model = HSTUModel(vocab_size=...)
trainer = SeqTrainer(model, device="cuda:0")
trainer.fit(train_loader, val_loader)
```

启用目录采样只需改变训练器参数，不在模型 forward 中加“提前返回向量”开关：

```python
trainer = SeqTrainer(
    model,
    device="cuda:0",
    loss_type="sampled_softmax",
    loss_params={"num_negatives": 128},
    amp=True,
    accumulation_steps=4,
    max_grad_norm=1.0,
)
trainer.fit(train_loader, val_loader)
item_ids = trainer.predict(test_loader, top_k=10)
```

模型分工：
`encode(history) -> [B,L,D]`；`score_items(states, ids) -> 候选分数`。
训练器负责目标对齐、抽样、损失、优化、评估。旧 `forward(history) -> [B,L,V]` 没有删改，原 ONNX 输出契约保留。

可选 HLLM 小配置：

```python
model = HLLMModel(
    item_embeddings=cache, vocab_size=len(cache),
    d_model=256, n_heads=8, n_layers=2, max_seq_len=50,
    project_items=True, learn_temperature=True,
)
trainer = SeqTrainer(
    model, device="cuda:0", amp=True,
    loss_type="sampled_softmax",
    loss_params={
        "num_negatives": 128,
        "negative_similarity_threshold": 0.99,
    },
)
```

更接近 HLLM 的 User LLM 适配需要你提供本地预训练模型。不要把 256 维小编码器称为 TinyLlama 本身：

```python
from transformers import AutoModel
base = AutoModel.from_pretrained(local_checkpoint, local_files_only=True)
# 可由调用方先冻结层或配置 LoRA；本模型不会重新初始化 base 的参数。
model = HLLMModel(
    cache, len(cache), d_model=base.config.hidden_size,
    user_model=base, learn_temperature=True, max_seq_len=50,
)
```

此路径仅验证 tiny Llama 的接口，未下载或验证真实 checkpoint/LoRA。调用方负责模型是否因果、缓存来自哪个 Item 模型、冻结策略和存储映射。

## 4. 核心代码差异举例

### 4.1 右 padding

输入 history=[1,2,3,0,0]，target=4：

- 原监督：[2,3,0,0,0]；末项 4 被 padding mask 丢失。
- 新监督：[2,3,4,0,0]；状态位置 2 预测 4。
- 左 padding [0,0,1,2,3] 仍监督 [0,0,2,3,4]。
- [1,0,3] 不是合法连续历史，拒绝而不是默认“跳过缺失事件”。

### 4.2 大词表训练

原：先产生 [B,L,V]，再 reshape 算 CE。

新：先得到 [B,L,D]，只取有效监督位置；正例分数使用小块的匹配对角线，负例只算 [有效位置数,128]。屏蔽抽中同一正例 ID 的负例。没有生成“所有正例两两组合”的巨大矩阵。

注意：sampled_softmax 在这里是候选集合上的对比 CE，没有实现 TensorFlow 那类带 log-Q 修正的通用无偏全词表 softmax 估计器。

### 4.3 评估

SeqTrainer.evaluate 的 accuracy 始终是完整目录 top-1，不用抽样目录虚增命中。sampled 模式的 val_loss 是固定负例的候选 CE，与全词表 CE 不可直接比数值。

SeqTrainer.predict 分块扫描整个目录取 exact top-k。PAD 永远剔除；filter_seen 默认 False，因为下一次真实交互可能重复历史物品。过滤后候选不足时填 -1，不把已看物品重新塞回 TopK。

SeqTrainer 的 CPU 绘图和 ONNX 导出会恢复原 device/training，包括执行失败的情况。已实测 CPU 画图/导出后继续 CUDA 反向，以及动态输入的 ONNX Runtime 与 PyTorch 输出一致。这不代表外部大 User LLM 也已完成 ONNX 导出。

原例子的 ranking evaluator 也改为只给最后一个真实状态评分，不再分配 [B,L,V]；它们仍一次给 [B,V] 打分。更大目录可直接使用 Trainer.predict 的分块路径。HSTU 示例仍采用原“过滤历史”协议，HLLM 示例默认不滤历史；比较模型效果时必须先统一协议。

### 4.4 保存与优化

每次 fit 正常结束和早停都恢复最佳验证权重。主指标仍是 accuracy，同分时以更低 val_loss 决胜。最佳快照存在 CPU，避免再复制一份大 item table 到 GPU。

AMP 只包前向；梯度累积完再 unscale、裁剪、step。不足一个完整累积组的最后几批按实际微批数除 loss。此实现是等权微批平均，不应宣称与变长序列拼接后的逐 token 加权平均完全相同。梯度累积扩大有效更新批量，不扩大负例池。[PyTorch AMP 说明](https://github.com/pytorch/pytorch/blob/v2.9.1/docs/source/notes/amp_examples.rst)

### 4.5 SID 对齐与碰撞

若业务物品 100、101 的原 RQ code 都是 [a7,b1,c4]：

- 原：随机 loader 导出可能把其他行的 code 放到 key=100；重复 code 还可能无法消歧。
- 新：按目录顺序量化，业务 item_ids 对齐行号；固定目录排序后分别输出 [a7,b1,c4,d0]、[a7,b1,c4,d1]。
- 不碰撞的物品也有 d0。
- 训练 RQ-VAE 的 loader 可以 shuffle；导出 loader 不可以。
- 缓存表第 0 行为 PAD 时，RQ 示例默认从导出的可推荐目录中移除该行；无 PAD 的普通向量集用 --padding_idx -1。

## 5. 可以直接运行的入口

HSTU MovieLens（已有预处理数据）：

```powershell
python examples/generative/run_hstu_movielens.py --dataset_path YOUR_DATA --device cuda:0 --loss_type sampled_softmax --batch_size 16 --amp --accumulation_steps 4
```

HLLM 小编码器（已有缓存向量和预处理数据）：

```powershell
python examples/generative/run_hllm_movielens.py --dataset_path YOUR_DATA --device cuda --loss_type sampled_softmax --d_model 256 --batch_size 8 --max_seq_len 50 --amp --learn_temperature
```

TIGER toy 需先显式准备，已有文件会拒绝覆盖：

```powershell
python examples/generative/run_tiger_movielens.py --mode generate-toy-data --data_inter_path NEW_DIR/inter.json --data_indice_path NEW_DIR/semantic_ids.json
python examples/generative/run_tiger_movielens.py --mode all --data_inter_path NEW_DIR/inter.json --data_indice_path NEW_DIR/semantic_ids.json --output_dir NEW_DIR/ckpt --epochs 2 --fp16 --gradient_checkpointing
```

TIGER 真数据：先训练 RQ-VAE，顺序导出唯一 SID；检查 inter.json 的 ID 与 embedding 行号或 item_ids 一致；再 train/test。all 不再负责制造数据。

## 6. 实测资源与验证范围

本机：RTX 5080，15.89GiB，Torch 2.9.1+cu130。单进程顺序执行。下表是合成数据、两批训练和目录预测的 shape 测量，不是训练到收敛；不含数据预处理 Item LLM、完整磁盘导出和全部训练轮次开销。

| 测量配置 | 峰值 allocated | 峰值 reserved |
| --- | ---: | ---: |
| HSTU：10万物品，B16/L200/D50，8层/2head，128负例、AMP | 279MiB | 306MiB |
| HLLM baseline：10万×2048冻结缓存，D256/2层，B8/L50，128负例、AMP | 867MiB | 896MiB |
| TIGER 小配置：约485万参数，B8，输入201 token，输出5 token，AMP＋checkpointing | 147MiB | 194MiB |

所以这些配置在本机 16GB 上可运行；不能由此推出“任何 16GB/24GB 数据集都能跑”。HLLM 冻结向量本身按 V×embedding_dim×4 bytes 增长；例如百万×4096 的 float32 表单独就约 15.3GiB，投影也不能消除这份缓存。完整双 LLM＋Adam 状态也不是本次资源测量对象。

已验证：

- HSTU/HLLM、小型外部 User Llama：默认与 sampled 各 CPU/GPU，目标对齐、B=1 尾批、有限损失、参数更新、候选打分一致、权重保存、exact top-k、已看过滤。
- RQ-VAE：CPU/GPU 两轮、eval_step 大于轮数仍保存、尾批一行、导出行映射、collision suffix 唯一、更换导出 batch_size 不改变 SID、shuffle 导出拒绝。
- TIGER：CPU/GPU 前向/反向、离线 token、保存重载、目录约束 beam 输出。
- 现有 HSTU/HLLM/TIGER 测试 30 项通过；没有新增测试文件，调整原 TIGER 端到端用例使其不依赖下载。
- 项目格式脚本按修改文件执行。修改文件 F821 检查通过。全项目 F821 仍有原 EDCN bridge_i 问题，归已有 pr/fix-validation-release，不混入本分支。

外部诊断脚本与日志放在本次 review-and-runtime-20261015 目录，不进入仓库。原示例入口的默认/采样配置也用隔离合成文件复跑；不是完整真实数据集实验。

## 7. 兼容与迁移提示

1. SeqTrainer 既有 positional 参数不变；新参数追加在末尾，默认不开 AMP/累积/采样。
2. HSTU/HLLM forward 仍返回 logits；新接口没有 forward 提前返回 flag。
3. HLLM 未开启投影/可学习温度/外部 User 模型时，原参数名结构仍可加载；PAD 行的错误影响会被修正，不能要求保持错误的 PAD 数值。
4. RQ-VAE 原 bn=False 实际被 MLP 的默认 BN 忽略。这次让它真正禁用 BN。旧 checkpoint 如含 BN 参数，加载时使用 bn=True 保留其原结构；不能 strict=False 当成正常迁移。通用 MLP 新参数 batchnorm 默认 True，召回/排序原默认不变。
5. SID 新增消歧段后，必须重新导出目录并重新训练 TIGER，不能混用旧 tokenizer/checkpoint。get_indices 的三段 Tensor 输出不变，仅语义字符串导出增加后缀。
6. 模型目录 fingerprint 能发现重分配 SID，即使语义 token 集合相同；老 checkpoint 没有 fingerprint 时只能沿用旧兼容行为，调用者应自查原目录。
7. 新 sampled 路径只支持单设备；多卡设计、负例跨卡汇集及完整恢复 optimizer/scaler 状态不在本分支。原 full-CE 的既有 gpus 入口保留，但不能据此宣称新路径已完成多卡训练。
8. 外部 User LLM 使用 FP16/BF16 时，缓存向量经投影后按该 LLM 的输入 embedding dtype 对齐；投影梯度保留。两种半精度、默认/采样损失的轻量 CPU 更新已复核，不代表完整预训练 LLM 训练验证。
9. TIGER 测试时先把 labels 中的训练掩码 -100 还原为 PAD，再解码。不等长 SID 的 padding 不能直接交给 tokenizer 当词表 ID。

## 8. 真正论文复现的下一步

- HSTU：单独适配 head-shared bias、原 QK 归一化、下一事件 timestamp 输入语义、官方采样分布和评估协议。保留当前教学 baseline，不能悄悄替换并仍称所有旧输出数值不变。
- RQ-VAE：现有通用 MLP 末层带 ReLU，encoder latent/decoder 重建输出的激活并非论文精确端口；尤其带负值的文本向量不能由非负重建无损表示。此次没有悄悄改最后输出层数值契约，应在结构复现分支明确线性末层与 checkpoint 迁移。构造器的 num_emb_list/layers 也仍需要显式配置（虽然旧签名写着 None），示例已提供，不把默认空参数称为完整可用 preset。
- HLLM：选定本地 Item/User 预训练 checkpoint，固化 item embedding pooling/ID 映射，明确冻结 Item/LoRA User 是消融设置；双 LLM 联合训练另做资源与训练数据闭环，不能用本次 tiny 随机模型验证代替。
- TIGER：固定文章属性文本和 embedding 模型、统一数据切分、确认 T5X 参数数量与 Torch 配置差别、匹配真实训练日程；报告数据集、seed、多次运行误差。碰撞处理和可运行 toy 只是必要条件。

教学优先建议：HSTU 教学 baseline → RQ-VAE/SID/TIGER → 冻结 Item＋预训练 User 的 HLLM 适配。三个模式的损失和输出不是同一种“生成式”，不要强行塞进同一个模型返回值或同一个抽样开关。
