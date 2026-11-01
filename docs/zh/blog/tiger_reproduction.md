---
title: TIGER 复现说明
description: TIGER 生成式推荐模型在 torch-rechub 中的运行说明，包括语义 ID 流水线、运行模式与 toy 训练流程
---

# TIGER 在 torch-rechub 中的复现说明

本文件说明当前 `torch-rechub` 中 TIGER（Transformer Index for GEnerative Recommenders）的实现与运行方式。TIGER 把"预测下一个 item"建模成"生成下一个 item 的语义 ID"的序列到序列任务：每个 item 先由 RQ-VAE 量化成一串 codebook token（语义 ID，如 `<a_1><b_3><c_5>`），再用 T5 自回归地生成下一个 item 的语义 ID，并通过前缀受限的 beam search 约束到合法 item 上。

这里复现 RQ-VAE 语义 ID、T5 自回归训练与合法候选受限解码的基本流程，使用 Hugging Face 实现以便单卡运行；并非官方训练环境或论文指标的完整复刻。先用小数据跑通，再按显存调整 batch size、AMP 和梯度累积。RQ-VAE 量化与梯度逻辑修正前的权重应重新训练，并重新导出语义 ID 后训练 TIGER，避免混用两套编码。

示例脚本按数据集拆分，与 HSTU / HLLM 保持一致的"一数据集一脚本"风格：

- `examples/generative/run_tiger_movielens.py`
- `examples/generative/run_tiger_amazon_books.py`

---

## 1. 模块划分

- **模型主体**：`torch_rechub/models/generative/tiger.py`
  - `TIGERModel`：继承自 `transformers.T5ForConditionalGeneration`，新增 `set_hyper(temperature)` 与温度缩放的 `ranking_loss`。
- **数据与受限解码**：`torch_rechub/utils/data.py`
  - `TigerSeqDataset`：把 `inter.json` 中的 item id 序列映射成语义 ID 字符串，并按 `train` / `valid` / `test` 三种模式做 leave-one-out 切分。
  - `Trie`：根据所有合法语义 ID 构建前缀树，生成 `prefix_allowed_tokens_fn` 用于受限 beam search。
- **语义 ID 生成**：`examples/generative/run_rqvae_amazon_books.py`
  - 用 RQ-VAE 对 item embedding 做残差量化，导出 `semantic_ids.json`。
- **示例脚本**：`run_tiger_movielens.py` / `run_tiger_amazon_books.py`，各自实现 `train` / `test` 与 toy 数据生成。

---

## 2. 数据格式

TIGER 需要两个 JSON 文件：

- **`inter.json`**：`{user_id: [item_id, item_id, ...]}`，每个用户按时间排序的 item id 历史。item id 从 1 开始，`0` 保留给 padding。
- **`semantic_ids.json`**：`{item_id: ["<a_..>", "<b_..>", ...]}`，`inter.json` 中出现的每个 item id 都要有对应的 RQ-VAE 语义 ID。

`TigerSeqDataset` 的切分约定（leave-one-out）：

- `train`：对历史 `items[:-2]` 逐位置展开成多个 `(history, next_item)` 样本。
- `valid`：以 `items[:-2]` 为历史，`items[-2]` 为标签。
- `test`：以 `items[:-1]` 为历史，`items[-1]` 为标签。

因此每个用户至少需要 4 个交互才能产生至少一个训练样本；3 个交互只能构造 valid/test 样本。

---

## 3. 运行模式

两个脚本都用 `--mode` 控制运行阶段：

| mode | 说明 |
| --- | --- |
| `generate-toy-data` | 生成一份小规模合成数据，写到 `--data_inter_path` / `--data_indice_path`（与读取路径完全一致） |
| `prepare-data` | 仅 MovieLens：从真实 `ratings.dat` 构建 `inter.json` 和 `movie_id_map.json` |
| `train` | 加入语义 ID token、`resize_token_embeddings`、从头训练 T5（随机初始化，不加载预训练权重），保存 tokenizer / config / 模型到 `--output_dir` |
| `test` | 从 `--ckpt_path`（默认回退到 `--output_dir`）加载，做受限 beam search 并报告 hit@k / ndcg@k |
| `all` | 依次执行 `generate-toy-data` → `train` → `test`（默认） |

关键实现细节：

- **从头训练，不加载预训练权重**：TIGER 论文中 T5 编码器-解码器是随机初始化、从头训练的（语义 ID 词表不是自然语言，预训练的 NL 权重无意义）。`train()` 用 `TIGERModel(config)` 从 `--base_model` 的 config 构建架构并随机初始化，`--base_model` 只提供架构/config 与 tokenizer，不加载预训练权重。
- **训练前必须加入语义 ID token**：`train()` 会 `tokenizer.add_tokens(dataset.get_new_tokens())` 后再 `resize_token_embeddings`，否则 `<a_1>` 这类 token 会被 T5 切成子词，训练无意义。
- **生成与读取路径一致**：`generate-toy-data` 直接写到数据集读取的同一路径，避免"生成文件名 ≠ 读取文件名"。
- **test 从 checkpoint 加载词表与权重**：用 checkpoint 自身的 config 加载模型（避免把扩词表后的 `vocab_size` 传进 `from_pretrained` 触发 embedding 尺寸不匹配），再按 tokenizer 对齐；若 checkpoint 缺少某些语义 ID token，会自动补齐并 resize 后再评估。

---

## 4. Toy 快速跑通

先安装生成式模型 extra：

```bash
pip install -e ".[generative]"
pip install sentencepiece
```

当前 `generative` extra 尚未声明 `sentencepiece`，而 `T5Tokenizer` 需要它，因此 TIGER 需单独安装。

Toy 模式不需要外部推荐数据集，但仍需从 `--base_model` 加载 T5 的 tokenizer 与 config。以下命令显式使用独立 toy 路径，CPU 也可做小规模端到端验证：

```bash
cd examples/generative

# MovieLens 形态的合成数据
python run_tiger_movielens.py --mode all \
    --data_inter_path ./tmp/tiger-toy/ml/inter.json \
    --data_indice_path ./tmp/tiger-toy/ml/semantic_ids.json \
    --output_dir ./tmp/tiger-toy/ml/ckpt \
    --toy_num_users 16 --toy_num_items 20 \
    --epochs 2 --per_device_batch_size 4 \
    --num_beams 4 --test_batch_size 2 --num_workers 0

# Amazon-Books 的内置 toy 数据
python run_tiger_amazon_books.py --mode all \
    --data_inter_path ./tmp/tiger-toy/amazon/inter.json \
    --data_indice_path ./tmp/tiger-toy/amazon/semantic_ids.json \
    --output_dir ./tmp/tiger-toy/amazon/ckpt \
    --epochs 5 --per_device_batch_size 4 \
    --num_beams 4 --test_batch_size 2 --num_workers 0
```

正常会看到 `Added N semantic-id tokens` → `Model saved to ...` → `Test results: {...}` 的输出。

> **数据覆盖风险**：`generate-toy-data` 和默认的 `all` 都会无条件重写 `--data_inter_path` 与 `--data_indice_path`。绝不要把 toy 命令指向真实数据；默认 mode 也是 `all`，真实训练必须显式写 `--mode train` 或 `--mode test`。

> `t5-small` 与 `google-t5/t5-small` 都可能访问 HuggingFace Hub。离线运行必须把 tokenizer/config 预先缓存好，或将 `--base_model` 指向包含这些文件的本地目录；只改成规范仓库名不能解决离线问题。

---

## 5. 真实数据流程（MovieLens-1M）

真实数据需要先得到与 `inter.json` 对齐的语义 ID，整体是 RQ-VAE → TIGER 两段式流水线：

1. **构建交互序列**：

   ```bash
   python run_tiger_movielens.py --mode prepare-data \
       --ratings_path ./data/ml-1m/ratings.dat \
       --vocab_path ./data/ml-1m/processed/vocab.pkl \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --min_seq_len 5 --max_his_len 20
   ```

   这一步按时间戳排序、过滤交互过少的用户，并额外写出 `movie_id_map.json`。传入 `--vocab_path` 时会复用 HSTU/HLLM 的 token id；不传时才会独立重映射成连续的 1-based item id。

2. **生成语义 ID**：为同一批 item id 准备 item embedding（例如 HLLM 预处理产出的文本 embedding），用 `run_rqvae_amazon_books.py` 风格的 RQ-VAE 训练并导出 `semantic_ids.json`。**RQ-VAE 输出必须按第 1 步相同的 item id 作为 key**，否则 `inter.json` 与 `semantic_ids.json` 对不上。

   通用 RQ-VAE 示例虽以 Amazon 命名，也可接收其他数据集的 item embedding。训练时打乱样本，导出时使用 `shuffle=False` 保持行号顺序，并删除 PAD 的语义 ID。仍需保证 embedding 第 `i` 行就是第 1 步的 token id `i`。

   至少应在训练前检查覆盖范围：

   ```python
   import json

   with open("./data/ml-1m/tiger/inter.json", encoding="utf-8") as f:
       interactions = json.load(f)
   with open("./data/ml-1m/tiger/semantic_ids.json", encoding="utf-8") as f:
       semantic_ids = json.load(f)

   referenced = {str(item) for seq in interactions.values() for item in seq}
   missing = sorted(referenced - set(semantic_ids))
   assert not missing, f"semantic_ids 缺少 {len(missing)} 个 item，例如 {missing[:10]}"
   ```

3. **训练与测试**：

   ```bash
   python run_tiger_movielens.py --mode train \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --data_indice_path ./data/ml-1m/tiger/semantic_ids.json \
       --output_dir ./ckpt/tiger_ml
   python run_tiger_movielens.py --mode test --ckpt_path ./ckpt/tiger_ml \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --data_indice_path ./data/ml-1m/tiger/semantic_ids.json
   ```

Amazon-Books 的真实数据流程相同：先用 `run_rqvae_amazon_books.py` 得到 `semantic_ids.json`，再准备 `inter.json`，最后 `--mode train` / `--mode test`。

---

## 6. 评估

`test` 阶段对每个测试样本做受限 beam search：

- 用 `Trie` 构建 `prefix_allowed_tokens_fn`，使每一步只允许生成语义 ID 表中合法的下一个 token。
- `--num_beams` 同时作为 `num_return_sequences`，得到 top-N 候选；`--filter_items` 会把不在合法 item 集合里的生成结果打到极低分。
- 指标为 leave-one-out 的 `hit@k` 与 `ndcg@k`（每个用户只有一个 ground-truth，因此 IDCG=1）。通过 `--metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10` 配置。

---

## 7. 常见问题

- **`OSError: We couldn't connect to 'https://huggingface.co'`**：先联网缓存 tokenizer/config，或将 `--base_model` 指向对应本地目录；规范仓库名本身不保证离线可用。
- **`KeyError` 出现在 `TigerSeqDataset._remap_items()`**：`inter.json` 引用的 item 没有语义 ID。运行上面的覆盖检查，并同时核实 RQ-VAE 行号与 item id 的对齐关系。
- **`Trainer.__init__() got an unexpected keyword argument 'tokenizer'`**：`transformers>=5` 把 `tokenizer` 参数改名为 `processing_class`，脚本已按签名自动适配，无需手动改。
- **`'TIGERModel' object has no attribute 'model_parallel'`**：旧版 T5 的 model-parallel 守卫在 `transformers>=5` 不再初始化，`TIGERModel.__init__` 已补 `model_parallel=False` / `device_map=None`，多卡 DataParallel 也能跑。
- **生成结果总是子词、命中率异常低**：确认训练时确实执行了 `add_tokens` + `resize_token_embeddings`（日志里有 `Added N semantic-id tokens`），且 `test` 是从训练保存的 `--output_dir` 加载词表。
