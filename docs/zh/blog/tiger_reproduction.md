---
title: TIGER 复现说明
description: TIGER 生成式推荐模型在 torch-rechub 中的运行说明，包括语义 ID 流水线、运行模式与 toy 训练流程
---

# TIGER 在 torch-rechub 中的复现说明

TIGER 给每件物品起一个由几段符号组成的“名字”，例如 `<a_1><b_3><c_5>`。训练时，把用户看过的电影换成这些名字，交给 T5，让它逐段写出下一部电影的名字。推理时每写一段，就在片库里查还有哪些合法的后续片段，因此能把结果映射回电影。这串名字就是电影的语义 ID。

第一次用 TIGER，可以直接跑第 4 节的小样本，两条命令就能看到测试集结果。换成真实电影时，再按第 5 节先用 RQ-VAE 编码电影向量。MovieLens 和 Amazon Books 各有一个入口，分别是 `examples/generative/run_tiger_movielens.py` 和 `examples/generative/run_tiger_amazon_books.py`。示例使用小型 T5 验证编码、训练和检索过程，论文指标还取决于数据和训练配置。

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
| `all` | 读取已准备好的数据，依次执行 `train` 和 `test`（默认） |

默认会根据语义 ID 在本地建立词表，随机初始化小型 T5。即使显式指定 `--base_model`，脚本读取的也只是结构配置和 tokenizer，权重仍从头训练。训练前会把 `<a_1>` 这样的片段注册为单个 token，并调整嵌入表大小。

测试要使用训练时保存的词表、配置和权重。脚本会核对物品与语义 ID 的映射以及词表大小。换过映射需要重新训练 TIGER。若之前用旧版 RQ-VAE 的量化和梯度逻辑训练过码本，应重新训练码本并导出编码，再训练 TIGER。

## 4. 用小样本跑一次训练和测试

从源码目录执行 `uv sync --extra generative`。默认的语义词表在本地创建，下面的 toy 示例可离线训练和评估。显式选用依赖 SentencePiece 的 T5Tokenizer 时，再安装 `sentencepiece` 并准备对应 tokenizer 文件。

```bash
cd examples/generative

# 第一次生成示例数据。后续训练直接复用这两个文件。
python run_tiger_movielens.py --mode generate-toy-data --data_inter_path ./tmp/tiger-toy/ml/inter.json --data_indice_path ./tmp/tiger-toy/ml/semantic_ids.json --toy_num_users 16 --toy_num_items 20

# 从相同路径读取数据，训练后重新加载 checkpoint 进行评估。
python run_tiger_movielens.py --mode all --data_inter_path ./tmp/tiger-toy/ml/inter.json --data_indice_path ./tmp/tiger-toy/ml/semantic_ids.json --output_dir ./tmp/tiger-toy/ml/ckpt --epochs 1 --per_device_batch_size 4 --num_beams 4 --metrics hit@1,ndcg@4 --test_batch_size 2 --num_workers 0
```

第一条命令只需运行一次，数据文件已存在时脚本会报错。再次训练只跑第二条。训练结束会先打印模型保存位置，然后评估每个测试用户的 4 个候选并打印 Hit@1、NDCG@4。若把指标改为 Hit@10，要同时把 `--num_beams` 提到至少 10。

这组 MovieLens toy 参数在 RTX 5080（16 GB）上实际完成了训练、checkpoint 加载和评估，PyTorch 峰值张量显存约 106 MiB。该数值对应 16 个用户、20 个物品的离线示例。

Amazon Books 示例 `run_tiger_amazon_books.py` 也先显式执行 `--mode generate-toy-data`，再用 `--mode all` 训练和评估。为它指定自己的数据路径与输出目录。

## 5. 用 MovieLens-1M 训练 TIGER

使用真实数据时，要让交互记录和语义 ID 表指向同一部电影。例如，若交互文件中的 `12` 表示电影 A，编号 `12` 的语义 ID 也必须来自电影 A 的向量。按下面的顺序准备两份文件。

1. **构建交互序列**：

   ```bash
   python run_tiger_movielens.py --mode prepare-data \
       --ratings_path ./data/ml-1m/ratings.dat \
       --vocab_path ./data/ml-1m/processed/vocab.pkl \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --min_seq_len 5 --max_his_len 20
   ```

   这一步按时间戳排序、过滤交互过少的用户，并额外写出 `movie_id_map.json`。传入 `--vocab_path` 时会复用 HSTU/HLLM 的 token id；不传时才会独立重映射成连续的 1-based item id。

2. **生成语义 ID**　从同版 MovieLens 词表和 HLLM 预处理得到电影向量。沿用上一步的工作目录运行 RQ-VAE，行 0 是 PAD，其余每行对应同一 token ID。

   ```bash
   python run_rqvae_amazon_books.py --data_path ./data/ml-1m/processed/item_embeddings_tinyllama.pt --model_path ./data/ml-1m/tiger --padding_idx 0 --num_workers 0 --device cuda:0
   ```

   脚本会将 `semantic_ids.json` 写入 `--model_path`。交互表中的物品编码应与文件的 key 对应。

   训练前可以先运行下面几行，检查交互记录中是否有电影缺少语义 ID。这个检查只核对编号是否存在，电影和向量是否对应还要靠同一份词表保证。

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

- 显式指定远程 `--base_model` 时，需要已有缓存或可访问模型库。离线复现可使用默认语义词表或本地模型配置目录。
- 在 `TigerSeqDataset._remap_items()` 遇到 `KeyError` 时，先从异常中找到缺失的物品 ID，再查 `semantic_ids.json` 是否有相同的键。若缺少许多 ID，检查 RQ-VAE 输入向量和交互文件是否使用了同一份词表。
- **生成结果总是子词、命中率异常低**：确认训练时确实执行了 `add_tokens` + `resize_token_embeddings`（日志里有 `Added N semantic-id tokens`），且 `test` 是从训练保存的 `--output_dir` 加载词表。
