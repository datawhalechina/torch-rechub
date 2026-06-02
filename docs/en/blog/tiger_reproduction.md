---
title: TIGER Reproduction Notes
description: How TIGER is run in torch-rechub, including the semantic-ID pipeline, run modes, and the toy training workflow
---

## TIGER in torch-rechub

This document describes how TIGER (Transformer Index for GEnerative Recommenders) is implemented and run in `torch-rechub`. TIGER frames "predict the next item" as a sequence-to-sequence task of "generate the next item's semantic ID": each item is first quantized by RQ-VAE into a tuple of codebook tokens (a *semantic ID*, e.g. `<a_1><b_3><c_5>`), then T5 autoregressively generates the next item's semantic ID, constrained to legal items via prefix-restricted beam search.

The example scripts follow the same "one script per dataset" layout as HSTU / HLLM:

- `examples/generative/run_tiger_movielens.py`
- `examples/generative/run_tiger_amazon_books.py`

---

## 1. Module Layout

- **Model**: `torch_rechub/models/generative/tiger.py`
  - `TIGERModel`: subclasses `transformers.T5ForConditionalGeneration`, adding `set_hyper(temperature)` and a temperature-scaled `ranking_loss`.
- **Dataset & constrained decoding**: `torch_rechub/utils/data.py`
  - `TigerSeqDataset`: maps item-id sequences in `inter.json` to semantic-ID strings and applies a leave-one-out split for `train` / `valid` / `test`.
  - `Trie`: builds a prefix tree over all legal semantic IDs and yields a `prefix_allowed_tokens_fn` for constrained beam search.
- **Semantic-ID generation**: `examples/generative/run_rqvae_amazon_books.py`
  - Trains an RQ-VAE over item embeddings and exports `semantic_ids.json`.
- **Example scripts**: `run_tiger_movielens.py` / `run_tiger_amazon_books.py`, each implementing `train` / `test` and toy-data generation.

---

## 2. Data Format

TIGER needs two JSON files:

- **`inter.json`**: `{user_id: [item_id, item_id, ...]}`, each user's chronologically ordered item-id history. Item ids are 1-based; `0` is reserved for padding.
- **`semantic_ids.json`**: `{item_id: ["<a_..>", "<b_..>", ...]}`, a semantic ID for every item id that appears in `inter.json`.

`TigerSeqDataset` leave-one-out split:

- `train`: expand the history `items[:-2]` into multiple `(history, next_item)` samples.
- `valid`: history is `items[:-2]`, label is `items[-2]`.
- `test`: history is `items[:-1]`, label is `items[-1]`.

So each user needs at least 3 interactions to produce a training sample.

---

## 3. Run Modes

Both scripts use `--mode` to select the stage(s):

| mode | Description |
| --- | --- |
| `generate-toy-data` | Write a small synthetic dataset to `--data_inter_path` / `--data_indice_path` (the exact paths the loader reads) |
| `prepare-data` | MovieLens only: build `inter.json` and `movie_id_map.json` from the real `ratings.dat` |
| `train` | Add semantic-ID tokens, `resize_token_embeddings`, fine-tune T5, save tokenizer / config / model to `--output_dir` |
| `test` | Load from `--ckpt_path` (defaults to `--output_dir`), run constrained beam search, report hit@k / ndcg@k |
| `all` | Run `generate-toy-data` → `train` → `test` (default) |

Key implementation details:

- **Semantic-ID tokens must be added before training**: `train()` calls `tokenizer.add_tokens(dataset.get_new_tokens())` and then `resize_token_embeddings`; otherwise tokens like `<a_1>` are split into sub-words and training is meaningless.
- **Generated and read paths are identical**: `generate-toy-data` writes to the same paths the dataset reads, avoiding a "generated filename ≠ read filename" mismatch.
- **`test` loads the vocabulary from the checkpoint**: tokenizer / config / model all come from `--ckpt_path`, keeping the semantic-ID vocabulary consistent with training. If the checkpoint is missing some semantic-ID tokens, they are added and the embedding table is resized before evaluation.

---

## 4. Quick Toy Run

No external data is required; this runs end to end on CPU:

```bash
cd examples/generative

# Synthetic MovieLens-shaped data
python run_tiger_movielens.py --mode all \
    --toy_num_users 16 --toy_num_items 20 \
    --epochs 2 --per_device_batch_size 4 \
    --num_beams 4 --test_batch_size 2 --num_workers 0

# Built-in Amazon-Books toy data
python run_tiger_amazon_books.py --mode all \
    --epochs 5 --per_device_batch_size 4 \
    --num_beams 4 --test_batch_size 2 --num_workers 0
```

You should see `Added N semantic-id tokens` → `Model saved to ...` → `Test results: {...}`.

> Offline / no access to the HuggingFace Hub: the legacy alias `t5-small` may not resolve, so use the canonical repo id: `--base_model google-t5/t5-small`.

---

## 5. Real-Data Pipeline (MovieLens-1M)

Real data needs semantic IDs aligned with `inter.json`, so it is a two-stage RQ-VAE → TIGER pipeline:

1. **Build interaction sequences**:

   ```bash
   python run_tiger_movielens.py --mode prepare-data \
       --ratings_path ./data/ml-1m/ratings.dat \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --min_seq_len 5 --max_his_len 20
   ```

   This orders interactions by timestamp, filters users with too few interactions, remaps movie ids to contiguous 1-based item ids, and also writes `movie_id_map.json`.

2. **Generate semantic IDs**: prepare item embeddings for the same item ids (e.g. the text/ID embeddings produced by HLLM preprocessing), then train an `run_rqvae_amazon_books.py`-style RQ-VAE and export `semantic_ids.json`. **The RQ-VAE output must be keyed by the same item ids as step 1**, otherwise `inter.json` and `semantic_ids.json` will not line up.

3. **Train and test**:

   ```bash
   python run_tiger_movielens.py --mode train \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --data_indice_path ./data/ml-1m/tiger/semantic_ids.json \
       --output_dir ./ckpt/tiger_ml
   python run_tiger_movielens.py --mode test --ckpt_path ./ckpt/tiger_ml \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --data_indice_path ./data/ml-1m/tiger/semantic_ids.json
   ```

The Amazon-Books real-data flow is identical: produce `semantic_ids.json` with `run_rqvae_amazon_books.py`, prepare `inter.json`, then `--mode train` / `--mode test`.

---

## 6. Evaluation

The `test` stage runs constrained beam search per test sample:

- A `Trie` builds the `prefix_allowed_tokens_fn` so each step only allows the next token of a legal semantic ID.
- `--num_beams` is also used as `num_return_sequences` to obtain top-N candidates; `--filter_items` pushes any generated result outside the legal item set to a very low score.
- Metrics are leave-one-out `hit@k` and `ndcg@k` (each user has a single ground truth, so IDCG=1). Configure with `--metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10`.

---

## 7. Troubleshooting

- **`OSError: We couldn't connect to 'https://huggingface.co'`**: offline, use `--base_model google-t5/t5-small` (canonical repo id) and make sure the weights are cached locally.
- **`Trainer.__init__() got an unexpected keyword argument 'tokenizer'`**: `transformers>=5` renamed the `tokenizer` argument to `processing_class`; the script auto-adapts via the signature, no manual change needed.
- **`'TIGERModel' object has no attribute 'model_parallel'`**: the legacy T5 model-parallel guards are no longer initialized in `transformers>=5`; `TIGERModel.__init__` now sets `model_parallel=False` / `device_map=None`, so multi-GPU DataParallel works too.
- **Outputs are always sub-words / accuracy is abnormally low**: confirm that training actually ran `add_tokens` + `resize_token_embeddings` (the log shows `Added N semantic-id tokens`) and that `test` loads the vocabulary from the `--output_dir` saved during training.
