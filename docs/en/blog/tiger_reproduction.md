---
title: Reproducing TIGER
description: Generate semantic IDs, train TIGER, and retrieve items on a small dataset with torch-rechub.
---

# Reproducing TIGER in torch-rechub

TIGER names each item with a sequence of symbols, such as `<a_1><b_3><c_5>`. During training, it replaces the movies in a user's history with these names and trains T5 to generate the next movie's name one symbol at a time. During retrieval, each symbol restricts the possible continuations to names in the catalog. The completed name maps back to a movie. This sequence of symbols is the movie's *semantic ID*.

For a first run, the two commands in section 4 take a small synthetic dataset through training and test evaluation. To work with real movies, follow section 5 to encode their vectors with RQ-VAE first. The entry points are `examples/generative/run_tiger_movielens.py` for MovieLens and `examples/generative/run_tiger_amazon_books.py` for Amazon Books. Both examples use a small T5 to demonstrate encoding, training, and retrieval. Reproducing the paper's metrics also depends on the data and training configuration.

## 1. Source files

- **Model** — `torch_rechub/models/generative/tiger.py`
  - `TIGERModel` extends `transformers.T5ForConditionalGeneration` with `set_hyper(temperature)` and a temperature-scaled `ranking_loss`.
- **Data and constrained decoding** — `torch_rechub/utils/data.py`
  - `TigerSeqDataset` maps the item IDs in `inter.json` to semantic ID strings and creates leave-one-out examples for `train`, `valid`, and `test`.
  - `Trie` builds a prefix tree of all valid semantic IDs and supplies `prefix_allowed_tokens_fn` for constrained beam search.
- **Semantic ID generation** — `examples/generative/run_rqvae_amazon_books.py`
  - RQ-VAE applies residual quantization to the item embeddings and exports `semantic_ids.json`.
- **Example scripts** — `run_tiger_movielens.py` and `run_tiger_amazon_books.py` both support `train`, `test`, and synthetic data generation.

---

## 2. Data format

TIGER needs two JSON files.

- **`inter.json`** maps `{user_id: [item_id, item_id, ...]}`. Each user's item history is in chronological order. Item IDs start at 1, with `0` reserved for padding.
- **`semantic_ids.json`** maps `{item_id: ["<a_..>", "<b_..>", ...]}`. Every item in `inter.json` needs a semantic ID from RQ-VAE.

`TigerSeqDataset` constructs the leave-one-out splits as follows.

- `train` expands `items[:-2]` into `(history, next_item)` examples, one per eligible position.
- `valid` uses `items[:-2]` as the history and `items[-2]` as the target.
- `test` uses `items[:-1]` as the history and `items[-1]` as the target.

A user needs at least four interactions to produce a training example. Three interactions yield only validation and test examples.

---

## 3. Run modes

Both scripts select the stage to run with `--mode`.

| Mode | What it does |
| --- | --- |
| `generate-toy-data` | Writes a small synthetic dataset to `--data_inter_path` and `--data_indice_path`, the same paths used for training and evaluation. |
| `prepare-data` | MovieLens only. Builds `inter.json` and `movie_id_map.json` from the original `ratings.dat`. |
| `train` | Adds semantic ID tokens, calls `resize_token_embeddings`, trains a randomly initialized T5 from scratch, and saves the tokenizer, configuration, and model to `--output_dir`. |
| `test` | Loads the checkpoint from `--ckpt_path` (defaults to `--output_dir`), runs constrained beam search, and reports hit@k and ndcg@k. |
| `all` | Reads the prepared dataset, then runs `train` and `test` in sequence. This is the default. |

By default, the scripts build a vocabulary locally from the semantic IDs and initialize a small T5 with random weights. If you supply `--base_model`, the scripts read its architecture configuration and tokenizer but still train the weights from scratch. Before training, they register segments such as `<a_1>` as individual tokens and resize the embedding table.

Testing uses the vocabulary, configuration, and weights saved during training. The scripts check the item-to-semantic-ID mapping and vocabulary size. Retrain TIGER if you change the mapping. If you trained a codebook with the older RQ-VAE quantization and gradient implementation, retrain the codebook, export fresh codes, and then train TIGER.

## 4. Train and evaluate on a small dataset

Run `uv sync --extra generative` from the repository root. The default semantic vocabulary is built locally, so the following toy example can train and evaluate offline. If you explicitly select a T5Tokenizer that requires SentencePiece, also install `sentencepiece` and make the tokenizer files available.

```bash
cd examples/generative

# Generate the example data once. Reuse these files for subsequent training runs.
python run_tiger_movielens.py --mode generate-toy-data --data_inter_path ./tmp/tiger-toy/ml/inter.json --data_indice_path ./tmp/tiger-toy/ml/semantic_ids.json --toy_num_users 16 --toy_num_items 20

# Train from the same paths, then reload the checkpoint for evaluation.
python run_tiger_movielens.py --mode all --data_inter_path ./tmp/tiger-toy/ml/inter.json --data_indice_path ./tmp/tiger-toy/ml/semantic_ids.json --output_dir ./tmp/tiger-toy/ml/ckpt --epochs 1 --per_device_batch_size 4 --num_beams 4 --metrics hit@1,ndcg@4 --test_batch_size 2 --num_workers 0
```

Run the first command once. The script reports an error if the data files already exist. For subsequent runs, use only the second command. After training it prints the model's save location, evaluates four candidates per test user, and prints Hit@1 and NDCG@4. To report Hit@10, increase `--num_beams` to at least 10.

This MovieLens toy configuration completed training, checkpoint loading, and evaluation on an RTX 5080 (16 GB). Peak memory allocated to PyTorch tensors was about 106 MiB for the offline example with 16 users and 20 items.

For the Amazon Books example, run `run_tiger_amazon_books.py` with `--mode generate-toy-data` and then `--mode all` to train and evaluate. Give it its own data paths and output directory.

## 5. Train TIGER on MovieLens-1M

With real data, the interaction file and semantic ID table must refer to the same movies. If item `12` in the interactions is movie A, the semantic ID for `12` must come from movie A's vector. Prepare the two files in this order.

1. **Build the interaction sequences.**

   ```bash
   python run_tiger_movielens.py --mode prepare-data \
       --ratings_path ./data/ml-1m/ratings.dat \
       --vocab_path ./data/ml-1m/processed/vocab.pkl \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --min_seq_len 5 --max_his_len 20
   ```

   This step sorts interactions by timestamp, filters out users with too few interactions, and writes `movie_id_map.json`. Supply `--vocab_path` to reuse HSTU/HLLM token IDs. Without it, the script independently remaps the items to consecutive IDs starting at 1.

2. **Generate semantic IDs.** Obtain the movie vectors from HLLM preprocessing with the same MovieLens vocabulary. Run RQ-VAE from the working directory used in step 1. Row 0 is PAD, and each remaining row corresponds to the same token ID in both steps.

   ```bash
   python run_rqvae_amazon_books.py --data_path ./data/ml-1m/processed/item_embeddings_tinyllama.pt --model_path ./data/ml-1m/tiger --padding_idx 0 --num_workers 0 --device cuda:0
   ```

   The script writes `semantic_ids.json` to `--model_path`. Its keys should match the item IDs in the interaction data.

   Before training, use the snippet below to find movies in the interactions that have no semantic ID. This checks that the IDs exist. Using the same vocabulary for both files is what ensures that each movie gets the right vector.

   ```python
   import json

   with open("./data/ml-1m/tiger/inter.json", encoding="utf-8") as f:
       interactions = json.load(f)
   with open("./data/ml-1m/tiger/semantic_ids.json", encoding="utf-8") as f:
       semantic_ids = json.load(f)

   referenced = {str(item) for seq in interactions.values() for item in seq}
   missing = sorted(referenced - set(semantic_ids))
   assert not missing, f"semantic_ids is missing {len(missing)} items, including {missing[:10]}"
   ```

3. **Train and test.**

   ```bash
   python run_tiger_movielens.py --mode train \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --data_indice_path ./data/ml-1m/tiger/semantic_ids.json \
       --output_dir ./ckpt/tiger_ml
   python run_tiger_movielens.py --mode test --ckpt_path ./ckpt/tiger_ml \
       --data_inter_path ./data/ml-1m/tiger/inter.json \
       --data_indice_path ./data/ml-1m/tiger/semantic_ids.json
   ```

For real Amazon Books data, follow the same sequence of stages: use `run_rqvae_amazon_books.py` to produce `semantic_ids.json`, prepare `inter.json`, and then run `--mode train` and `--mode test`.

---

## 6. Evaluation

In `test` mode, the script runs constrained beam search for each test example.

- `Trie` supplies `prefix_allowed_tokens_fn` so each decoding step can only produce a next token that occurs in a valid semantic ID.
- `--num_beams` also sets `num_return_sequences`, the number of top candidates returned. With `--filter_items`, generated sequences outside the valid item set receive a very low score.
- The script reports leave-one-out `hit@k` and `ndcg@k`. Each user has one ground-truth item, so IDCG equals 1. Set the cutoffs with `--metrics hit@1,hit@5,hit@10,ndcg@5,ndcg@10`.

---

## 7. Troubleshooting

- A remote `--base_model` requires a cached copy or access to the model repository. For offline runs, use the default semantic vocabulary or a local directory with the model configuration.
- If `TigerSeqDataset._remap_items()` raises `KeyError`, look up the missing item ID from the error in `semantic_ids.json`. If many IDs are missing, check that the RQ-VAE input vectors and the interaction file use the same vocabulary.
- **The model keeps generating subwords, or the hit rate is unexpectedly low.** Check that training called both `add_tokens` and `resize_token_embeddings`. Look for `Added N semantic-id tokens` in the log and confirm that `test` loads the vocabulary from the `--output_dir` saved during training.
