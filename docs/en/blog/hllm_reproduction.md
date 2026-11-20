---
title: Reproducing HLLM
description: Prepare item embeddings, train the user encoder, and evaluate the single-GPU HLLM example in torch-rechub.
---

# Reproducing HLLM in torch-rechub

MovieLens records the order in which users watch movies, along with each movie's title and genres. HLLM uses both. First, a language model encodes each movie's text and saves one vector per movie. During training, a Transformer reads those vectors in the order a user watched the movies and predicts what they will watch next.

Once you have generated the item vectors, you can reuse them across experiments with different user encoders. The single-GPU example freezes the item vectors and trains the user encoder. The paper's end-to-end setup trains the item and user models jointly. Start with MovieLens below, or skip to the Amazon Books section for book data.

## 1. Source files

- Model: `torch_rechub/models/generative/hllm.py`
  - `HLLMTransformerBlock`
  - `HLLMModel`
- Shared sequence trainer: `torch_rechub/trainers/seq_trainer.py`
- MovieLens-1M
  - `examples/generative/data/ml-1m/preprocess_ml_hstu.py`
  - `examples/generative/data/ml-1m/preprocess_hllm_data.py`
  - `examples/generative/run_hllm_movielens.py`
- Amazon Books
  - `examples/generative/data/amazon-books/preprocess_amazon_books.py`
  - `examples/generative/data/amazon-books/preprocess_amazon_books_hllm.py`
  - `examples/generative/run_hllm_amazon_books.py`

Preprocessing creates `examples/generative/data/*/processed/`. It contains the generated `pkl` and embedding files.

## 2. From item text to the next movie

### 2.1 Store one vector per item

The preprocessor accepts two values for `--model_type`.

| Option | Hugging Face model | Hidden size |
| --- | --- | ---: |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 |
| `baichuan2` | `baichuan-inc/Baichuan2-7B-Chat` | 4096 |

The MovieLens prompt uses the following format.

```text
Compress the following sentence into embedding: title: {title}genres: {genres}
```

For Amazon Books, the script takes the text from `title` and `description`. It saves the hidden state of the last token to `item_embeddings_<model_type>.pt`. Row 0 is reserved for padding. The training vocabulary determines the remaining row indices.

For example, if `item_to_idx[42]` in `vocab.pkl` is 17, row 17 must contain the vector for movie 42. Swapping two rows preserves the shape of the matrix but gives the user encoder the wrong movie. Keep the vocabulary with the embeddings it was used to generate. The log reports how many items have text. Rows for items with missing text stay zero.

### 2.2 Train the user encoder on watch histories

The forward pass takes these steps.

```text
seq_tokens [B, L]
  -> look up the frozen item embeddings
  + learned absolute position embeddings
  + optional time bucket embeddings
  -> causal Transformer blocks
  -> L2 normalize hidden states
  -> hidden @ normalized_item_embeddings.T / 0.07
  -> logits [B, L, V]
```

Each block uses pre-norm multi-head self-attention, a feed-forward network, and residual connections. The causal mask hides future tokens. The padding mask hides PAD keys, and outputs at PAD positions are set to zero.

### 2.3 Turn a history into training targets

Each valid position in a training history predicts the next item.

```text
logits[:, i, :]  -> seq_tokens[:, i + 1]
logits[:, -1, :] -> held-out targets
```

The trainer skips padding labels (`0`). The default, `--loss_type nce`, scores the full vocabulary and computes the negative log-likelihood of the target. The model already applies temperature scaling, so the script passes a temperature of 1 to the loss. To score fewer candidates during training, use `--loss_type sampled_softmax` and set `--num_negatives` to the number of negatives per target. The `cross_entropy` option also scores the full vocabulary.

## 3. Installation and model cache

Run this command from the repository root.

```bash
pip install -e ".[generative]"
```

The `generative` extra installs `transformers` and `accelerate`. If the model's tokenizer reports that SentencePiece is missing, run `pip install sentencepiece` separately. This is common with Baichuan2.

The MovieLens preprocessor first checks the Hugging Face cache with `local_files_only=True` and exits if the selected language model is missing. Download the model beforehand or copy its cache to the training machine. The `--no_download` option controls dataset downloads only.

## 4. Run MovieLens-1M

Run the following commands from the repository root.

```bash
# 1. Build the sequences. Missing ratings.dat, movies.dat, and users.dat are downloaded automatically.
python examples/generative/data/ml-1m/preprocess_ml_hstu.py

# 2. Build the text mapping and align item embeddings with the token IDs.
python examples/generative/data/ml-1m/preprocess_hllm_data.py \
    --model_type tinyllama \
    --device cuda

# 3. Train on one GPU with 8 histories per batch, up to 32 items per history, and 32 negatives per target.
mkdir -p outputs/hllm_ml
python examples/generative/run_hllm_movielens.py --model_type tinyllama --device cuda --epoch 1 --batch_size 8 --max_seq_len 32 --d_model 128 --loss_type sampled_softmax --num_negatives 32 --accumulation_steps 2 --amp --save_dir outputs/hllm_ml
```

These training settings were tested on an RTX 5080 (16 GB) with MovieLens-1M users 1–32 and up to 24 of their most recent interactions per user. The subset contained 578 movies, each matched by item ID to its 2,048-dimensional TinyLlama vector. Peak memory allocated to PyTorch tensors during training and evaluation was about 38 MiB. That figure covers the user encoder on this small subset. Generating text embeddings is a separate workload.

By default, `examples/generative/data/ml-1m/processed/` should contain the following files.

```text
vocab.pkl
train_data.pkl
val_data.pkl
test_data.pkl
movie_text_map.pkl
item_embeddings_tinyllama.pt
```

MovieLens uses a chronological split for each user. The last movie is the test target, the second-to-last is the validation target, and all earlier views are used for training.

For a custom directory, pass the same `--output_dir` to both preprocessors, then point `--dataset_path` at that directory during training. The training script resolves explicit relative paths in relation to the script directory, so use an absolute path in automated jobs.

## 5. Run Amazon Books

Choose the same data source for the interaction sequences and item metadata. The default, `bytedance`, downloads ByteDance's processed files. The `raw` option uses the original Stanford SNAP files.

```bash
# 1. Build the sequences.
python examples/generative/data/amazon-books/preprocess_amazon_books.py \
    --data_source bytedance \
    --max_seq_len 200 \
    --min_seq_len 5

# 2. Build the text mapping and item embeddings.
python examples/generative/data/amazon-books/preprocess_amazon_books_hllm.py \
    --data_source bytedance \
    --model_type tinyllama \
    --device cuda

# 3. Train and evaluate.
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

The Amazon preprocessor writes the following files.

```text
vocab.pkl
train_data.pkl
val_data.pkl
test_data.pkl
item_text_map.pkl
item_embeddings_tinyllama.pt
```

Rerunning preprocessing rewrites the mappings, splits, and embeddings in the output directory. The `--overwrite` option applies only to downloaded files. To keep an existing experiment, back up its outputs or use a different `--output_dir`.

## 6. Read the training and evaluation results

The logged loss measures prediction error across valid history positions. Top-1 accuracy measures how often the held-out item ranks first for each validation or test record. The script also computes HR and NDCG at K = 10, 50, and 200. HR counts whether the target was retrieved, while NDCG also accounts for its rank. Use the same candidate catalog and the same filtering of previously seen items when comparing runs.

If GPU memory is tight, check the costs of vocabulary scoring and the user encoder separately. Full-vocabulary scoring builds a tensor of shape `[batch size, sequence length, vocabulary size]`. Sampled training reduces the last dimension. The `--d_model 128` option trains a projection from the wide cached vectors into a smaller user encoder. At evaluation time, `SeqTrainer.predict()` can search the full catalog in chunks.

## 7. Compare with the paper

This setup tests whether a fixed set of text embeddings helps predict the next interaction. To compare against ByteDance's official HLLM results, also match the two model architectures, joint training procedure, data split, negative sampling, and evaluation candidate set. The small-sample results here apply only to the configuration described above.
