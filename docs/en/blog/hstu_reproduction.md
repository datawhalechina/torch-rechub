---
title: Reproducing HSTU
description: How torch-rechub implements Meta's HSTU layer and trains it for next-item prediction.
---

# Reproducing HSTU in torch-rechub

HSTU reads a user's viewing history in chronological order and predicts the next movie. It can use both the order of the movies and the time between views. For a first run, follow the MovieLens commands below and look for Hit@K and NDCG@K after training. To read the code alongside the paper, start with the three equations in section 2.

This example uses next-item prediction as a single task to reproduce the core HSTU layer. Its treatment of position and time, data split, and training budget differ from those in the original paper. Early versions of the project used a different output structure for the core layer. Retrain models from those experiments when upgrading.

## 1. Source files

The HSTU implementation uses the following modules.

- **Model** — `torch_rechub/models/generative/hstu.py`
  - `HSTUModel` contains the item token embeddings, position and time input embeddings, `HSTUBlock`, and output projection.
- **Layer and block** — `torch_rechub/basic/layers.py`
  - `HSTULayer` implements one sequential transduction unit from equations 2–4 of the HSTU paper.
  - `HSTUBlock` stacks layers with `x = x + HSTULayer(x)`.
- **HSTU and HLLM utilities** — `torch_rechub/utils/hstu_utils.py`
  - `RelativeBucketedTimeAndPositionBias` supplies the `rab^{p,t}` bias for HSTU attention scores.
  - `RelPosBias` is an older relative position bias used by HLLM and retained for older experiments. The current `HSTUModel` uses `RelativeBucketedTimeAndPositionBias`.
  - `VocabMask` masks PAD or invalid items during inference and ranking.
- **Data and training**
  - `torch_rechub/utils/data.py` defines `SeqDataset` and `SequenceDataGenerator`.
  - `torch_rechub/trainers/seq_trainer.py` defines `SeqTrainer`.
  - `examples/generative/run_hstu_movielens.py` and `examples/generative/run_hstu_amazon_books.py` provide training and ranking evaluation examples.

### Train for one epoch on MovieLens

From the repository root, run preprocessing to create the vocabulary and training data in `processed/`, then start training.

```bash
pip install -e .

# Download missing MovieLens source files. Add --no_download if you already have them.
python examples/generative/data/ml-1m/preprocess_ml_hstu.py

# Run one epoch on the real dataset using the CPU.
mkdir -p outputs/hstu_ml
python examples/generative/run_hstu_movielens.py \
    --device cpu --epoch 1 --batch_size 16 \
    --save_dir outputs/hstu_ml
```

For one CUDA GPU, reuse the preprocessed MovieLens files with the settings below. Each batch contains eight histories of up to 32 recent interactions. Training samples 32 negatives per target and updates the parameters every two batches. After training, the script prints Hit@K and NDCG@K on the held-out items.

```bash
python examples/generative/run_hstu_movielens.py --device cuda --epoch 1 --batch_size 8 --max_seq_len 32 --loss_type sampled_softmax --num_negatives 32 --accumulation_steps 2 --amp --save_dir outputs/hstu_ml
```

These settings were tested on an RTX 5080 (16 GB) with MovieLens-1M users 1–32 and up to 24 recent interactions per user. The subset contained 578 movies and 32 user sequences in each of the training, validation, and test splits. Peak memory allocated to PyTorch tensors was about 20 MiB across training and evaluation. This excludes the driver, the CUDA context, and memory held in reserve by the allocator. Measure the full dataset with its actual vocabulary and sequence length.

For Amazon Books, prepare the input with `examples/generative/data/amazon-books/preprocess_amazon_books.py` and train with `examples/generative/run_hstu_amazon_books.py`. The training script accepts the same single-GPU sampling options. With the default cross-entropy loss, the scoring tensor grows with `[batch size, sequence length, vocabulary size]`. Preprocessing writes the vocabulary and all three splits, so choose a separate output directory for a new experiment.

---

## 2. HSTULayer and equations 2–4

### 2.1 Equation 2 — Joint UVQK projection

`HSTULayer` applies `LayerNorm` to the input and uses one linear layer to produce `Q/K/U/V` together.

```python
proj_out = F.silu(self.proj1(self.norm_in(x)))
```

Apply `SiLU` to the entire projection before splitting it into four parts. This gives `U`, `V`, `Q`, and `K` the same nonlinearity.

### 2.2 Equation 3 — Add `rab^{p,t}` to the attention scores

The layer computes attention as follows.

```python
scores = Q @ K.transpose(-2, -1)
scores = scores + rab(time_diffs, seq_len)
attn_weights = silu(scores) / max_seq_len
AV = attn_weights @ V
```

Two movies watched a day apart can receive a different bias from the same movies watched a year apart. The `rab` term maps differences in position and time to biases and adds them to the attention scores. All attention heads share these bias parameters. The layer then computes the weights with `silu(scores) / N`. The mask lets each position attend only to valid items at that position or earlier, skipping padding.

`RelativeBucketedTimeAndPositionBias` expects `time_diffs` to equal `query_time - timestamp[i]`. When the layer takes pairwise differences, the common query time cancels, leaving the time difference between events.

### 2.3 Equation 4 — Gate and project the output

The layer projects the gated attention output as follows.

```python
gated = LayerNorm(AV) * U
output = f2(gated)
```

`U` has already passed through `SiLU`, so it can gate the normalized attention output directly. The product is the sole input to `proj2`.

### 2.4 Residual connection

`HSTUBlock` adds the residual outside the layer.

```python
for layer in self.layers:
    x = x + layer(x, padding_mask=padding_mask, time_diffs=time_diffs)
```

The previous layer's input passes directly to the next layer, and `HSTULayer` learns the change to add to it.

---

## 3. `rab^{p,t}` and time features

### 3.1 `RelativeBucketedTimeAndPositionBias`

`RelativeBucketedTimeAndPositionBias` has two sets of learned parameters.

- `pos_w` stores the per-head bias for relative positions `i - j` and has `2 * max_seq_len - 1` entries.
- `ts_w` stores the per-head bias for relative time buckets and has `num_time_buckets + 1` entries.

The layer assigns time differences to buckets as follows.

```text
dt = abs(time_diffs[i] - time_diffs[j]) / 60.0
bucket = sqrt(dt) or log(dt)
bucket = clamp(bucket / time_bucket_divisor, 0, num_time_buckets)
```

With `time_diffs=None`, `rab` provides a position-only bias of shape `(1, H, L, L)`. With `time_diffs`, its shape is `(B, H, L, L)`. The current `HSTULayer` creates this module with `n_heads=1` and broadcasts its bias to all attention heads.

### 3.2 The other position bias in the source

`hstu_utils.py` also contains `RelPosBias` for HLLM and older experiments. For the bias used in HSTU attention, see `RelativeBucketedTimeAndPositionBias`.

## 4. From a viewing history to candidate scores

`HSTUModel` embeds a sequence of item IDs, passes the vectors through `HSTUBlock`, and scores every vocabulary item at each sequence position.

From input to output, the model uses these components in order.

- `token_embedding(vocab_size, d_model, padding_idx=0)`.
- Absolute `position_embedding(max_seq_len, d_model)`.
- Optional input `time_embedding(num_time_buckets, d_model)`, controlled by `use_time_embedding`.
- `HSTUBlock(..., num_time_buckets, time_bucket_fn, time_bucket_divisor)`.
- By default, an output projection tied to the token embeddings: `F.linear(hstu_output, token_embedding.weight, output_bias)`.

Time and position enter the model at two points.

- In equation 3 of the core HSTU layer, `rab^{p,t}` adds relative position and time information to the attention scores.
- The `HSTUModel` wrapper keeps absolute position embeddings and optional input time embeddings for compatibility with the project's data interface and existing experiments.

Token `0` is PAD. The model explicitly zeros PAD rows after input embedding and after the HSTU output so that position or time embeddings at padding positions cannot carry a signal.

---

## 5. Data and training conventions

### 5.1 Time differences at each position

The preprocessing scripts generate `seq_time_diffs` as follows.

```text
time_diffs[i] = query_time - timestamp[i]
```

Here `query_time` is usually the timestamp of the last interaction in the current history.

```text
timestamps  = [100, 200, 300, 400]
query_time  = 400
time_diffs  = [300, 200, 100, 0]
```

Both entries use the same query time. It cancels when the layer takes their difference, so the model can recover the time between the two interactions.

### 5.2 Dataset format

`SequenceDataGenerator` uses four-tuples.

```text
(seq_tokens, seq_positions, seq_time_diffs, targets)
```

The data interface retains `seq_positions` for compatibility. The current `HSTUModel` generates position indices from the sequence length in its forward pass, and `SeqTrainer` does not read `seq_positions` from the batch.

### Per-user sequence sampling on MovieLens

The MovieLens preprocessor defaults to `user_sequence`. It keeps the latest training sequence for each user with a long enough history and computes next-token loss at every valid position.

For each user, the last interaction is the test target and the second-to-last is the validation target. For a history of `1 2 3 4 5 6 7 8`, `user_sequence` uses `1 2 3 4 5` as training input and `2 3 4 5 6` as the position-wise labels. The validation target is `7` and the test target is `8`. With a maximum input length of 200, the training window uses the last 201 training interactions at most. This adapts per-user sampling to the project's separate validation set, left padding, and attention bias based on the times of the observed events. These choices differ from the full official training protocol.

Run these commands from the repository root. The training example defaults to 101 epochs.

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

`preprocessing_config.json` records the sampling mode, maximum length, and number of held-out interactions.

Each epoch visits every eligible user once. Validation selects the model weights during training, and the logs record optimizer updates and elapsed time. The test set measures final performance.

### 5.3 Training objective

`SeqTrainer` computes next-token cross-entropy across the sequence.

```text
logits[:, i, :] -> seq_tokens[:, i + 1]
logits[:, -1, :] -> targets
```

The loss excludes PAD token `0` via `ignore_index=0`. On validation and test data, `evaluate` returns the average loss and top-1 accuracy on the final held-out target. The example scripts also report HR@K and NDCG@K for ranking.

---

## 6. Comparison with Meta's implementation

Three operations connect the code to the paper. The `SiLU` after the joint projection implements equation 2. The relative time and position bias on the scores implements equation 3. The output projection after `LayerNorm(AV) * U` implements equation 4. The enclosing `HSTUBlock` adds the residual.

Several differences outside the core layer affect comparisons. The model keeps absolute input position embeddings and optional input time embeddings. The attention bias uses the time differences between the observed interactions. MovieLens supplies one training sequence per user and a separate validation target. Both training and evaluation use only next-item prediction. Record these choices and the training budget when reporting results.
