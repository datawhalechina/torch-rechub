---
title: Generative recommendation models
description: Next-item prediction, item embeddings and semantic IDs in Torch-RecHub
---

# Generative recommendation models

Given the movies a user has watched, which movie might they watch next? HSTU and HLLM score candidate movies. TIGER generates the next movie’s code one token at a time. This code is called a semantic ID. RQ-VAE can learn these IDs from movie embeddings.

## 1. HSTUModel

HSTU reads item IDs in chronological order. During training, the first two movies predict the third, the first three predict the fourth, and so on. Sequences are padded with `0`, and padded targets are excluded from the loss.

If interaction timestamps are available, pass a time difference for each position. The attention score includes a bias indexed by relative position and time difference. All heads share this bias in this implementation. See how the core layer corresponds to the paper in the [HSTU reproduction notes](/blog/hstu_reproduction).

### Usage

```python
import torch

from torch_rechub.models.generative import HSTUModel

model = HSTUModel(
    vocab_size=128,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=16,
    num_time_buckets=128,
    time_bucket_unit="seconds",
)

seq_tokens = torch.randint(1, 128, (2, 16))
time_diffs = torch.zeros_like(seq_tokens)  # Seconds relative to the query time
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # torch.Size([2, 16, 128])
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| vocab_size | int | Item vocabulary size, including `0` reserved for PAD | Required |
| d_model | int | Hidden dimension | 512 |
| n_heads | int | Attention heads | 8 |
| n_layers | int | Stacked HSTU layers | 4 |
| dqk | int | Query/key dimension per head | 64 |
| dv | int | Value/U dimension per head | 64 |
| max_seq_len | int | Maximum sequence length | 256 |
| dropout | float | Dropout probability | 0.1 |
| use_time_embedding | bool | Add a time-bucket embedding to the input. `time_diffs` still contributes to `rab^{p,t}` when this is off. | True |
| num_time_buckets | int | Number of buckets for time embeddings and attention bias | 128 |
| time_bucket_fn | {"sqrt", "log"} | Function used to bucket time differences | "sqrt" |
| time_bucket_divisor | float | Divisor applied after the bucket transform to adjust its range | 1.0 |
| tie_embeddings | bool | Share weights between the output projection and token embeddings | True |

## 2. HLLMModel

HLLM also reads a user’s viewing history in order. Each movie starts with a vector derived from text such as its title and genres. Compute the vectors before training, then train the user Transformer with the movie embedding table frozen.

If the vocabulary assigns a movie ID 42, `item_embeddings[42]` must contain that movie’s vector. The model scores candidates by the cosine similarity between the user’s history representation and each movie representation.

### Usage

```python
from torch_rechub.models.generative import HLLMModel

import torch

# Shape: [vocab_size, d_model]. Row i must correspond to token i.
item_embeddings = torch.load("item_embeddings_tinyllama.pt", map_location="cpu")
model = HLLMModel(
    item_embeddings=item_embeddings,
    vocab_size=item_embeddings.shape[0],
    d_model=item_embeddings.shape[1],
    n_heads=8,
    n_layers=2,
    max_seq_len=50,
    dropout=0.1,
    temperature=0.07,
)

seq_tokens = torch.randint(1, item_embeddings.shape[0], (32, 50))
time_diffs = torch.zeros_like(seq_tokens)  # Seconds
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # [32, 50, vocab_size]
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_embeddings | Tensor or str | Two-dimensional table indexed by item ID, or a path readable by `torch.load` | Required |
| vocab_size | int | Vocabulary size including PAD=0. Must equal the number of embedding rows. | Required |
| d_model | int | User encoder width. Set `project_items=True` when this differs from the input item vector width. | 512 |
| n_heads / n_layers | int | Transformer heads / layers | 8 / 4 |
| max_seq_len | int | Maximum sequence length | 256 |
| use_rel_pos_bias | bool | Use relative position bias | True |
| use_time_embedding | bool | Use time-difference embeddings | True |
| temperature | float | Temperature for cosine logits | 0.07 |

## 3. RQVAEModel

RQ-VAE takes existing item vectors and assigns a sequence of discrete codes to each item. The first codebook selects the closest vector. Each subsequent codebook encodes the remaining error. The resulting codes can be written as `<a_12><b_7><c_91>` and used as semantic IDs for TIGER.

### Generate a semantic ID for each item with RQ-VAE

```python
import os
import torch
from torch.utils.data import DataLoader

from torch_rechub.models.generative import RQVAEModel
from torch_rechub.trainers.rqvae_trainer import Trainer
from torch_rechub.utils.data import EmbDataset

dataset = EmbDataset("examples/generative/data/amazon-books/processed/item_embeddings_tinyllama.pt")
model = RQVAEModel(
    in_dim=dataset.dim,
    num_emb_list=[256, 256, 256],
    e_dim=32,
    layers=[512, 256, 128, 64],
    sk_epsilons=[0.0, 0.0, 0.0],
)

os.makedirs("saved/rqvae", exist_ok=True)
train_loader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)
trainer = Trainer(model, n_epoch=50, device="cpu", model_path="saved/rqvae", eval_step=5)
trainer.fit(train_loader)

# Keep the dataset in its original order so that each row maps to the correct item.
semantic_loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
state = torch.load("saved/rqvae/model_best_collision_rate.pth", map_location="cpu")
model.load_state_dict(state)
model.eval()
semantic_ids = model.generate_semantic_ids(
    dataset,
    semantic_loader,
    prefix=["<a_{}>", "<b_{}>", "<c_{}>"],
    use_sk=True,
    device="cpu",
)
```

The keys in `semantic_ids` are row indices in `EmbDataset`. Keep the embedding file in the same row order as the item ID mapping you will use with TIGER. Leave shuffling off when generating IDs.

### Key parameters

| Parameter | Description |
| --- | --- |
| in_dim | Original item embedding dimension |
| num_emb_list | Codebook size at each residual quantization level. Its length determines the number of semantic ID tokens. |
| e_dim | Dimension of the quantized representation |
| layers | Encoder hidden layers. The decoder uses the layers in reverse order. |
| loss_type | Reconstruction loss, `mse` or `l1` |
| sk_epsilons / sk_iters | Sinkhorn assignment settings. Supply one epsilon per codebook level. |

## 4. TIGERModel

TIGER (Transformer Index for GEnerative Recommenders) represents the movies a user watched as semantic IDs. T5 reads the resulting history and predicts the tokens of the next movie’s ID in order.

Suppose the catalog contains only two valid continuations of `<a_1>`: `<b_3>` and `<b_7>`. Once the decoder generates `<a_1>`, it can choose between those two tokens. The example inserts all catalog IDs into a prefix tree and looks up valid next tokens at each decoding step.

### Usage

This example adds three semantic ID tokens and constructs a small T5 model. To train and evaluate it, follow the [TIGER reproduction notes](/blog/tiger_reproduction). Start with synthetic data and then try MovieLens or Amazon Books.

```python
from transformers import T5Config
from torch_rechub.models.generative.tiger import TIGERModel

tokenizer = TIGERModel.build_tokenizer(["<a_1>", "<b_3>", "<c_5>"])
config = T5Config(vocab_size=len(tokenizer), d_model=128, d_ff=512, num_layers=2,
                  num_heads=4, pad_token_id=tokenizer.pad_token_id,
                  eos_token_id=tokenizer.eos_token_id, decoder_start_token_id=tokenizer.pad_token_id)
model = TIGERModel(config)
model.set_hyper(temperature=1.0)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| config | T5Config | T5 configuration. Set `vocab_size` to the vocabulary size after adding semantic ID tokens. | Required |
| temperature | float | Logit temperature, set through `set_hyper` | 1.0 |

## 5. Choosing a model

| Available data | Training objective | Output |
| --- | --- | --- |
| Chronological interactions, optionally with timestamps | HSTU predicts the next item from its history | Scores for candidate items |
| Interaction sequences and text vectors aligned by item ID | HLLM learns to predict the next item using these vectors | Match scores between a user history and candidate items |
| Item vectors | RQ-VAE learns to reconstruct the vectors | A mapping from item IDs to semantic IDs |
| Interaction sequences and a semantic ID mapping | TIGER predicts the next item’s semantic ID | Candidate codes that can be looked up in the mapping |

## Train HSTU with SeqTrainer

```python
import os
import pickle
import torch

from torch_rechub.models.generative import HSTUModel
from torch_rechub.trainers import SeqTrainer
from torch_rechub.utils.data import SequenceDataGenerator

with open("examples/generative/data/ml-1m/processed/train_data.pkl", "rb") as f:
    train_data = pickle.load(f)
with open("examples/generative/data/ml-1m/processed/val_data.pkl", "rb") as f:
    val_data = pickle.load(f)
with open("examples/generative/data/ml-1m/processed/test_data.pkl", "rb") as f:
    test_data = pickle.load(f)
with open("examples/generative/data/ml-1m/processed/vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

train_gen = SequenceDataGenerator(
    train_data["seq_tokens"],
    train_data["seq_positions"],
    train_data["targets"],
    train_data["seq_time_diffs"],
)
val_gen = SequenceDataGenerator(
    val_data["seq_tokens"],
    val_data["seq_positions"],
    val_data["targets"],
    val_data["seq_time_diffs"],
)
test_gen = SequenceDataGenerator(
    test_data["seq_tokens"],
    test_data["seq_positions"],
    test_data["targets"],
    test_data["seq_time_diffs"],
)

train_dl = train_gen.generate_dataloader(batch_size=512, num_workers=0)[0]
val_dl = val_gen.generate_dataloader(batch_size=512, num_workers=0)[0]
test_dl = test_gen.generate_dataloader(batch_size=512, num_workers=0)[0]

item_to_idx = vocab["item_to_idx"] if "item_to_idx" in vocab else vocab
vocab_size = max(item_to_idx.values()) + 1  # Include PAD at token 0. len(...) alone can omit it.
model = HSTUModel(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=200,
    dropout=0.1,
    time_bucket_unit="seconds",
)

os.makedirs("saved/hstu", exist_ok=True)
trainer = SeqTrainer(
    model,
    optimizer_fn=torch.optim.Adam,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=10,
    earlystop_patience=10,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_path="saved/hstu",
)

trainer.fit(train_dl, val_dl)
test_loss, top1_acc = trainer.evaluate(test_dl)
print(f"test_loss={test_loss:.4f}, top1_acc={top1_acc:.4f}")
```

## GPU memory and evaluation

Scoring the full vocabulary with HSTU or HLLM produces a tensor of shape `[batch_size, sequence_length, vocab_size]`. If it uses too much GPU memory, try `loss_type="sampled_softmax"` on a single GPU. For example, `loss_params={"num_negatives": 32}` samples 32 negatives per target. On CUDA, `amp=True` enables mixed precision. Use `accumulation_steps` to accumulate gradients over several batches before an update.

`SeqTrainer.evaluate()` returns loss and top-1 accuracy on held-out targets. To retrieve the ten highest-scoring items, use `predict(data_loader, top_k=10)`, which scores candidates in chunks. The [HSTU](/blog/hstu_reproduction), [HLLM](/blog/hllm_reproduction) and [TIGER](/blog/tiger_reproduction) guides cover data preparation, single-GPU settings and the Hit@K and NDCG@K reported by the scripts.
