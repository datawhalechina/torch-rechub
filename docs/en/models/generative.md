---
title: Generative Recommendation Models
description: Torch-RecHub generative recommendation models detailed introduction
---

# Generative Recommendation Models

On this page, “generative recommendation” means using a sequence model to predict or generate the next item token or semantic ID. The current implementations target next-item recommendation; they do not generate recommendation explanations, product copy, or natural-language conversations.

## 1. HSTUModel

### Description

HSTU (Hierarchical Sequential Transduction Units) is an autoregressive sequence recommender for next-item prediction. In Torch-RecHub, `HSTUModel` consumes padded item-token sequences plus optional per-position time-difference features and returns logits over the item vocabulary at every sequence position.

### Core Principles

- **Eq. 2 UVQK projection**: applies one `SiLU` to the joint `UVQK` projection before splitting, so `U`, `V`, `Q`, and `K` all pass through the same non-linearity.
- **Eq. 3 attention bias**: adds per-head bucketed relative position/time bias `rab^{p,t}` to attention scores before `silu(scores) / max_seq_len`.
- **Eq. 4 gated output**: projects `LayerNorm(A V) * U` through one output linear layer, without concat-u/x bypasses or a separate FFN.
- **External residuals**: each layer is wrapped as `x = x + HSTULayer(x)` in `HSTUBlock`.
- **Generative training**: predicts the next token in the sequence and masks PAD token `0` in the loss.

### Usage

```python
import torch

from torch_rechub.models.generative import HSTUModel

model = HSTUModel(
    vocab_size=100000,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=200,
    num_time_buckets=128,
    time_bucket_unit="seconds",
)

seq_tokens = torch.randint(1, 100000, (32, 200))
time_diffs = torch.zeros_like(seq_tokens)  # seconds from query time
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # torch.Size([32, 200, 100000])
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| vocab_size | int | Item vocabulary size, with PAD reserved as token `0` | required |
| d_model | int | Hidden dimension | 512 |
| n_heads | int | Number of attention heads | 8 |
| n_layers | int | Number of stacked HSTU layers | 4 |
| dqk | int | Query/key dimension per head | 64 |
| dv | int | Value/U dimension per head | 64 |
| max_seq_len | int | Maximum supported sequence length | 256 |
| dropout | float | Dropout rate | 0.1 |
| use_time_embedding | bool | Add input-side time-bucket embedding; `time_diffs` is still used by `rab^{p,t}` | True |
| num_time_buckets | int | Number of time buckets for embeddings and attention bias | 128 |
| time_bucket_fn | {"sqrt", "log"} | Time bucketization function | "sqrt" |
| time_bucket_divisor | float | Divisor applied after bucketization | 1.0 |
| tie_embeddings | bool | Tie output projection to token embedding weights | True |

### Use Cases

- Large-scale sequence recommendation
- Long sequence modeling
- Next-item prediction

## 2. HLLMModel

### Description

`HLLMModel` is a lightweight sequential recommender. Item embeddings are produced by an LLM outside the model, then frozen while only the user-sequence Transformer is trained. It does not load BERT or another LLM in `forward`, and it does not accept `user_features` / `item_features` lists.

### Core Principles

- **Offline item semantic table**: `item_embeddings[token_id]` must align with the token IDs used by the training data, and row 0 is PAD.
- **User-sequence model**: position encoding, optional time-bucket embeddings, and relative-position bias.
- **Cosine scoring**: normalized Transformer outputs are multiplied by the frozen item table to return full-vocabulary logits.

### Usage

```python
from torch_rechub.models.generative import HLLMModel

import torch

# Must be [vocab_size, d_model], with row i corresponding to token i
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
time_diffs = torch.zeros_like(seq_tokens)  # seconds
logits = model(seq_tokens, time_diffs)
print(logits.shape)  # [32, 50, vocab_size]
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_embeddings | Tensor or str | Precomputed `[vocab_size, d_model]` embeddings, or a path readable by `torch.load` | required |
| vocab_size | int | Vocabulary size including PAD=0; must equal the number of embedding rows | required |
| d_model | int | Transformer dimension; must equal the embedding width | 512 |
| n_heads / n_layers | int | Number of Transformer heads / layers | 8 / 4 |
| max_seq_len | int | Maximum sequence length | 256 |
| use_rel_pos_bias | bool | Whether to use relative-position bias | True |
| use_time_embedding | bool | Whether to use time-difference embeddings | True |
| temperature | float | Cosine-logit temperature | 0.07 |

### Use Cases

- You can generate item embeddings offline and align them by token ID.
- Next-item sequence prediction.
- You want to fix the item semantic space and train only the user-sequence tower.

## 3. RQVAEModel

### Description

RQ-VAE converts each continuous item embedding into a tuple of discrete codebook indices through multi-level residual vector quantization. These indices are formatted as semantic IDs such as `<a_12><b_7><c_91>` and become TIGER's item tokens.

### Minimal Training and Semantic-ID Pipeline

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

# Preserve dataset order when generating semantic IDs, or item indices will be misaligned
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

The keys of `semantic_ids` are row numbers in `EmbDataset`. The embedding-file row order must therefore match the item-ID mapping used by subsequent TIGER data. Do not use `shuffle=True` during generation.

### Main Parameters

| Parameter | Description |
| --- | --- |
| in_dim | Original item-embedding dimension |
| num_emb_list | Codebook size at each residual-quantizer level; its length is the number of semantic-ID segments |
| e_dim | Quantization-space dimension |
| layers | Encoder hidden layers; the decoder is built symmetrically in reverse order |
| loss_type | `mse` or `l1` reconstruction loss |
| sk_epsilons / sk_iters | Sinkhorn assignment parameters; list length must match the number of codebooks |

## 4. TIGERModel

### Description

TIGER (Transformer Index for GEnerative Recommenders) frames recommendation as a sequence-to-sequence task: "generate the semantic ID of the next item". Each item is first quantized by RQ-VAE into a short tuple of codebook tokens (a *semantic ID*, e.g. `<a_1><b_3><c_5>`); TIGER autoregressively generates the next item's semantic ID on top of T5, then constrains beam search to legal items via a prefix trie. `TIGERModel` subclasses `transformers`' `T5ForConditionalGeneration`.

### Core Principles

- **Semantic IDs**: RQ-VAE applies multi-level residual quantization over item embeddings, giving each item a tuple of codebook tokens. Similar items share prefixes, producing a natural hierarchy.
- **Seq-to-seq**: the input is the concatenated semantic IDs of a user's history; the label is the next item's semantic ID, trained with T5 teacher-forcing cross-entropy.
- **New tokens**: all semantic-ID tokens are added to the tokenizer and `resize_token_embeddings` is called *before* training, otherwise tokens like `<a_1>` are split into sub-words.
- **Constrained generation**: at inference a `Trie` builds a `prefix_allowed_tokens_fn` so beam search only emits semantic IDs that exist in the item table.

### Usage

The full workflow (generate toy data / train / test, plus the RQ-VAE → TIGER pipeline for real data) is documented in the [TIGER Reproduction Notes](/blog/tiger_reproduction) and the example scripts `examples/generative/run_tiger_movielens.py` / `run_tiger_amazon_books.py`. Minimal model usage:

```python
from transformers import T5Config, T5Tokenizer

from torch_rechub.models.generative.tiger import TIGERModel

tokenizer = T5Tokenizer.from_pretrained("t5-small")
tokenizer.add_tokens(["<a_1>", "<b_3>", "<c_5>"])  # semantic-ID tokens

config = T5Config.from_pretrained("t5-small")
config.vocab_size = len(tokenizer)
model = TIGERModel(config)
model.set_hyper(temperature=1.0)
model.resize_token_embeddings(len(tokenizer))
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| config | T5Config | T5 config; `vocab_size` must match the tokenizer size *after* adding semantic-ID tokens | required |
| temperature | float | Logit temperature, set via `set_hyper` | 1.0 |

### Use Cases

- Semantic-ID based generative retrieval
- Very large item catalogs that benefit from compressed item representations
- Sequence-recommendation experiments that represent each item with multiple discrete tokens

## 5. RPGModel

### Description

RPG (Recommendation with Parallel Generation) keeps TIGER's semantic-ID idea but drops the ordering between digits. Each item is quantized by OPQ into `n_digit` **unordered** codes, so no digit refines another and all of them can be predicted in a single forward pass. Removing autoregressive decoding is what lets semantic IDs grow from 4 tokens to 32 or 64, which is where the accuracy gain comes from.

### Core Principles

- **Unordered semantic IDs**: `OPQTokenizer` runs whitened PCA over item embeddings and quantizes them with FAISS `OPQ{m},IVF1,PQ{m}x{bits}`. Digit `j` quantizes its own subspace, so the digits are independent rather than residual.
- **One position per item**: an item is represented by mean-pooling the embeddings of its `n_digit` tokens, so a history of `L` items is `L` positions rather than `L * n_digit`.
- **Multi-token prediction**: `n_digit` independent `ResBlock` heads (identity at initialization) produce one query per digit. Each query is scored against its own codebook by cosine similarity divided by `temperature`, and the heads are trained with one cross-entropy per digit, averaged.
- **Graph-constrained decoding**: item similarity is derived from the learned codebooks, and only the `n_edges` nearest neighbours of each item are kept. Retrieval starts from a random beam and walks this graph for `propagation_steps` rounds, so only a fraction of the catalog is ever scored. Scoring the whole catalog stays available and is what validation uses.

### Usage

The full workflow (preprocess / tokenize / train / test) is in `examples/generative/run_rpg_amazon_2014.py`. Minimal model usage:

```python
import numpy as np

from torch_rechub.models.generative import RPGModel
from torch_rechub.utils.opq import OPQTokenizer

item_embeddings = np.load("item_embeddings.npy")  # (n_items - 1, dim)
tokenizer = OPQTokenizer(n_codebook=32, codebook_size=256, pca_dim=128)
tokenizer.fit(item_embeddings)

model = RPGModel(tokenizer.item_tokens(), codebook_size=256, temperature=0.03)
states, loss = model(input_ids, attention_mask, labels)

model.build_decoding_graph(n_edges=200)
preds = model.generate(input_ids, attention_mask, seq_lens, topk=10, use_graph=True)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| item_tokens | LongTensor | `(n_items, n_digit)` token table from `OPQTokenizer.item_tokens()`; row `0` is PAD | required |
| codebook_size | int | Codes per digit | 256 |
| n_embd | int | Hidden dimension | 448 |
| n_layer | int | Transformer layers | 2 |
| n_head | int | Attention heads | 4 |
| n_inner | int | Feed-forward dimension | 1024 |
| max_seq_len | int | Maximum items per sequence | 50 |
| embd_pdrop / attn_pdrop | float | Embedding / attention dropout; the paper relies on heavy dropout to regularize a small backbone | 0.5 |
| temperature | float | Divides the cosine logits | 0.07 |

### Use Cases

- Very large item catalogs where TIGER's beam search over digits is the inference bottleneck
- Settings that benefit from long semantic IDs (32-64 tokens) rather than the usual 4
- Latency- or memory-constrained retrieval, thanks to graph-constrained decoding

## 6. Model Comparison

| Model | Complexity | Expressiveness | Efficiency | Use Cases |
| --- | --- | --- | --- | --- |
| HSTUModel | Medium | High | Medium | Next-item prediction and long-sequence modeling |
| HLLMModel | Medium | Medium | Medium | Sequence recommendation with precomputed LLM item embeddings |
| RQVAEModel | Medium | Medium | Medium | Quantizing continuous item embeddings into TIGER semantic IDs |
| TIGERModel | High | High | Medium | Semantic-ID generative retrieval, very large item spaces |
| RPGModel | Medium | High | High | Long semantic IDs, low-latency generative retrieval |

## 7. Usage Recommendations

1. Use HSTUModel for direct next-item prediction over item tokens.
2. Use HLLMModel when you already have item embeddings aligned by token ID and want to freeze the item semantic space.
3. Before TIGER, use RQVAEModel to generate semantic IDs. Quantization and generation must share exactly the same item-row mapping.
4. HSTU/HLLM return `[batch, seq_len, vocab_size]`; estimate the memory required by these logits before using a large vocabulary.
5. Prefer RPGModel over TIGER when decoding latency matters: it predicts every digit in one pass, so long semantic IDs cost no extra decoding steps.

## 8. Complete Training Example

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
vocab_size = max(item_to_idx.values()) + 1  # token 0 is PAD; len(...) can be wrong
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

## 9. Current Implementation Boundaries

- `SeqTrainer` trains HSTU/HLLM and reports loss plus token-level top-1 accuracy. It does not provide built-in Recall@K, NDCG@K, BLEU, or ROUGE evaluation.
- RQ-VAE and TIGER form a two-stage pipeline: quantize item embeddings offline, then train TIGER. The project does not generate the original item embeddings automatically.
- RPG is a two-stage pipeline as well: fit `OPQTokenizer` on item embeddings offline, then train the model on the resulting tokens. Its OPQ step needs `faiss`, and `RPGTrainer` reports Recall@K and NDCG@K rather than token-level accuracy.
- Trainers can optionally use single-machine `DataParallel`, but there is no DDP, multi-machine distributed training, pipeline parallelism, or automatic mixed-precision workflow.
- This module does not include production serving, TensorRT, edge deployment, natural-language content generation, or multimodal recommendation. Those capabilities must be implemented and validated outside the project.
