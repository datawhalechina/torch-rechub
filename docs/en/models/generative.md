---
title: Generative Recommendation Models
description: Torch-RecHub generative recommendation models detailed introduction
---

# Generative Recommendation Models

Generative recommendation models are an emerging approach that leverages generative AI technologies (such as large language models) for recommendations, capable of generating personalized recommendation content and providing richer, more natural recommendation experiences. Torch-RecHub provides various advanced generative recommendation models that combine the advantages of recommendation systems and generative AI.

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
- Real-time recommendation scenarios
- Trillion-parameter recommendation systems

## 2. HLLMModel

### Description

HLLM (Hybrid Large Language Model) is a hybrid recommendation model that integrates large language model (LLM) capabilities, combining the collaborative filtering ability of recommendation systems with the semantic understanding ability of LLMs.

### Core Principles

- **Hybrid Architecture**: Combines the advantages of traditional recommendation models and large language models
- **Semantic Understanding**: Leverages LLM's powerful semantic understanding ability to process text information
- **Collaborative Filtering**: Retains traditional recommendation model's collaborative filtering ability, utilizing user-item interaction data
- **Multi-modal Fusion**: Supports fusion of multiple modalities such as text and images

### Usage

```python
from torch_rechub.models.generative import HLLMModel

# Create model
model = HLLMModel(
    user_features=user_features,
    item_features=item_features,
    llm_params={
        "model_name": "bert-base-uncased",
        "hidden_size": 768,
        "num_heads": 12,
        "num_layers": 12,
        "dropout": 0.1
    },
    fusion_params={
        "fusion_type": "concat",
        "fusion_dims": [512, 256, 128],
        "dropout": 0.2
    }
)
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| user_features | list | User feature list | None |
| item_features | list | Item feature list | None |
| llm_params | dict | Large language model parameters | None |
| fusion_params | dict | Feature fusion parameters | None |

### Use Cases

- Recommendation scenarios integrating LLM capabilities
- Recommendation scenarios with rich text information
- Scenarios requiring generative recommendations
- Multi-modal recommendation scenarios

## 3. TIGERModel

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
- Scenarios where prefix-sharing across similar items improves cold-start and generalization

## 4. Model Comparison

| Model | Complexity | Expressiveness | Efficiency | Use Cases |
| --- | --- | --- | --- | --- |
| HSTUModel | High | High | Medium | Large-scale sequence recommendation, long sequence modeling |
| HLLMModel | High | High | Low | LLM integration, text-rich scenarios |
| TIGERModel | High | High | Medium | Semantic-ID generative retrieval, very large item spaces |

## 5. Usage Recommendations

1. **Choose models based on business requirements**:
   - For large-scale sequence recommendation, use HSTUModel
   - For scenarios requiring LLM capabilities, use HLLMModel
   - For text-rich recommendation scenarios, use HLLMModel

2. **Choose models based on computational resources**:
   - With limited resources, use HSTUModel
   - With sufficient resources, try HLLMModel
   - Consider model compression techniques like knowledge distillation, quantization, etc.

3. **Model training recommendations**:
   - Use pretrain + finetune approach to improve model effectiveness and training efficiency
   - Use mixed precision training to accelerate model training
   - Use distributed training to handle large-scale data

4. **Model deployment recommendations**:
   - Consider model compression techniques to reduce model size and inference latency
   - Use service-oriented deployment to support high-concurrency requests
   - Consider edge computing to deploy models on edge devices

## 6. Complete Training Example

```python
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

vocab_size = len(vocab["item_to_idx"]) if "item_to_idx" in vocab else len(vocab)
model = HSTUModel(
    vocab_size=vocab_size,
    d_model=128,
    n_heads=4,
    n_layers=2,
    dqk=32,
    dv=32,
    max_seq_len=200,
    dropout=0.1,
)

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

## 7. FAQ

### Q: How to handle large-scale data?
A: Try the following approaches:
- Use distributed training with multi-GPU or multi-machine parallel training
- Use data sampling techniques like negative sampling, stratified sampling, etc.
- Use model parallelism or pipeline parallelism for very large models
- Consider mixed precision training to accelerate training

### Q: How to improve inference efficiency of generative recommendation models?
A: Try the following approaches:
- Use model compression techniques like knowledge distillation, quantization, pruning, etc.
- Use deployment optimization like TensorRT, ONNX Runtime, etc.
- Consider edge computing to deploy models on edge devices
- Use asynchronous inference or batch processing to improve concurrency

### Q: How to evaluate generative recommendation model performance?
A: Try the following approaches:
- Traditional recommendation metrics: AUC, Precision@K, Recall@K, NDCG@K, etc.
- Generative evaluation metrics: BLEU, ROUGE, METEOR, Perplexity, etc.
- Human evaluation: User surveys or A/B testing
- Business metrics: CTR, conversion rate, user retention, etc.

### Q: How to handle cold start problems?
A: Try the following approaches:
- For new users, use content-based or popularity-based recommendations
- For new items, leverage LLM's semantic understanding based on item descriptions
- Use transfer learning to transfer knowledge from related domains
- Use meta-learning to quickly adapt to new users or items

## 8. Application Scenarios

1. **Personalized Content Generation**:
   - Generate personalized recommendation reasons
   - Generate personalized product descriptions
   - Generate personalized marketing copy

2. **Multi-modal Recommendation**:
   - Combine text, image, audio, and other modalities
   - Generate multi-modal recommendation content
   - Support cross-modal recommendations

3. **Interactive Recommendation**:
   - Support natural language interaction between users and recommendation systems
   - Dynamically adjust recommendations based on user feedback
   - Generate conversational recommendations

4. **Contextual Recommendation**:
   - Generate recommendations based on user's current context
   - Generate context-aware recommendation content
   - Support complex scenario recommendations

## 9. Future Trends

1. **Deep Integration of LLMs and Recommendation Systems**:
   - More tightly combine the advantages of LLMs and recommendation systems
   - Develop LLMs specifically optimized for recommendation scenarios
   - Leverage LLM's contextual understanding for more personalized recommendations

2. **Multi-modal Generative Recommendation**:
   - Combine multiple modalities for generative recommendation
   - Support cross-modal content generation and recommendation
   - Develop more efficient multi-modal fusion methods

3. **Real-time Generative Recommendation**:
   - Achieve low-latency generative recommendation
   - Support real-time user interaction and feedback
   - Develop more efficient inference architectures

4. **Controllable Generative Recommendation**:
   - Support user control and adjustment of recommendation results
   - Achieve explainable and trustworthy generative recommendation
   - Develop safer and more reliable generative recommendation systems

5. **Large-scale Generative Recommendation**:
   - Support billions of users and items for large-scale recommendation
   - Develop more efficient model training and inference methods
   - Achieve distributed and scalable generative recommendation systems

Generative recommendation is an important development direction for recommendation systems, capable of providing richer, more natural, and more personalized recommendation experiences. Torch-RecHub provides various advanced generative recommendation models for developers to choose based on business requirements. With the continuous development of large language models and generative AI technologies, generative recommendation will be applied in more scenarios, providing users with better recommendation experiences.
