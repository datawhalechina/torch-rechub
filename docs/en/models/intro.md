---
title: Model Library Overview
description: Overview of the Torch-RecHub model library
---

# Model Library Overview

Torch-RecHub provides a rich recommendation model library that covers ranking, matching, multi-task learning, and generative recommendation. All models are implemented in PyTorch and are easy to use and extend.

## Model Library Structure

The model library is organized by recommendation stage and task type:

1. **Ranking Models**: Used during fine ranking to predict click-through rates or user preference scores for items
2. **Matching Models**: Used during candidate generation to retrieve candidates from a large item collection
3. **Multi-Task Models**: Jointly optimize multiple related tasks to improve generalization
4. **Generative Recommendation**: Uses generative models to produce personalized recommendations

## Model Selection Guide

### Choosing a Ranking Model

| Model | Suitable Scenario | Characteristics |
| --- | --- | --- |
| WideDeep | Basic ranking tasks | Combines linear and deep models to balance memorization and generalization |
| DeepFM | Scenarios where feature interactions matter | Captures both low-order and high-order feature interactions |
| DCN/DCNv2 | Explicit feature-crossing scenarios | Explicitly learns high-order feature crosses with high computational efficiency |
| DIN | Scenarios with dynamically changing user interests | Uses attention to capture user interests |
| DIEN | Long-sequence interest modeling | Models the dynamic evolution of user interests |
| BST | Scenarios where sequence features matter | Uses a Transformer to model sequence features |
| AutoInt | Automatic feature-interaction learning | Automatically learns feature-interaction patterns |

### Choosing a Matching Model

| Model | Suitable Scenario | Characteristics |
| --- | --- | --- |
| DSSM | Text-matching scenarios | Uses a two-tower architecture to map users and items into the same vector space |
| YoutubeDNN | Large-scale recommendation | Deep matching based on user behavior sequences |
| MIND | Multi-interest recommendation | Learns multiple interest representations for each user |
| GRU4Rec/SASRec | Sequential recommendation | Models a user's recent behavior sequence |
| ComirecDR/ComirecSA | Controllable multi-interest recommendation | Allows control over the number of generated interests |

### Choosing a Multi-Task Model

| Model | Suitable Scenario | Characteristics |
| --- | --- | --- |
| SharedBottom | Scenarios with strongly related tasks | All tasks share the bottom network |
| MMOE | Scenarios with substantial task conflict | Uses a multi-gate mixture of experts so each task learns a different expert combination |
| PLE | Complex multi-task scenarios | Uses progressive layered extraction to alleviate negative transfer |
| ESMM | Scenarios with sample-selection bias | Uses entire-space modeling to address sample-selection bias |
| AITM | Scenarios with dependencies between tasks | Uses adaptive information transfer to learn task dependencies |

### Choosing a Generative Recommendation Model

| Model | Suitable Scenario | Characteristics |
| --- | --- | --- |
| HSTU | Next-item sequential recommendation | Hierarchical sequential transduction units with positional and temporal biases |
| HLLM | Sequential recommendation with precomputed LLM item embeddings | Freezes the item semantic table and trains a user-sequence Transformer |
| RQ-VAE | Item semantic-ID quantization | Compresses continuous item embeddings into multi-level codebook IDs |
| TIGER | Semantic-ID generative retrieval | Uses T5 to generate a valid semantic ID for the next item |

## Model Documentation

### Ranking Models

Detailed descriptions of ranking-model principles, usage, and parameters.

[View Ranking Model Documentation](/models/ranking)

### Matching Models

Detailed descriptions of matching-model principles, usage, and parameters.

[View Matching Model Documentation](/models/matching)

### Multi-Task Models

Detailed descriptions of multi-task-model principles, usage, and parameters.

[View Multi-Task Model Documentation](/models/mtl)

### Generative Recommendation Models

Detailed descriptions of generative recommendation-model principles, usage, and parameters.

[View Generative Recommendation Model Documentation](/models/generative)

## Usage Example

```python
# Ranking-model example
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

# Create the model
model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2})

# Create the trainer
trainer = CTRTrainer(model, optimizer_params={"lr": 0.001}, device="cpu")

# Train the model
trainer.fit(train_dataloader, val_dataloader)

# Matching-model example
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

# Create the model
model = DSSM(user_features=user_features, item_features=item_features, temperature=1.0,
             user_params={"dims": [256, 128, 64]}, item_params={"dims": [256, 128, 64]})

# Create the trainer
trainer = MatchTrainer(model, mode=0, device="cpu")

# Train the model
trainer.fit(train_dataloader)
```

## Model Papers

* **Ranking** — [DeepFM](https://arxiv.org/abs/1703.04247), [Wide&Deep](https://arxiv.org/abs/1606.07792), [DCN](https://arxiv.org/abs/1708.05123), [DCNv2](https://arxiv.org/abs/2008.13535), [DIN](https://arxiv.org/abs/1706.06978), [DIEN](https://arxiv.org/abs/1809.03672), [BST](https://arxiv.org/abs/1905.06874), [AFM](https://arxiv.org/abs/1708.04617), [AutoInt](https://arxiv.org/abs/1810.11921), [FiBiNET](https://doi.org/10.1145/3298689.3347043), [FAT-DeepFFM](https://arxiv.org/abs/1905.06336), [EDCN](https://dlp-kdd.github.io/assets/pdf/DLP-KDD_2021_paper_12.pdf).
* **Retrieval** — [DSSM](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf), [YouTubeDNN](https://dl.acm.org/doi/10.1145/2959100.2959190), [YouTubeSBC](https://doi.org/10.1145/3298689.3346996), [MIND](https://arxiv.org/abs/1904.08030), [SINE](https://arxiv.org/abs/2102.09267), [GRU4Rec](https://arxiv.org/abs/1511.06939), [SASRec](https://arxiv.org/abs/1808.09781), [NARM](https://arxiv.org/abs/1711.04725), [STAMP](https://doi.org/10.1145/3219819.3219950), [ComiRec](https://arxiv.org/abs/2005.09347).
* **Multi-task learning** — [ESMM](https://arxiv.org/abs/1804.07931), [MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007), [PLE](https://dl.acm.org/doi/10.1145/3383313.3412236), [AITM](https://arxiv.org/abs/2105.08489).
* **Sequential and generative recommendation** — [HSTU](https://arxiv.org/abs/2402.17152), [HLLM](https://arxiv.org/abs/2409.12740), [TIGER](https://arxiv.org/abs/2305.05065).

The FAT-DeepFFM paper also describes DeepFFM, its base model.

## Contributing a New Model

If you would like to contribute a new model, see the [Contributing Guide](/community/contributing) and follow the project's coding standards and documentation requirements.
