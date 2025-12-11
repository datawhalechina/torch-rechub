---
title: Model Library Overview
description: Torch-RecHub model library overview
---

# Model Library Overview

Torch-RecHub provides a rich set of recommendation models covering ranking, matching, multi-task learning, and generative recommendation. All models are implemented with PyTorch and are easy to use and extend.

## Library Structure

Models are organized by stage/task:

1. **Ranking**: fine-ranking models to predict CTR or preference scores.
2. **Matching**: retrieval models to fetch candidate items from large catalogs.
3. **Multi-Task**: models that jointly optimize related tasks.
4. **Generative**: models that generate personalized recommendations.

## Model Selection Guide

### Ranking Models

| Model    | Scenario                            | Highlights                                      |
| ---      | ---                                 | ---                                             |
| WideDeep | Basic ranking                       | Combines linear (memorization) and deep (generalization) |
| DeepFM   | Feature interactions matter         | Captures low- and high-order feature crosses    |
| DCN/DCNv2| Explicit feature crossing           | Efficient high-order crosses                    |
| DIN      | Dynamic user interest               | Attention-based interest modeling               |
| DIEN     | Long sequence interest              | Models evolving interests                       |
| BST      | Sequence-focused                    | Transformer for sequential features             |
| AutoInt  | Auto feature interaction            | Learns interaction patterns automatically       |

### Matching Models

| Model           | Scenario                | Highlights                                   |
| ---             | ---                     | ---                                          |
| DSSM            | Text / semantic match   | Two-tower mapping to a shared vector space   |
| YoutubeDNN      | Large-scale retrieval   | Sequence-based deep retrieval                |
| MIND            | Multi-interest          | Learns multiple user interests               |
| GRU4Rec/SASRec  | Sequential recommendation| Models recent behavior sequences             |
| ComirecDR/ComirecSA | Controllable interests | Control number of interests                  |

### Multi-Task Models

| Model        | Scenario                     | Highlights                                      |
| ---          | ---                          | ---                                             |
| SharedBottom | Tasks highly related         | Shared bottom network                           |
| MMOE         | Task conflict exists         | Multi-gate mixture-of-experts per task          |
| PLE          | Complex multi-task           | Progressive layered extraction to reduce negative transfer |
| ESMM         | Sample selection bias        | Full-space modeling to mitigate bias            |
| AITM         | Task dependency              | Adaptive information transfer between tasks     |

### Generative Recommendation

| Model | Scenario                | Highlights                                  |
| ---   | ---                     | ---                                         |
| HSTU  | Large-scale sequences   | Hierarchical Sequential Transduction Units  |
| HLLM  | LLM-enhanced recommendation | Combines LLM semantic understanding      |

## Documentation Navigation

- Ranking models: detailed principles, usage, and parameters.  
  [See ranking docs](/en/models/ranking)
- Matching models: detailed principles, usage, and parameters.  
  [See matching docs](/en/models/matching)
- Multi-task models: detailed principles, usage, and parameters.  
  [See multi-task docs](/en/models/mtl)
- Generative models: principles, usage, and parameters.  
  [See generative docs](/en/models/generative)

## Usage Examples

```python
# Ranking example
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

model = DeepFM(
    deep_features=deep_features,
    fm_features=fm_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2}
)
trainer = CTRTrainer(model, optimizer_params={"lr": 0.001}, device="cuda:0")
trainer.fit(train_dataloader, val_dataloader)
```

```python
# Matching example
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

model = DSSM(
    user_features=user_features,
    item_features=item_features,
    temperature=0.02,
    user_params={"dims": [256, 128, 64]},
    item_params={"dims": [256, 128, 64]}
)
trainer = MatchTrainer(model, mode=0, optimizer_params={"lr": 0.001}, device="cuda:0")
trainer.fit(train_dataloader)
```

