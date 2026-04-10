---
title: Tutorial Overview
description: Torch-RecHub scenario tutorials overview and entry points
---

# Tutorial Overview

This section focuses on practical Torch-RecHub usage patterns across different recommendation scenarios. All code snippets in this section assume you are using the sample data included in the repository and running from the **repository root**.

> **Code resources**
> - Full Python example scripts: `examples/`
> - Step-by-step tutorials in the docs: `docs/en/tutorials/`

## Tutorial List

| Tutorial | Best for | Link |
| --- | --- | --- |
| CTR Prediction | Ranking / click-through rate prediction | [CTR tutorial](/tutorials/ctr) |
| Retrieval Models | Two-tower retrieval / vector search | [Retrieval tutorial](/tutorials/retrieval) |
| Multi-Task Learning | Joint CTR/CVR modeling | [Multi-task tutorial](/tutorials/pipeline) |

## Quick Navigation

### CTR Prediction (Ranking)

Best if you want to quickly run through `WideDeep / DeepFM / DCN`.

```python
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer

model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128]})
trainer = CTRTrainer(model, device="cuda:0")
trainer.fit(train_dl, val_dl)
```

[View full tutorial →](/tutorials/ctr)

### Retrieval Models

Best if you want to run a two-tower or multi-interest retrieval pipeline such as `DSSM / YoutubeDNN / MIND`.

```python
from torch_rechub.models.matching import DSSM
from torch_rechub.trainers import MatchTrainer

model = DSSM(user_features, item_features)
trainer = MatchTrainer(model, mode=0, device="cuda:0")
trainer.fit(train_dl)
```

[View full tutorial →](/tutorials/retrieval)

### Multi-Task Learning

Best if you want to understand the training flow of `MMOE / PLE / ESMM` on the Ali-CCP sample data.

```python
from torch_rechub.models.multi_task import MMOE
from torch_rechub.trainers import MTLTrainer

model = MMOE(
    features,
    task_types=["classification", "classification"],
    n_expert=8,
    expert_params={"dims": [16]},
    tower_params_list=[{"dims": [8]}, {"dims": [8]}],
)
trainer = MTLTrainer(model, task_types=["classification", "classification"], device="cuda:0")
trainer.fit(train_dl, val_dl)
```

[View full tutorial →](/tutorials/pipeline)

## Model Tutorials

The model-specific tutorial pages below provide focused walkthroughs, including model setup, trainer usage, and tuning-oriented notes for each family.

### Ranking Models

| Model | Summary | Link |
| --- | --- | --- |
| DeepFM | FM + deep network for ranking | [DeepFM](/tutorials/models/ranking/deepfm) |
| Wide&Deep | Memorization + generalization | [WideDeep](/tutorials/models/ranking/widedeep) |
| DCN / DCNv2 | Explicit feature crossing | [DCN](/tutorials/models/ranking/dcn) |
| DIN | Target-aware attention over user history | [DIN](/tutorials/models/ranking/din) |
| DIEN | Interest evolution modeling | [DIEN](/tutorials/models/ranking/dien) |
| BST | Transformer-based sequence ranking | [BST](/tutorials/models/ranking/bst) |

### Retrieval Models

| Model | Summary | Link |
| --- | --- | --- |
| DSSM | Classic two-tower semantic matching | [DSSM](/tutorials/models/matching/dssm) |
| YoutubeDNN | YouTube-style deep retrieval | [YoutubeDNN](/tutorials/models/matching/youtube_dnn) |
| MIND | Multi-interest retrieval with capsules | [MIND](/tutorials/models/matching/mind) |

### Multi-Task Models

| Model | Summary | Link |
| --- | --- | --- |
| MMOE | Multi-gate mixture-of-experts | [MMOE](/tutorials/models/multi_task/mmoe) |
| PLE | Progressive layered extraction | [PLE](/tutorials/models/multi_task/ple) |

## Suggested Validation Order

1. Start with [Quick Start](/guide/quick_start) to verify that your environment, trainers, and sample datasets are all working.
2. Then read the [CTR tutorial](/tutorials/ctr) or [Retrieval tutorial](/tutorials/retrieval) to understand the full data flow.
3. Finally, go deeper into model-specific pages for parameter explanations, tuning suggestions, and ONNX / visualization usage.
