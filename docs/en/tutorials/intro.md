---
title: Tutorial Overview
description: Torch-RecHub scenario tutorials overview and entry points
---

# Tutorial Overview

To check your installation, train DeepFM once with the [Quick Start](/guide/quick_start). The three tutorials below use datasets included in the repository. Open a notebook or terminal at the repository root so the example paths resolve correctly.

## Tutorial List

- **Rank candidate ads.** Predict the probability of a click on a Criteo impression, then use AUC to see how well the model ranks clicked ads above unclicked ones. [Ranking tutorial](/tutorials/ctr)
- **Retrieve candidate movies.** Train DSSM on MovieLens, compute user and movie embeddings separately, and retrieve ten movies for one user. [Retrieval tutorial](/tutorials/retrieval)
- **Predict clicks and purchases together.** Train MMOE or PLE with two Ali-CCP labels and inspect AUC for each task. The ESMM example also uses a label for impressions that led to both a click and a purchase. [Multi-task tutorial](/tutorials/pipeline)

## Model Tutorials

If you already know which model you want, open its tutorial to see which fields it needs.

### Ranking Models

| Model | Summary | Link |
| --- | --- | --- |
| DeepFM | FM + deep network for ranking | [DeepFM](/tutorials/models/ranking/deepfm) |
| Wide&Deep | A linear layer and a deep network score each impression | [Wide&Deep tutorial](/tutorials/models/ranking/widedeep) |
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
