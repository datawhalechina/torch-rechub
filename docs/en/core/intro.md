---
title: Core Components Overview
description: Torch-RecHub core components overview
---

# Core Components Overview

Torch-RecHub is modular: features, data, models, training, and tools are separated for clarity and extensibility.

## Architecture

1) **Feature layer** – dense, sparse, and sequence feature definitions.  
2) **Data layer** – loading, preprocessing, and dataloader generation.  
3) **Model layer** – ranking, matching, multi-task, and generative models.  
4) **Training layer** – unified trainers for fit/eval/predict/ONNX export.  
5) **Tools layer** – ONNX export, visualization, callbacks, losses, etc.

## Component Relations

- Feature layer guides preprocessing in the data layer.  
- Data generators feed the training layer.  
- Models are consumed by trainers.  
- Trainers call tools for export/visualization/tracking.

## Component Details

- **Feature processing**: `DenseFeature`, `SparseFeature`, `SequenceFeature`. See [Features](/en/core/features).  
- **Data pipeline**: `TorchDataset`, `PredictDataset`, `DataGenerator`, `MatchDataGenerator`. See [Data](/en/core/data).  
- **Training & evaluation**: `CTRTrainer`, `MatchTrainer`, `MTLTrainer` (and generative trainer variants). See [Training & Evaluation](/en/core/evaluation).

