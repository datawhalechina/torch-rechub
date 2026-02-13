---
title: Project Introduction
description: Overview of Torch-RecHub project architecture, features, and design principles
---

# Project Introduction

## Project Overview

**Torch-RecHub** is a flexible and easily extensible recommendation system framework built using PyTorch. It aims to simplify the research and application of recommendation algorithms, providing common model implementations, data processing tools, and evaluation metrics.

![Torch-RecHub Banner](/img/banner.png "Torch-RecHub Banner")

## Features

* **Modular Design:** Easy to add new models, datasets, and evaluation metrics.
* **Based on PyTorch:** Leverage PyTorch's dynamic graph and GPU acceleration capabilities.
* **Rich Model Library:** Includes various classic and cutting-edge recommendation algorithms.
* **Standardized Process:** Provide unified data loading, training, and evaluation processes.
* **Easy to Configure:** Easily adjust experimental settings through configuration files or command-line parameters.
* **Reproducibility:** Aims to ensure the reproducibility of experimental results.
* **Easy to Extend:** Decouple model training from model definition, without the concept of a base model.
* **Native Functions:** Use PyTorch's native classes and functions as much as possible without excessive customization.
* **Concise Model Code:** Facilitate beginners' learning while adhering to the ideas of academic papers.
* **Other Features:** For example, support negative sampling, multi-task learning, etc.

## Overall Architecture

![Architecture Design Diagram](/img/project_framework.png "Architecture Design Diagram")

## Core Components

Torch-RecHub adopts a modular design, dividing the core functions of recommendation systems into multiple components:

### 1. Feature Processing

Handles different types of features, including dense features, sparse features, and sequence features.

See [Feature Definitions](/en/core/features) for details.

### 2. Data Pipeline

Responsible for data loading, preprocessing, and generating dataloaders, supporting data processing for ranking and matching models.

See [Data Pipeline](/en/core/data) for details.

### 3. Model Library

Implements various recommendation models, including ranking models, matching models, multi-task models, and generative recommendation models.

See [Model Library](/en/models/intro) for details.

### 4. Training & Evaluation

Provides a unified training interface, supporting model training, evaluation, prediction, and ONNX export.

See [Training & Evaluation](/en/core/evaluation) for details.

### 5. Development Tools

Provides various utility functions, such as ONNX export, model visualization, callbacks, and loss functions.

See [Development Tools](/en/tools/intro) for details.

## Supported Models

### Ranking Models - 13

* DeepFM, Wide&Deep, DCN, DCN-v2, DIN, DIEN, BST, AFM, AutoInt, FiBiNET, DeepFFM, EDCN

### Matching Models - 12

* DSSM, YoutubeDNN, YoutubeSBC, MIND, SINE, GRU4Rec, SASRec, NARM, STAMP, ComiRec, FacebookDSSM

### Multi-Task Models - 5

* ESMM, MMoE, PLE, AITM, SharedBottom

### Generative Recommendation - 2

* HSTU, HLLM

## Quick Start

To get started with Torch-RecHub, see the [Quick Start](/en/guide/quick_start) page to learn how to install the framework and run your first example.

## Production Deployment

Torch-RecHub supports exporting trained models to ONNX format for production deployment. See [Production Deployment](/en/serving/intro) for details.

## Community Contribution

We welcome all forms of contributions! See the [Contributing Guide](/en/community/contributing) for detailed contribution guidelines.

## FAQ

If you encounter issues, see the [FAQ](/en/community/faq) page or submit an Issue on GitHub.
