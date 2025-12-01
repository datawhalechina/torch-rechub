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
* **Rich Model Library:** Includes various classic and cutting - edge recommendation algorithms (listed below).
* **Standardized Process:** Provide unified data loading, training, and evaluation processes.
* **Easy to Configure:** Easily adjust experimental settings through configuration files or command - line parameters.
* **Reproducibility:** Aims to ensure the reproducibility of experimental results.
* **Easy to Extend:** Decouple model training from model definition, without the concept of a base model.
* **Native Functions:** Use PyTorch's native classes and functions as much as possible without excessive customization.
* **Concise Model Code:** Facilitate beginners' learning while adhering to the ideas of academic papers.
* **Other Features:** For example, support negative sampling, multi - task learning, etc.

## Overall Architecture

![Architecture Design Diagram](/img/project_framework.jpg "Architecture Design Diagram")

## Data Layer Design

### Feature Classes

Numerical Features

* Such as age, salary, daily click - through rate, etc.

Categorical Features

* Such as city, education level, gender, etc.
* Encode with LabelEncoder to obtain Embedding vectors.

Sequence Features

* Ordered interest sequences: such as the item list clicked in the last week.
* Unordered tag features: such as movie genres (action | suspense | crime).
* Encode with LabelEncoder to obtain sequence Embedding vectors.
* Perform pooling to reduce dimensions.
* Preserve the sequence for model operations such as attention with other features.
* Share the Embedding Table with Sparse features.

### Data Classes
* Dataset
* Dataloader

### Tool Classes
* Sequence feature generation
* Sample construction
* Negative sampling
* Vectorized retrieval

## Model Layer Design

### Model Classes
General Layers
Shallow Feature Modeling

* LR: Logistic Regression
* MLP: Multi - Layer Perceptron, parameters such as dims can be set through a dictionary.
* EmbeddingLayer: A general Embedding layer that handles three types of features, maintains an EmbeddingTable in dictionary format, and outputs the input embeddings required by the model.

Deep Feature Modeling

* FM, FFM, CIN
* self - attention, target - attention, transformer

Custom Layers