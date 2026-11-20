---
layout: home

hero:
  name: "Torch-RecHub"
  text: "A lightweight PyTorch framework for recommender systems"
  tagline: "Train models on browsing, click and purchase histories. Try the bundled samples first, then switch to your own data."
  image:
    src: /img/logo.png
    alt: Torch-RecHub
  actions:
    - theme: brand
      text: Quick start
      link: /guide/quick_start
    - theme: alt
      text: Browse the models
      link: /models/intro
    - theme: alt
      text: ONNX deployment
      link: /serving/onnx

features:
  - icon: R
    title: Click prediction
    details: Estimate how likely a user is to click a candidate item with DeepFM or DIN.
    link: /models/ranking
    linkText: Ranking models

  - icon: M
    title: Candidate retrieval
    details: Learn user and item embeddings with DSSM, or use MIND to represent several interests for each user.
    link: /models/matching
    linkText: Retrieval models

  - icon: MT
    title: Multi-task learning
    details: Learn from click and conversion labels on the same examples and track each task's metrics separately.
    link: /models/mtl
    linkText: Multi-task models

  - icon: G
    title: Sequential and generative recommendation
    details: Predict the next item from an interaction history, or use TIGER to generate its semantic ID one token at a time.
    link: /models/generative
    linkText: Generative models

  - icon: ONNX
    title: ONNX export
    details: Export a trained model and compare PyTorch and ONNX Runtime predictions on the same input.
    link: /serving/onnx
    linkText: Export a model

  - icon: V
    title: Vector search
    details: Index item embeddings with Annoy, Faiss or Milvus and look up the nearest items to a user vector.
    link: /serving/vector_index
    linkText: Build a vector index

  - icon: T
    title: Experiment tracking
    details: View training loss and validation metrics for each epoch in TensorBoard, W&B or SwanLab.
    link: /tools/tracking
    linkText: Track an experiment

  - icon: D
    title: Training data
    details: Feed encoded features and labels to a trainer in batches, or stream larger datasets from Parquet files.
    link: /core/data
    linkText: Prepare your data
---

<div class="home-badges">
  <span>Ranking</span>
  <span>Retrieval</span>
  <span>Multi-task learning</span>
  <span>Generative models</span>
  <span>ONNX</span>
  <span>Vector search</span>
  <span>Experiment tracking</span>
  <span>Visualization</span>
</div>

<div class="home-quick-grid">
  <a class="home-quick-card" href="./guide/quick_start">
    <strong>Train your first DeepFM model</strong>
    <span>Run two epochs on a CPU with the Criteo click sample and check the test AUC.</span>
  </a>
  <a class="home-quick-card" href="./models/intro">
    <strong>Choose a model for your task</strong>
    <span>Use the data you have to choose a retrieval, ranking, multi-task or generative model.</span>
  </a>
  <a class="home-quick-card" href="./serving/demo">
    <strong>Predict with a trained model</strong>
    <span>Give PyTorch and ONNX Runtime the same impression and compare their click scores.</span>
  </a>
</div>
