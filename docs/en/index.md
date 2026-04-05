---
layout: home

hero:
  name: "Torch-RecHub"
  text: "Recommendation engineering that moves from research to production"
  tagline: "Build ranking, matching, multi-task, generative, and serving workflows on top of a lightweight PyTorch stack that stays easy to extend."
  image:
    src: /img/logo.png
    alt: Torch-RecHub
  actions:
    - theme: brand
      text: Quick Start
      link: /guide/quick_start
    - theme: alt
      text: Model Zoo
      link: /models/intro
    - theme: alt
      text: ONNX Export
      link: /serving/onnx

features:
  - icon: R
    title: Ranking Workflows
    details: Start with production-friendly ranking models such as WideDeep, DeepFM, DIN, BST, DIEN, DCN, and more.

  - icon: M
    title: Matching and Retrieval
    details: Cover two-tower retrieval, sequential recall, multi-interest modeling, vector indexing, and recall evaluation in one workflow.

  - icon: MT
    title: Multi-Task Learning
    details: Train ESMM, MMOE, PLE, AITM, and related architectures with a consistent trainer and feature definition style.

  - icon: G
    title: Generative Extensions
    details: Explore newer recommendation directions without leaving the same core data, trainer, and deployment conventions.

  - icon: ONNX
    title: Deployment Ready
    details: Export models to ONNX, validate runtime behavior, and connect serving steps without rebuilding your pipeline from scratch.

  - icon: V
    title: Vector and Search Tooling
    details: Plug in Annoy, Faiss, or Milvus flows to support recall experiments and retrieval system construction.

  - icon: EXP
    title: Experiment Visibility
    details: Add callbacks, tracking, and visualization to keep runs reproducible and easier to debug across datasets and models.

  - icon: DATA
    title: Unified Data Pipeline
    details: Reuse the same feature schema, data processing, and trainer patterns across ranking, matching, and tutorial examples.
---

<div class="home-badges">
  <span>Ranking</span>
  <span>Matching</span>
  <span>Multi-task</span>
  <span>Generative</span>
  <span>ONNX</span>
  <span>Vector Search</span>
  <span>Tracking</span>
  <span>Visualization</span>
</div>

<div class="home-quick-grid">
  <a class="home-quick-card" href="/guide/quick_start">
    <strong>Get your first run working</strong>
    <span>Install the project, define features, train a baseline, and understand the common trainer flow.</span>
  </a>
  <a class="home-quick-card" href="/models/intro">
    <strong>Choose the right model family</strong>
    <span>Browse ranking, matching, multi-task, and generative models with a consistent documentation structure.</span>
  </a>
  <a class="home-quick-card" href="/serving/onnx">
    <strong>Prepare for deployment</strong>
    <span>Export to ONNX, verify runtime outputs, and wire serving-friendly artifacts into downstream systems.</span>
  </a>
</div>
