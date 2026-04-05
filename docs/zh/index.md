---
layout: home

hero:
  name: "Torch-RecHub"
  text: "把推荐系统从研究验证顺畅推进到生产部署"
  tagline: "基于轻量 PyTorch 栈，统一构建排序、召回、多任务、生成式与服务化流程，同时保持足够易用和易扩展。"
  image:
    src: /img/logo.png
    alt: Torch-RecHub
  actions:
    - theme: brand
      text: 快速开始
      link: /zh/guide/quick_start
    - theme: alt
      text: 模型总览
      link: /zh/models/intro
    - theme: alt
      text: ONNX 部署
      link: /zh/serving/onnx

features:
  - icon: 排序
    title: 排序模型工作流
    details: 直接上手 WideDeep、DeepFM、DIN、BST、DIEN、DCN 等主流排序模型，并沿用统一训练流程。

  - icon: 召回
    title: 召回与检索能力
    details: 覆盖双塔召回、序列召回、多兴趣建模、向量索引和召回评估，方便从实验走向系统搭建。

  - icon: 多任务
    title: 多任务学习
    details: 用一致的特征定义和 Trainer 方式训练 ESMM、MMOE、PLE、AITM 等多任务架构。

  - icon: 生成式
    title: 生成式扩展
    details: 在同一套数据、训练和部署约定下，探索更新的推荐建模方向，而不用重搭基础设施。

  - icon: ONNX
    title: 面向部署
    details: 支持导出 ONNX、校验运行时结果，并衔接后续服务化流程，减少从训练到上线的摩擦。

  - icon: 向量
    title: 向量检索工具链
    details: 可接入 Annoy、Faiss、Milvus 等能力，用于召回实验、向量索引和检索系统原型搭建。

  - icon: 实验
    title: 实验可见性
    details: 结合回调、实验追踪和可视化能力，让训练过程更可复现，也更容易排查问题。

  - icon: 数据
    title: 统一数据流水线
    details: 在排序、召回和教程案例之间复用同一套特征规范、数据处理方式和训练入口。
---

<div class="home-badges">
  <span>排序</span>
  <span>召回</span>
  <span>多任务</span>
  <span>生成式</span>
  <span>ONNX</span>
  <span>向量检索</span>
  <span>实验追踪</span>
  <span>可视化</span>
</div>

<div class="home-quick-grid">
  <a class="home-quick-card" href="/zh/guide/quick_start">
    <strong>先把第一条训练链路跑通</strong>
    <span>完成安装、定义特征、训练基线模型，并快速理解这个项目最常用的 Trainer 工作流。</span>
  </a>
  <a class="home-quick-card" href="/zh/models/intro">
    <strong>快速挑选模型家族</strong>
    <span>按排序、召回、多任务、生成式四条主线浏览能力，建立对项目模型结构的全局认识。</span>
  </a>
  <a class="home-quick-card" href="/zh/serving/onnx">
    <strong>面向部署准备产物</strong>
    <span>了解 ONNX 导出、运行时校验与服务化对接方式，让实验结果更容易走向实际使用。</span>
  </a>
</div>
