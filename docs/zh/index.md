---
layout: home

hero:
  name: "Torch-RecHub"
  text: "基于 PyTorch 的轻量推荐系统框架"
  tagline: "用浏览、点击和购买记录训练推荐模型。可以先跑仓库自带的样本，再换成自己的数据。"
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
    title: 点击率预估
    details: 给定一个用户和一件候选商品，用 DeepFM 或 DIN 估计点击概率。
    link: /zh/models/ranking
    linkText: 查看排序模型

  - icon: 召回
    title: 候选召回
    details: 用 DSSM 学习用户和物品向量，或用 MIND 为同一个用户保留多个兴趣向量。
    link: /zh/models/matching
    linkText: 查看召回模型

  - icon: 多任务
    title: 多任务学习
    details: 用同一批样本学习点击和转化，分别查看每个目标的指标。
    link: /zh/models/mtl
    linkText: 查看多任务模型

  - icon: 生成式
    title: 序列与生成式推荐
    details: 根据交互历史预测下一个物品，或用 TIGER 逐步生成物品的语义 ID。
    link: /zh/models/generative
    linkText: 查看生成式模型

  - icon: ONNX
    title: 导出 ONNX
    details: 导出训练好的模型，用同一条样本核对 PyTorch 与 ONNX Runtime 的预测结果。
    link: /zh/serving/onnx
    linkText: 查看 ONNX 部署

  - icon: 向量
    title: 向量检索
    details: 为物品向量建立索引，用用户向量查找最相似的物品。可选 Annoy、Faiss 或 Milvus。
    link: /zh/serving/vector_index
    linkText: 查看向量索引

  - icon: 实验
    title: 实验记录
    details: 在 TensorBoard、W&B 或 SwanLab 中查看每轮的训练损失和验证指标。
    link: /zh/tools/tracking
    linkText: 查看实验追踪

  - icon: 数据
    title: 准备训练数据
    details: 把已编码的特征和标签分批送入训练器。较大的 Parquet 文件可按批读取。
    link: /zh/core/data
    linkText: 查看数据流水线
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
  <a class="home-quick-card" href="./guide/quick_start">
    <strong>第一次训练 DeepFM</strong>
    <span>用 Criteo 点击样本在 CPU 上训练两轮，查看测试集 AUC。</span>
  </a>
  <a class="home-quick-card" href="./models/intro">
    <strong>按任务找模型</strong>
    <span>从已有的数据出发，选择召回、排序、多任务或生成式推荐模型。</span>
  </a>
  <a class="home-quick-card" href="./serving/demo">
    <strong>用训练好的模型做预测</strong>
    <span>给 PyTorch 和 ONNX Runtime 同一条曝光记录，核对两个点击分数。</span>
  </a>
</div>
