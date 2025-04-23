---
# template: home.html # 可选，使用 Material for MkDocs 的着陆页模板  <- 删除或注释掉此行
hide:
  - navigation # 隐藏左侧导航
  - toc # 隐藏右侧目录
title: Welcome to use Torch-RecHub
---

<style>
  .md-typeset h1, .md-content__button { display: none; } /* 隐藏默认标题和编辑按钮 */
  /* 更新特性项样式以适应卡片 */
  .feature-card {
    text-align: center;
    border: 1px solid var(--md-default-fg-color--lightest); /* 可选：添加边框 */
    border-radius: 4px; /* 可选：添加圆角 */
    height: 100%; /* 让卡片等高 */
    display: flex; /* 使用 flex 布局 */
    flex-direction: column; /* 垂直排列 */
    justify-content: flex-start; /* 从顶部开始对齐 */
  }
  .feature-card .md-card__content {
      flex-grow:
  .feature-item { text-align: center; }
  .feature-icon { font-size: 3em; margin-bottom: 0.5em; }
</style>

<!-- Hero Section -->
<section class="mdx-container">
  <div class="md-grid md-typeset">
    <div class="md-grid__cell md-grid__cell--center">
      <!-- <img src="assets/logo.png" alt="Torch-RecHub Logo" width="200"> -->
      <h1 style="text-align: center; font-size: 3em; margin-top: 1em; margin-bottom: 0.5em;">Torch-RecHub</h1>
      <p style="text-align: center; font-size: 1.5em; margin-bottom: 1.5em;">An easy-to-use, scalable, and high-performance recommendation system framework based on PyTorch</p>
      <div style="text-align: center;">
        <a href="getting-started/" class="md-button md-button--primary md-button--lg">
          Quick Start
        </a>
        <a href="https://github.com/datawhalechina/torch-rechub" target="_blank" rel="noopener" class="md-button md-button--lg">
          View GitHub
        </a>
      </div>
    </div>
  </div>
</section>
