---
# template: home.html # 可选，使用 Material for MkDocs 的着陆页模板  <- 删除或注释掉此行
hide:
  - navigation # 隐藏左侧导航
  - toc # 隐藏右侧目录
title: 欢迎使用 Torch-RecHub
---

<style>
  .md-typeset h1, .md-content__button { display: none; } /* 隐藏默认标题和编辑按钮 */
  .mdx-container {
    background: 
      url('../file/img/homepage.png');
    background-size: contain;  /* 修改为 contain 保持完整比例 */
    background-position: center center;
    background-repeat: no-repeat;
    min-height: 78vh;
    width: 100%;
    position: relative;
    overflow: hidden;  /* 隐藏溢出部分 */
  }
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
      <img src="../file/img/logo.png" alt="Torch-RecHub Logo" width="100">
      <h1 style="text-align: center; font-size: 3em; margin-top: 1em; margin-bottom: 0.5em;">Torch-RecHub</h1>
      <p style="text-align: center; font-size: 1.5em; margin-bottom: 1.5em; 
           text-shadow: 
             0 0 2.5px #fff, 
             0 0 2.5px #fff, 
             0 0 2.5px #fff, 
             0 0 2.5px #fff;">一个基于 PyTorch 的易用、可扩展且高性能的推荐系统框架</p>
      <div style="text-align: center;">
        <a href="getting-started/" class="md-button md-button--primary md-button--lg">
          快速开始
        </a>
        <a href="https://github.com/datawhalechina/torch-rechub" target="_blank" rel="noopener" class="md-button md-button--lg">
          查看 GitHub
        </a>
      </div>
    </div>
  </div>
</section>
