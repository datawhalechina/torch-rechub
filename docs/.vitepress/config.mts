import { defineConfig } from 'vitepress'

const rootNav = [
  { text: 'Getting Started', link: '/guide/intro' },
  { text: 'Core', link: '/core/intro' },
  { text: 'Models', link: '/models/intro' },
  { text: 'Tools', link: '/tools/intro' },
  { text: 'Serving', link: '/serving/intro' },
  { text: 'Tutorials', link: '/tutorials/intro' },
  { text: 'API', link: '/api/api' },
  { text: 'Community', link: '/community/faq' },
  { text: 'Blog', link: '/blog/match' },
]

const rootSidebar = {
  '/guide/': [
    {
      text: 'Getting Started',
      items: [
        { text: 'Overview', link: '/guide/intro' },
        { text: 'Installation', link: '/guide/install' },
        { text: 'Quick Start', link: '/guide/quick_start' },
      ],
    },
  ],
  '/core/': [
    {
      text: 'Core',
      items: [
        { text: 'Overview', link: '/core/intro' },
        { text: 'Feature Columns', link: '/core/features' },
        { text: 'Data Pipeline', link: '/core/data' },
        { text: 'Training and Evaluation', link: '/core/evaluation' },
      ],
    },
  ],
  '/models/': [
    {
      text: 'Models',
      items: [
        { text: 'Overview', link: '/models/intro' },
        { text: 'Ranking Models', link: '/models/ranking' },
        { text: 'Matching Models', link: '/models/matching' },
        { text: 'Multi-Task Models', link: '/models/mtl' },
        { text: 'Generative Models', link: '/models/generative' },
      ],
    },
  ],
  '/tools/': [
    {
      text: 'Tools',
      items: [
        { text: 'Overview', link: '/tools/intro' },
        { text: 'Visualization', link: '/tools/visualization' },
        { text: 'Experiment Tracking', link: '/tools/tracking' },
        { text: 'Callbacks', link: '/tools/callbacks' },
        { text: 'Benchmark', link: '/tools/benchmark' },
      ],
    },
  ],
  '/serving/': [
    {
      text: 'Serving',
      items: [
        { text: 'Overview', link: '/serving/intro' },
        { text: 'ONNX and Quantization', link: '/serving/onnx' },
        { text: 'Vector Indexing', link: '/serving/vector_index' },
        { text: 'Serving Demo', link: '/serving/demo' },
      ],
    },
  ],
  '/tutorials/': [
    {
      text: 'Tutorials',
      items: [
        { text: 'Overview', link: '/tutorials/intro' },
        { text: 'CTR Pipeline', link: '/tutorials/ctr' },
        { text: 'Retrieval System', link: '/tutorials/retrieval' },
        { text: 'Big Data Pipeline', link: '/tutorials/pipeline' },
      ],
    },
  ],
  '/api/': [
    {
      text: 'API',
      items: [{ text: 'Main API', link: '/api/api' }],
    },
  ],
  '/community/': [
    {
      text: 'Community',
      items: [
        { text: 'FAQ', link: '/community/faq' },
        { text: 'Contributing', link: '/community/contributing' },
        { text: 'Changelog', link: '/community/changelog' },
      ],
    },
  ],
  '/blog/': [
    {
      text: 'Blog',
      items: [
        { text: 'Matching Models Guide', link: '/blog/match' },
        { text: 'Ranking Models Guide', link: '/blog/rank' },
        { text: 'HLLM Reproduction', link: '/blog/hllm_reproduction' },
      ],
    },
  ],
}

const zhNav = [
  { text: '快速开始', link: '/zh/guide/intro' },
  { text: '核心组件', link: '/zh/core/intro' },
  { text: '模型库', link: '/zh/models/intro' },
  { text: '工具', link: '/zh/tools/intro' },
  { text: '部署', link: '/zh/serving/intro' },
  { text: '教程', link: '/zh/tutorials/intro' },
  { text: 'API', link: '/zh/api/api' },
  { text: '社区', link: '/zh/community/faq' },
  { text: '博客', link: '/zh/blog/match' },
]

const zhSidebar = {
  '/zh/guide/': [
    {
      text: '快速开始',
      items: [
        { text: '概览', link: '/zh/guide/intro' },
        { text: '安装指南', link: '/zh/guide/install' },
        { text: '快速上手', link: '/zh/guide/quick_start' },
      ],
    },
  ],
  '/zh/core/': [
    {
      text: '核心组件',
      items: [
        { text: '概览', link: '/zh/core/intro' },
        { text: '特征定义', link: '/zh/core/features' },
        { text: '数据流水线', link: '/zh/core/data' },
        { text: '训练与评估', link: '/zh/core/evaluation' },
      ],
    },
  ],
  '/zh/models/': [
    {
      text: '模型库',
      items: [
        { text: '概览', link: '/zh/models/intro' },
        { text: '排序模型', link: '/zh/models/ranking' },
        { text: '召回模型', link: '/zh/models/matching' },
        { text: '多任务模型', link: '/zh/models/mtl' },
        { text: '生成式模型', link: '/zh/models/generative' },
      ],
    },
  ],
  '/zh/tools/': [
    {
      text: '工具',
      items: [
        { text: '概览', link: '/zh/tools/intro' },
        { text: '可视化', link: '/zh/tools/visualization' },
        { text: '实验追踪', link: '/zh/tools/tracking' },
        { text: '回调函数', link: '/zh/tools/callbacks' },
        { text: 'Benchmark', link: '/zh/tools/benchmark' },
      ],
    },
  ],
  '/zh/serving/': [
    {
      text: '部署',
      items: [
        { text: '概览', link: '/zh/serving/intro' },
        { text: 'ONNX 与量化', link: '/zh/serving/onnx' },
        { text: '向量索引', link: '/zh/serving/vector_index' },
        { text: '部署示例', link: '/zh/serving/demo' },
      ],
    },
  ],
  '/zh/tutorials/': [
    {
      text: '教程',
      items: [
        { text: '概览', link: '/zh/tutorials/intro' },
        { text: 'CTR 流程', link: '/zh/tutorials/ctr' },
        { text: '召回系统', link: '/zh/tutorials/retrieval' },
        { text: '大数据流水线', link: '/zh/tutorials/pipeline' },
      ],
    },
  ],
  '/zh/api/': [
    {
      text: 'API',
      items: [{ text: 'API 参考', link: '/zh/api/api' }],
    },
  ],
  '/zh/community/': [
    {
      text: '社区',
      items: [
        { text: '常见问题', link: '/zh/community/faq' },
        { text: '贡献指南', link: '/zh/community/contributing' },
        { text: '版本日志', link: '/zh/community/changelog' },
      ],
    },
  ],
  '/zh/blog/': [
    {
      text: '博客',
      items: [
        { text: '召回模型训练指南', link: '/zh/blog/match' },
        { text: '排序模型训练指南', link: '/zh/blog/rank' },
        { text: 'HLLM 复现说明', link: '/zh/blog/hllm_reproduction' },
        { text: 'HSTU 复现说明', link: '/zh/blog/hstu_reproduction' },
      ],
    },
  ],
}

export default defineConfig({
  title: 'torch-rechub',
  description:
    'A lightweight PyTorch framework for recommendation systems with ranking, matching, multi-task, and serving workflows.',
  base: '/torch-rechub/',
  head: [
    ['link', { rel: 'icon', href: '/torch-rechub/favicon.ico' }],
    ['meta', { name: 'theme-color', content: '#0f766e' }],
  ],
  markdown: {
    math: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark',
    },
  },
  rewrites: {
    'en/:rest*': ':rest*',
  },
  themeConfig: {
    logo: '/img/logo.png',
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/datawhalechina/torch-rechub' },
    ],
    outline: {
      level: [2, 3],
    },
  },
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      themeConfig: {
        nav: rootNav,
        sidebar: rootSidebar,
        outlineTitle: 'On this page',
        returnToTopLabel: 'Back to top',
        darkModeSwitchLabel: 'Appearance',
        lightModeSwitchTitle: 'Switch to light theme',
        darkModeSwitchTitle: 'Switch to dark theme',
        sidebarMenuLabel: 'Menu',
        docFooter: {
          prev: 'Previous page',
          next: 'Next page',
        },
      },
    },
    zh: {
      label: '中文',
      lang: 'zh-CN',
      link: '/zh/',
      themeConfig: {
        nav: zhNav,
        sidebar: zhSidebar,
        outlineTitle: '本页目录',
        returnToTopLabel: '返回顶部',
        darkModeSwitchLabel: '外观',
        lightModeSwitchTitle: '切换到浅色模式',
        darkModeSwitchTitle: '切换到深色模式',
        sidebarMenuLabel: '菜单',
        docFooter: {
          prev: '上一页',
          next: '下一页',
        },
      },
    },
  },
})
