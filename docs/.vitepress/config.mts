import { defineConfig } from 'vitepress'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "torch-rechub",
  description: "A Lighting Pytorch Framework for Recommendation Models, Easy-to-use and Easy-to-extend.",
  head: [
    ['link', { rel: 'icon', href: '/torch-rechub/favicon.ico' }]
  ],
  // GitHub Pages 部署路径（仓库名）
  base: '/torch-rechub/',

  // 路径重写：将 en/ 目录映射到根路径
  rewrites: {
    'en/:rest*': ':rest*'
  },

  // 多语言配置
  locales: {
    root: {
      label: 'English',
      lang: 'en',
      title: 'torch-rechub',
      description: 'A Lighting Pytorch Framework for Recommendation Models, Easy-to-use and Easy-to-extend.',
      themeConfig: {
        nav: [
          { text: 'Home', link: '/' },
          { text: 'Introduction', link: '/introduction' },
          { text: 'Documentation', link: '/manual/installation' },
          { text: 'API', link: '/manual/api-reference/basic' },
          { text: 'Blog', link: '/blog/match' },
          { text: 'Contributing', link: '/contributing' }
        ],
        sidebar: {
          '/manual/': [
            {
              text: 'Getting Started',
              items: [
                { text: 'Installation', link: '/manual/installation' },
                { text: 'Getting Started', link: '/manual/getting-started' }
              ]
            },
            {
              text: 'Tutorials',
              items: [
                { text: 'Recall Models', link: '/manual/tutorials/matching' },
                { text: 'Ranking Models', link: '/manual/tutorials/ranking' },
                { text: 'Multi-Task Learning', link: '/manual/tutorials/multi-task' }
              ]
            },
            {
              text: 'API Reference',
              items: [
                { text: 'Basic Components', link: '/manual/api-reference/basic' },
                { text: 'Models', link: '/manual/api-reference/models' },
                { text: 'Trainers', link: '/manual/api-reference/trainers' },
                { text: 'Utilities', link: '/manual/api-reference/utils' }
              ]
            },
            {
              text: 'Others',
              items: [
                { text: 'FAQ', link: '/manual/faq' }
              ]
            }
          ],
          '/blog/': [
            {
              text: 'Blog',
              items: [
                { text: 'Matching Training Guide', link: '/blog/match' },
                { text: 'Ranking Training Guide', link: '/blog/rank' },
                { text: 'HSTU Reproduction', link: '/blog/hstu_reproduction' },
                { text: 'HLLM Reproduction', link: '/blog/hllm_reproduction' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/torch-rechub' }
        ]
      }
    },
    zh: {
      label: '中文',
      lang: 'zh-CN',
      title: 'torch-rechub',
      description: '一个基于 PyTorch 的易用、可扩展且高性能的推荐系统框架',
      themeConfig: {
        nav: [
          { text: '首页', link: '/zh/' },
          { text: '项目介绍', link: '/zh/introduction' },
          { text: '项目文档', link: '/zh/manual/installation' },
          { text: 'API', link: '/zh/manual/api-reference/basic' },
          { text: '博客', link: '/zh/blog/match' },
          { text: '贡献指南', link: '/zh/contributing' }
        ],
        sidebar: {
          '/zh/manual/': [
            {
              text: '入门指南',
              items: [
                { text: '安装指南', link: '/zh/manual/installation' },
                { text: '快速开始', link: '/zh/manual/getting-started' }
              ]
            },
            {
              text: '教程',
              items: [
                { text: '召回模型', link: '/zh/manual/tutorials/matching' },
                { text: '排序模型', link: '/zh/manual/tutorials/ranking' },
                { text: '多任务学习', link: '/zh/manual/tutorials/multi-task' }
              ]
            },
            {
              text: 'API 参考',
              items: [
                { text: '基础组件', link: '/zh/manual/api-reference/basic' },
                { text: '模型', link: '/zh/manual/api-reference/models' },
                { text: '训练器', link: '/zh/manual/api-reference/trainers' },
                { text: '工具类', link: '/zh/manual/api-reference/utils' }
              ]
            },
            {
              text: '其他',
              items: [
                { text: '常见问题', link: '/zh/manual/faq' },
                { text: '参考资料', link: '/zh/参考资料/参考资料' }
              ]
            }
          ],
          '/zh/blog/': [
            {
              text: '博客',
              items: [
                { text: '召回模型训练指南', link: '/zh/blog/match' },
                { text: '排序模型训练指南', link: '/zh/blog/rank' },
                { text: 'HSTU 模型复现', link: '/zh/blog/hstu_reproduction' },
                { text: 'HLLM 模型复现', link: '/zh/blog/hllm_reproduction' }
              ]
            }
          ]
        },
        socialLinks: [
          { icon: 'github', link: 'https://github.com/datawhalechina/torch-rechub' }
        ]
      }
    }
  },

  themeConfig: {
    logo: '/img/logo.png'
  }
})
