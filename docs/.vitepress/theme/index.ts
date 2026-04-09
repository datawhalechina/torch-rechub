// https://vitepress.dev/guide/custom-theme
import { h } from 'vue'
import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme'
import imageViewer from 'vitepress-plugin-image-viewer'
import vImageViewer from 'vitepress-plugin-image-viewer/lib/vImageViewer.vue'
import { useRoute } from 'vitepress'
import ScrollTopProgress from './components/ScrollTopProgress.vue'
import './style.css'
import './custom.css'

export default {
  extends: DefaultTheme,
  Layout: () =>
    h(DefaultTheme.Layout, null, {
      'layout-bottom': () => h(ScrollTopProgress),
    }),
  enhanceApp({ app }) {
    app.component('vImageViewer', vImageViewer)
  },
  setup() {
    const route = useRoute()
    imageViewer(route)
  },
} satisfies Theme
