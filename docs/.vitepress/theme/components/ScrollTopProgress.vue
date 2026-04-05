<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { useData } from 'vitepress'

const { lang } = useData()

const progress = ref(0)
const visible = ref(false)

const updateScrollState = () => {
  const root = document.documentElement
  const scrollTop = window.scrollY || root.scrollTop || 0
  const scrollable = Math.max(root.scrollHeight - window.innerHeight, 1)
  const ratio = Math.min(scrollTop / scrollable, 1)

  progress.value = Math.round(ratio * 100)
  visible.value = scrollTop > 240
}

const scrollToTop = () => {
  window.scrollTo({ top: 0, behavior: 'smooth' })
}

const label = computed(() => (lang.value.startsWith('zh') ? '返回顶部' : 'Back to top'))

const progressStyle = computed(() => ({
  '--scroll-progress': `${progress.value}%`,
}))

onMounted(() => {
  updateScrollState()
  window.addEventListener('scroll', updateScrollState, { passive: true })
  window.addEventListener('resize', updateScrollState, { passive: true })
})

onBeforeUnmount(() => {
  window.removeEventListener('scroll', updateScrollState)
  window.removeEventListener('resize', updateScrollState)
})
</script>

<template>
  <Transition name="scroll-progress-fade">
    <button
      v-if="visible"
      class="scroll-progress-button"
      :style="progressStyle"
      type="button"
      :aria-label="label"
      :title="label"
      @click="scrollToTop"
    >
      <span class="scroll-progress-ring" aria-hidden="true">
        <span class="scroll-progress-core">
          <span class="scroll-progress-arrow">↑</span>
          <span class="scroll-progress-value">{{ progress }}%</span>
        </span>
      </span>
    </button>
  </Transition>
</template>
