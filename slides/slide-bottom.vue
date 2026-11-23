<script setup lang="ts">
import { computed } from 'vue'
import { useSlideContext } from '@slidev/client'

const { $frontmatter } = useSlideContext()

const hideFooter = computed(() => {
  const layout = $frontmatter?.layout
  return layout === 'cover' || layout === 'end' || $frontmatter?.hideFooter === true
})
</script>

<template>
  <footer
    v-if="!hideFooter"
    class="absolute bottom-2 left-12 right-12 flex items-center justify-between text-[11px] text-slate-400 pointer-events-none select-none"
  >
    <span>AI &amp; Machine Learning — Đồ án Cuối kỳ</span>
    <span class="tabular-nums font-medium">
      <SlideCurrentNo /> / <SlidesTotal />
    </span>
  </footer>
</template>
