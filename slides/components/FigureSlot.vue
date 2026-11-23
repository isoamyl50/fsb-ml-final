<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    src?: string
    expectedFile: string
    caption: string
    hint?: string
    maxHeight?: string
  }>(),
  {
    src: '',
    hint: '',
    maxHeight: '300px',
  },
)

const hasImage = computed(() => Boolean(props.src && props.src.trim().length > 0))
</script>

<template>
  <figure v-if="hasImage" class="flex flex-col items-center justify-center my-1">
    <img
      :src="src"
      :alt="caption"
      :style="{ maxHeight }"
      class="mx-auto object-contain rounded border border-slate-200 bg-white p-1 shadow-sm"
    />
    <figcaption class="mt-1.5 text-center text-xs text-slate-600 leading-snug">
      {{ caption }}
    </figcaption>
  </figure>

  <div
    v-else
    :style="{ minHeight: maxHeight === '195px' ? '150px' : '180px' }"
    class="w-full rounded-lg border-2 border-dashed border-slate-300 bg-slate-50/90 px-4 py-3 flex flex-col items-center justify-center text-center gap-1.5"
  >
    <div class="inline-flex items-center gap-1.5 rounded-full bg-amber-100 border border-amber-300 px-2.5 py-0.5 text-[11px] font-semibold text-amber-800">
      <span>CHỜ BIỂU ĐỒ TỪ NOTEBOOK</span>
    </div>
    <code class="text-[11px] bg-white px-2 py-0.5 rounded border border-slate-200 text-slate-700">
      {{ expectedFile }}
    </code>
    <p class="text-xs font-medium text-slate-700 m-0 leading-snug">
      {{ caption }}
    </p>
    <p v-if="hint" class="text-[11px] text-slate-500 m-0 leading-snug max-w-md">
      {{ hint }}
    </p>
  </div>
</template>
