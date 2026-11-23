<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    label: string
    value: string
    subtext?: string
    accent?: 'blue' | 'emerald' | 'amber' | 'slate'
  }>(),
  {
    subtext: '',
    accent: 'blue',
  },
)

const borderClass = computed(() => {
  switch (props.accent) {
    case 'emerald':
      return 'border-emerald-200 border-t-3 border-t-emerald-600 bg-emerald-50/40'
    case 'amber':
      return 'border-amber-200 border-t-3 border-t-amber-600 bg-amber-50/40'
    case 'slate':
      return 'border-slate-200 border-t-3 border-t-slate-500 bg-slate-50'
    default:
      return 'border-blue-200 border-t-3 border-t-blue-700 bg-slate-50'
  }
})

const valueClass = computed(() => {
  switch (props.accent) {
    case 'emerald':
      return 'text-emerald-800'
    case 'amber':
      return 'text-amber-800'
    case 'slate':
      return 'text-slate-800'
    default:
      return 'text-blue-900'
  }
})
</script>

<template>
  <div class="rounded-lg border px-3 py-2 flex flex-col justify-between" :class="borderClass">
    <div class="text-[11px] font-semibold uppercase tracking-wide text-slate-500">
      {{ label }}
    </div>
    <div class="text-lg font-bold tabular-nums leading-tight mt-0.5" :class="valueClass">
      {{ value }}
    </div>
    <div v-if="subtext" class="text-[11px] text-slate-600 mt-0.5 leading-tight">
      {{ subtext }}
    </div>
  </div>
</template>
