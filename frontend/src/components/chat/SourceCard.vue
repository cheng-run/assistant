<script setup lang="ts">
import { computed, ref } from "vue"
import type { Source } from "../../types"

const props = defineProps<{ source: Source }>()
const open = ref(false)

const heading = computed(
  () => props.source.heading_path?.join(" › ") || props.source.source_file || "文档",
)
const scorePct = computed(() =>
  props.source.score != null ? `${Math.round(props.source.score * 100)}%` : null,
)
</script>

<template>
  <div class="source-card" @click="open = !open">
    <div class="sc-top">
      <span class="sc-file" :title="source.source_file">📄 {{ source.source_file }}</span>
      <span v-if="scorePct" class="sc-score">{{ scorePct }}</span>
    </div>
    <div class="sc-heading" :title="heading">{{ heading }}</div>
    <div class="sc-content" :class="{ open }">{{ source.content }}</div>
  </div>
</template>

<style scoped>
.source-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 9px 11px;
  cursor: pointer;
  transition: border-color 0.18s ease, background 0.18s ease;
}
.source-card:hover {
  border-color: var(--border-strong);
  background: var(--surface-2);
}
.sc-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 6px;
}
.sc-file {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-2);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.sc-score {
  font-family: var(--mono);
  font-size: 10px;
  color: var(--accent-hover);
  background: var(--accent-soft);
  padding: 1px 6px;
  border-radius: 10px;
  flex-shrink: 0;
}
.sc-heading {
  font-size: 11px;
  color: var(--text-3);
  margin: 4px 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.sc-content {
  font-size: 12px;
  color: var(--text-2);
  line-height: 1.5;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}
.sc-content.open {
  -webkit-line-clamp: unset;
  display: block;
}
</style>
