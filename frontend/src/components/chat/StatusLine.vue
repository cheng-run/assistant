<script setup lang="ts">
import { computed } from "vue"
import type { StatusKind } from "../../types"
import { renderMarkdown } from "../../utils/markdown"

const props = defineProps<{ status: { kind: StatusKind; text: string } }>()

const icon: Record<StatusKind, string> = {
  tool: "🔧",
  plan: "📋",
  subagent: "👤",
}
const html = computed(() => renderMarkdown(props.status.text).replace(/^<p>|<\/p>$/g, ""))
</script>

<template>
  <div class="status-line" :class="status.kind">
    <span class="status-icon">{{ icon[status.kind] ?? "•" }}</span>
    <span class="status-text" v-html="html"></span>
  </div>
</template>

<style scoped>
.status-line {
  display: flex;
  align-items: baseline;
  gap: 7px;
  font-size: 12.5px;
  color: var(--text-3);
  animation: fade-in 0.25s ease both;
}
.status-icon { flex-shrink: 0; font-size: 12px; }
.status-text { line-height: 1.5; }
.status-text :deep(p) { margin: 0; }
.status-line.subagent { color: var(--accent-hover); }
.status-line.tool .status-icon { opacity: 0.85; }
</style>
