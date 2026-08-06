<script setup lang="ts">
import { computed } from "vue"
import { useUploadStore } from "../../stores/upload"

const upload = useUploadStore()
const pct = computed(() => Math.round(upload.progress * 100))
const isError = computed(() => !!upload.error)
</script>

<template>
  <div class="up-progress">
    <div class="up-bar"><div class="up-fill" :class="{ error: isError }" :style="{ width: `${pct}%` }"></div></div>
    <div class="up-label" :class="{ error: isError }">
      <span>{{ upload.label || "等待中…" }}</span>
      <span v-if="!isError" class="up-pct">{{ pct }}%</span>
    </div>
    <div v-if="upload.lastResult && !isError" class="up-result">
      ✓ {{ upload.lastResult.source_file }} · {{ upload.lastResult.chunks }} 片段
      <span v-if="upload.lastResult.kg"> · {{ upload.lastResult.kg }}</span>
    </div>
  </div>
</template>

<style scoped>
.up-progress {
  display: flex;
  flex-direction: column;
  gap: 5px;
  padding: 6px 2px;
}
.up-bar {
  height: 4px;
  border-radius: 4px;
  background: var(--surface-3);
  overflow: hidden;
}
.up-fill {
  height: 100%;
  border-radius: 4px;
  background: linear-gradient(90deg, var(--accent), #60a5fa);
  transition: width 0.3s ease;
}
.up-fill.error { background: #ef4444; }
.up-label {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: var(--text-2);
}
.up-label.error { color: #ef4444; }
.up-pct { font-family: var(--mono); color: var(--text-3); }
.up-result {
  font-size: 11px;
  color: #22c55e;
}
</style>
