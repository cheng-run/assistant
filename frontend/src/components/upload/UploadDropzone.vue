<script setup lang="ts">
import { ref } from "vue"
import { useUploadStore } from "../../stores/upload"

const emit = defineEmits<{ done: [] }>()
const upload = useUploadStore()
const drag = ref(false)
const fileInput = ref<HTMLInputElement>()

const ACCEPT = [".pdf", ".docx", ".md", ".txt"]

function pick() {
  fileInput.value?.click()
}

function onFile(f?: File | null) {
  if (!f) return
  const ok = ACCEPT.some((x) => f.name.toLowerCase().endsWith(x))
  if (!ok) {
    window.alert("仅支持 PDF / DOCX / MD / TXT")
    return
  }
  void upload.upload(f).then(() => {
    if (upload.lastResult && !upload.error) emit("done")
  })
}

function onDrop(e: DragEvent) {
  drag.value = false
  onFile(e.dataTransfer?.files?.[0])
}
</script>

<template>
  <div
    class="dropzone"
    :class="{ drag }"
    @click="pick"
    @dragover.prevent="drag = true"
    @dragleave="drag = false"
    @drop.prevent="onDrop"
  >
    <input
      ref="fileInput"
      type="file"
      accept=".pdf,.docx,.md,.txt"
      class="hidden-input"
      @change="onFile(($event.target as HTMLInputElement).files?.[0])"
    />
    <span class="dz-icon">⤒</span>
    <span class="dz-text">{{ drag ? "松开上传" : "拖入或点击上传" }}</span>
    <span class="dz-formats">PDF · DOCX · MD · TXT</span>
  </div>
</template>

<style scoped>
.dropzone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
  padding: 16px 10px;
  border: 1px dashed var(--border-strong);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: center;
}
.dropzone:hover,
.dropzone.drag {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.dz-icon { font-size: 18px; color: var(--text-3); }
.dropzone:hover .dz-icon { color: var(--accent-hover); }
.dz-text { font-size: 12.5px; color: var(--text-2); }
.dz-formats {
  font-size: 10px;
  color: var(--text-3);
  letter-spacing: 0.04em;
}
.hidden-input { display: none; }
</style>
