import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "../api/endpoints"
import { consumeSSE } from "../api/sse"
import type { UploadResult } from "../types"

export const useUploadStore = defineStore("upload", () => {
  const isUploading = ref(false)
  const progress = ref(0)
  const label = ref("")
  const stage = ref("")
  const lastResult = ref<UploadResult | null>(null)
  const error = ref("")

  async function upload(file: File) {
    isUploading.value = true
    progress.value = 0
    label.value = "准备上传..."
    stage.value = ""
    error.value = ""
    lastResult.value = null

    const form = new FormData()
    form.append("file", file)
    // 视觉/KG 索引由后端自动处理（无需前端传参）

    const res = await fetch(api.uploadUrl(), { method: "POST", body: form })
    try {
      await consumeSSE(res, {
        stage: (d) => {
          stage.value = String(d.stage ?? "")
          progress.value = Number(d.pct ?? 0)
          label.value = String(d.label ?? "")
        },
        progress: (d) => {
          progress.value = Number(d.pct ?? 0)
          label.value = String(d.label ?? "")
        },
        result: (d) => {
          lastResult.value = d as unknown as UploadResult
          progress.value = 1
          label.value = "✅ 索引完成"
        },
        error: (d) => {
          error.value = String(d.message ?? "索引失败")
          label.value = "❌ 索引失败"
        },
      })
    } finally {
      isUploading.value = false
    }
  }

  function reset() {
    progress.value = 0
    label.value = ""
    lastResult.value = null
    error.value = ""
  }

  return { isUploading, progress, label, stage, lastResult, error, upload, reset }
})
