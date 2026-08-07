import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "../api/endpoints"

export const useSettingsStore = defineStore("settings", () => {
  // 只保留"文档问答"总开关；视觉/图谱由后端路由引擎自动决策
  const ragEnabled = ref(localStorage.getItem("qa-rag") !== "0")
  const agentMode = ref("reactive")
  const model = ref("")
  const checkpointEnabled = ref(false)

  function setRag(v: boolean) {
    ragEnabled.value = v
    localStorage.setItem("qa-rag", v ? "1" : "0")
  }

  async function loadConfig() {
    try {
      const cfg = await api.config()
      agentMode.value = cfg.agent_mode
      model.value = cfg.model
      checkpointEnabled.value = cfg.checkpoint_enabled
    } catch {
      /* 服务不可达时保持默认 */
    }
  }

  return {
    ragEnabled,
    agentMode,
    model,
    checkpointEnabled,
    setRag,
    loadConfig,
  }
})
