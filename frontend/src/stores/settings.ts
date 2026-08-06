import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "../api/endpoints"

export const useSettingsStore = defineStore("settings", () => {
  const ragEnabled = ref(localStorage.getItem("qa-rag") !== "0")
  const visualEnabled = ref(localStorage.getItem("qa-visual") === "1")
  const kgEnabled = ref(localStorage.getItem("qa-kg") === "1")
  const agentMode = ref("reactive")
  const model = ref("")
  const checkpointEnabled = ref(false)

  function setRag(v: boolean) {
    ragEnabled.value = v
    localStorage.setItem("qa-rag", v ? "1" : "0")
  }
  function setVisual(v: boolean) {
    visualEnabled.value = v
    localStorage.setItem("qa-visual", v ? "1" : "0")
  }
  function setKg(v: boolean) {
    kgEnabled.value = v
    localStorage.setItem("qa-kg", v ? "1" : "0")
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
    visualEnabled,
    kgEnabled,
    agentMode,
    model,
    checkpointEnabled,
    setRag,
    setVisual,
    setKg,
    loadConfig,
  }
})
