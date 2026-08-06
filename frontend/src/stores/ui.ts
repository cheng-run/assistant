import { defineStore } from "pinia"
import { ref } from "vue"

export const useUiStore = defineStore("ui", () => {
  const settingsOpen = ref(false)
  return { settingsOpen }
})
