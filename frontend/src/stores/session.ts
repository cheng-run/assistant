import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "../api/endpoints"
import type { Session } from "../types"

export const useSessionStore = defineStore("session", () => {
  const sessions = ref<Session[]>([])
  const currentId = ref<string | null>(localStorage.getItem("qa-current-session"))

  async function fetchSessions() {
    sessions.value = await api.sessions.list()
  }

  async function createSession(): Promise<Session> {
    const s = await api.sessions.create()
    sessions.value.unshift(s)
    setCurrent(s.id)
    return s
  }

  function setCurrent(id: string | null) {
    currentId.value = id
    if (id) localStorage.setItem("qa-current-session", id)
    else localStorage.removeItem("qa-current-session")
  }

  async function deleteSession(id: string) {
    await api.sessions.remove(id)
    sessions.value = sessions.value.filter((s) => s.id !== id)
    if (currentId.value === id) setCurrent(null)
  }

  async function renameSession(id: string, title: string) {
    const s = await api.sessions.rename(id, title)
    const i = sessions.value.findIndex((x) => x.id === id)
    if (i >= 0) sessions.value[i] = s
  }

  return { sessions, currentId, fetchSessions, createSession, setCurrent, deleteSession, renameSession }
})
