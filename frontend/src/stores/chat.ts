import { defineStore } from "pinia"
import { ref } from "vue"
import { api } from "../api/endpoints"
import { consumeSSE } from "../api/sse"
import type { Message, Source, StatusKind } from "../types"
import { useSessionStore } from "./session"
import { useSettingsStore } from "./settings"

interface StatusItem {
  kind: StatusKind
  text: string
}

export const useChatStore = defineStore("chat", () => {
  const messagesBySession = ref<Record<string, Message[]>>({})
  const streaming = ref(false)
  const answerText = ref("") // 流式中的完整答案文本
  const statuses = ref<StatusItem[]>([])
  const sources = ref<Source[]>([])
  const streamError = ref("")
  let controller: AbortController | null = null

  const sessionStore = useSessionStore()
  const settingsStore = useSettingsStore()

  function localMessages(sessionId: string): Message[] {
    return messagesBySession.value[sessionId] ?? []
  }

  async function loadMessages(sessionId: string) {
    if (messagesBySession.value[sessionId]) return
    messagesBySession.value[sessionId] = await api.sessions.messages(sessionId)
  }

  function abort() {
    controller?.abort()
  }

  function pushLocal(sid: string, m: Partial<Message> & { role: "user" | "assistant" }) {
    const arr = messagesBySession.value[sid] ?? []
    arr.push({
      id: -Date.now(),
      session_id: sid,
      content: "",
      sources: null,
      created_at: Date.now(),
      ...m,
    } as Message)
    messagesBySession.value[sid] = [...arr]
  }

  async function sendMessage(text: string) {
    const sid = sessionStore.currentId
    if (!sid || streaming.value) return
    const trimmed = text.trim()
    if (!trimmed) return

    pushLocal(sid, { role: "user", content: trimmed })
    streaming.value = true
    answerText.value = ""
    statuses.value = []
    sources.value = []
    streamError.value = ""
    controller = new AbortController()

    const res = await fetch(api.chatUrl(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sid,
        message: trimmed,
        rag_enabled: settingsStore.ragEnabled,
        visual_enabled: settingsStore.visualEnabled,
        kg_enabled: settingsStore.kgEnabled,
      }),
      signal: controller.signal,
    })

    try {
      await consumeSSE(res, {
        status: (d) => statuses.value.push({ kind: d.kind as StatusKind, text: String(d.text ?? "") }),
        source: (d) => sources.value.push(d as unknown as Source),
        token: (d) => {
          answerText.value = String(d.text ?? "")
        },
        error: (d) => {
          streamError.value = String(d.message ?? "生成出错")
        },
        done: (d) => {
          const arr = messagesBySession.value[sid] ?? []
          arr.push({
            id: Number(d.message_id),
            session_id: sid,
            role: "assistant",
            content: String(d.answer ?? ""),
            sources: (d.sources as Source[]) ?? [],
            created_at: Date.now(),
          })
          messagesBySession.value[sid] = [...arr]
          answerText.value = ""
          sources.value = []
        },
      })
    } catch (e) {
      // 中止：保留已流出的部分
      if ((e as Error)?.name !== "AbortError") {
        streamError.value = String((e as Error)?.message ?? e)
      }
      if (answerText.value) {
        const arr = messagesBySession.value[sid] ?? []
        arr.push({
          id: -Date.now(),
          session_id: sid,
          role: "assistant",
          content: answerText.value,
          sources: sources.value,
          created_at: Date.now(),
        })
        messagesBySession.value[sid] = [...arr]
        answerText.value = ""
        sources.value = []
      }
    } finally {
      streaming.value = false
      controller = null
    }
  }

  function clearLocal(sid: string) {
    messagesBySession.value[sid] = []
  }

  return {
    messagesBySession,
    streaming,
    answerText,
    statuses,
    sources,
    streamError,
    localMessages,
    loadMessages,
    sendMessage,
    abort,
    clearLocal,
  }
})
