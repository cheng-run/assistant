import { request } from "./client"
import type { Config, Message, Session } from "../types"

export const api = {
  sessions: {
    list: () => request<Session[]>("/api/sessions"),
    create: () =>
      request<Session>("/api/sessions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      }),
    rename: (id: string, title: string) =>
      request<Session>(`/api/sessions/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title }),
      }),
    remove: (id: string) => request<null>(`/api/sessions/${id}`, { method: "DELETE" }),
    messages: (id: string) => request<Message[]>(`/api/sessions/${id}/messages`),
  },
  config: () => request<Config>("/api/config"),
  docs: {
    sources: () => request<{ sources: string[]; outline: string }>("/api/docs/sources"),
    clear: () => request<{ cleared: boolean }>("/api/docs", { method: "DELETE" }),
    remove: (source: string) =>
      request<{ removed: boolean }>(`/api/docs/${encodeURIComponent(source)}`, { method: "DELETE" }),
  },
  chatUrl: () => "/api/chat",
  uploadUrl: () => "/api/upload",
}
