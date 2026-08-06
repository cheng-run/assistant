export interface Session {
  id: string
  title: string
  created_at: number
  updated_at: number
}

export interface Source {
  id?: number | null
  source_file: string
  heading_path: string[]
  score?: number | null
  content: string
  modality: string
}

export interface Message {
  id: number
  session_id: string
  role: "user" | "assistant"
  content: string
  sources?: Source[] | null
  created_at: number
}

export type StatusKind = "tool" | "plan" | "subagent"

export interface StatusEvent {
  kind: StatusKind
  text: string
}

export type ChatEvent =
  | { type: "status"; kind: StatusKind; text: string }
  | { type: "source"; source_file: string; heading_path: string[]; score?: number | null; content: string; modality: string }
  | { type: "token"; delta: string; text: string }
  | { type: "done"; answer: string; sources: Source[]; message_id: number }
  | { type: "error"; message: string }

export interface UploadEvent {
  stage: string
  pct: number
  label: string
  done?: number
  total?: number
}

export interface UploadResult {
  source_file: string
  chunks: number
  kg: string
  visual: boolean
  sources: string[]
}

export interface Config {
  agent_mode: string
  checkpoint_enabled: boolean
  model: string
  use_langchain: boolean
}
