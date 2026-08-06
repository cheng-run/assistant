<script setup lang="ts">
import { onMounted, ref } from "vue"
import { useRouter } from "vue-router"
import { useSessionStore } from "../../stores/session"
import { useSettingsStore } from "../../stores/settings"
import { useTheme } from "../../utils/theme"
import { useUiStore } from "../../stores/ui"
import UploadDropzone from "../upload/UploadDropzone.vue"
import UploadProgress from "../upload/UploadProgress.vue"
import { useUploadStore } from "../../stores/upload"

const session = useSessionStore()
const settings = useSettingsStore()
const ui = useUiStore()
const upload = useUploadStore()
const { isDark, toggleTheme } = useTheme()
const router = useRouter()

const sources = ref<string[]>([])

async function refreshSources() {
  try {
    const res = await fetch("/api/docs/sources")
    const data = await res.json()
    sources.value = data.sources ?? []
  } catch {
    /* ignore */
  }
}

async function newSession() {
  const s = await session.createSession()
  router.push(`/s/${s.id}`)
}

function goSession(id: string) {
  session.setCurrent(id)
  router.push(`/s/${id}`)
}

async function removeSource(name: string) {
  await fetch(`/api/docs/${encodeURIComponent(name)}`, { method: "DELETE" })
  refreshSources()
}

const modeLabel: Record<string, string> = {
  reactive: "LangGraph Agent",
  deep: "Deep Agents",
  legacy: "Legacy",
}

onMounted(() => {
  session.fetchSessions()
  refreshSources()
})
</script>

<template>
  <aside class="sidebar">
    <div class="brand">
      <div class="brand-mark">文</div>
      <div class="brand-text">
        <div class="brand-title">智能文档问答</div>
        <div class="brand-sub">FastAPI · LangGraph · Deep Agents</div>
      </div>
    </div>

    <button class="new-btn" @click="newSession">
      <span class="plus">＋</span> 新对话
    </button>

    <nav class="session-list">
      <button
        v-for="s in session.sessions"
        :key="s.id"
        class="session-item"
        :class="{ active: s.id === session.currentId }"
        @click="goSession(s.id)"
      >
        <span class="s-title">{{ s.title }}</span>
        <span class="s-del" @click.stop="session.deleteSession(s.id)">✕</span>
      </button>
      <div v-if="!session.sessions.length" class="empty-hint">暂无会话</div>
    </nav>

    <div class="side-divider"></div>

    <section class="docs">
      <div class="section-title">
        <span>文档库</span>
        <span v-if="sources.length" class="count">{{ sources.length }}</span>
      </div>
      <UploadDropzone @done="refreshSources" />
      <UploadProgress v-if="upload.isUploading || upload.lastResult" />
      <ul class="source-list">
        <li v-for="name in sources" :key="name" class="source-item">
          <span class="si-file">📄 {{ name }}</span>
          <span class="si-del" @click="removeSource(name)">✕</span>
        </li>
      </ul>
    </section>

    <footer class="side-foot">
      <span class="mode-badge" :title="`AGENT_MODE=${settings.agentMode}`">
        {{ modeLabel[settings.agentMode] ?? settings.agentMode }}
      </span>
      <div class="foot-actions">
        <button class="icon-btn" :title="isDark ? '切换浅色' : '切换深色'" @click="toggleTheme">
          {{ isDark ? "☀" : "☾" }}
        </button>
        <button class="icon-btn" title="设置" @click="ui.settingsOpen = true">⚙</button>
      </div>
    </footer>
  </aside>
</template>

<style scoped>
.sidebar {
  width: 292px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  background: var(--bg-soft);
  border-right: 1px solid var(--border);
  padding: 18px 14px;
  gap: 12px;
  overflow: hidden;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 2px 4px 6px;
}
.brand-mark {
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--accent), #1d4ed8);
  color: #fff;
  font-family: var(--serif);
  font-size: 19px;
  font-weight: 600;
  display: grid;
  place-items: center;
  box-shadow: 0 4px 14px var(--accent-soft);
}
.brand-title {
  font-family: var(--serif);
  font-size: 16.5px;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.brand-sub {
  font-size: 10.5px;
  color: var(--text-3);
  margin-top: 1px;
  letter-spacing: 0.02em;
}

.new-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  padding: 9px 0;
  border-radius: var(--radius);
  border: 1px dashed var(--border-strong);
  background: var(--surface);
  color: var(--text-2);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.18s ease;
}
.new-btn:hover {
  border-color: var(--accent);
  color: var(--accent-hover);
  background: var(--accent-soft);
}
.plus { font-size: 14px; }

.session-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin: 0 -4px;
  padding: 0 4px;
}
.session-item {
  display: flex;
  align-items: center;
  gap: 6px;
  width: 100%;
  padding: 8px 10px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-2);
  font-size: 13px;
  text-align: left;
  cursor: pointer;
  transition: background 0.15s ease;
}
.session-item:hover { background: var(--surface-2); }
.session-item.active { background: var(--accent-soft); color: var(--text-1); }
.s-title {
  flex: 1;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.s-del {
  opacity: 0;
  color: var(--text-3);
  font-size: 11px;
  padding: 2px 4px;
  transition: opacity 0.15s ease;
}
.session-item:hover .s-del { opacity: 1; }
.s-del:hover { color: #ef4444; }
.empty-hint { color: var(--text-3); font-size: 12px; padding: 8px; text-align: center; }

.side-divider { height: 1px; background: var(--border); margin: 2px 0; }

.docs { display: flex; flex-direction: column; gap: 8px; }
.section-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  font-weight: 600;
  color: var(--text-3);
  letter-spacing: 0.04em;
  padding: 0 2px;
}
.count {
  background: var(--surface-3);
  color: var(--text-2);
  border-radius: 20px;
  font-size: 10.5px;
  padding: 1px 8px;
}
.source-list {
  list-style: none;
  margin: 0;
  padding: 0;
  max-height: 120px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.source-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 12px;
  color: var(--text-2);
  padding: 5px 8px;
  border-radius: var(--radius-sm);
}
.source-item:hover { background: var(--surface-2); }
.si-file { white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.si-del { opacity: 0; color: var(--text-3); font-size: 11px; cursor: pointer; padding: 0 3px; }
.source-item:hover .si-del { opacity: 1; }
.si-del:hover { color: #ef4444; }

.side-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding-top: 10px;
  border-top: 1px solid var(--border);
}
.mode-badge {
  font-size: 10.5px;
  color: var(--text-2);
  background: var(--surface-3);
  padding: 3px 9px;
  border-radius: 20px;
  letter-spacing: 0.03em;
}
.foot-actions { display: flex; gap: 4px; }
.icon-btn {
  width: 30px;
  height: 30px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-2);
  font-size: 15px;
  cursor: pointer;
  transition: background 0.15s ease;
}
.icon-btn:hover { background: var(--surface-2); color: var(--text-1); }
</style>
