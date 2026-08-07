<script setup lang="ts">
import { computed, onMounted, ref, watch } from "vue"
import { useRoute, useRouter } from "vue-router"
import { useSessionStore } from "../../stores/session"
import { useChatStore } from "../../stores/chat"
import { useSettingsStore } from "../../stores/settings"
import { useUiStore } from "../../stores/ui"
import MessageList from "./MessageList.vue"
import ChatInput from "./ChatInput.vue"
import EmptyState from "../common/EmptyState.vue"

const route = useRoute()
const router = useRouter()
const session = useSessionStore()
const chat = useChatStore()
const settings = useSettingsStore()
const ui = useUiStore()

const currentId = computed(() => (route.params.id as string) || session.currentId)
const hasSession = computed(() => !!currentId.value)
const currentTitle = computed(() => session.sessions.find((s) => s.id === currentId.value)?.title ?? "")

// 视觉/图谱检索由后端路由引擎自动决策（按问题类型组合），前端只留"文档问答"总开关
const toggles = ref([
  { key: "rag", label: "文档问答", get: () => settings.ragEnabled, set: settings.setRag },
])

async function ensureSession() {
  if (session.sessions.length === 0) await session.fetchSessions()
  if (!currentId.value) {
    // 无有效会话 → 自动新建（用户可直接开始聊天）
    if (session.sessions.length > 0) {
      const first = session.sessions[0].id
      session.setCurrent(first)
      router.replace(`/s/${first}`)
      return
    }
    const s = await session.createSession()
    router.replace(`/s/${s.id}`)
    return
  }
  session.setCurrent(currentId.value)
  try {
    await chat.loadMessages(currentId.value)
  } catch {
    // 陈旧 currentId（会话已被删除）→ 自动新建并切换到新会话
    const s = await session.createSession()
    router.replace(`/s/${s.id}`)
  }
}

onMounted(async () => {
  await settings.loadConfig()
  await ensureSession()
})

watch(currentId, async (id) => {
  if (id) await chat.loadMessages(id)
})
</script>

<template>
  <div class="chat-window">
    <header class="chat-top">
      <div class="top-left">
        <h1 class="chat-title">{{ currentTitle || "新对话" }}</h1>
      </div>
      <div class="top-right">
        <div class="toggles">
          <button
            v-for="t in toggles"
            :key="t.key"
            class="pill"
            :class="{ on: t.get() }"
            @click="t.set(!t.get())"
          >
            {{ t.label }}
          </button>
        </div>
        <button class="gear-btn" title="设置" @click="ui.settingsOpen = true">⚙</button>
      </div>
    </header>

    <EmptyState v-if="!hasSession" />
    <MessageList v-else :session-id="currentId ?? ''" />
    <ChatInput v-if="hasSession" />
  </div>
</template>

<style scoped>
.chat-window {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-height: 0;
}

.chat-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 28px 12px;
  border-bottom: 1px solid var(--border);
}
.chat-title {
  font-family: var(--serif);
  font-size: 17px;
  font-weight: 600;
  margin: 0;
  max-width: 420px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.top-right { display: flex; align-items: center; gap: 12px; }

.toggles { display: flex; gap: 6px; }
.pill {
  padding: 5px 14px;
  border-radius: 20px;
  border: 1px solid var(--border-strong);
  background: var(--surface);
  color: var(--text-3);
  font-size: 12px;
  cursor: pointer;
  transition: all 0.18s ease;
}
.pill:hover { color: var(--text-2); }
.pill.on {
  background: var(--accent-soft);
  border-color: var(--accent);
  color: var(--accent-hover);
}
.gear-btn {
  width: 32px;
  height: 32px;
  border: none;
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--text-2);
  font-size: 15px;
  cursor: pointer;
}
.gear-btn:hover { background: var(--surface-2); }
</style>
