<script setup lang="ts">
import { ref } from "vue"
import type { Message } from "../../types"
import StreamingMarkdown from "./StreamingMarkdown.vue"
import SourceCard from "./SourceCard.vue"

defineProps<{ message: Message }>()
const showSources = ref(true)
</script>

<template>
  <div class="msg-row" :class="message.role">
    <div class="msg-meta">
      <span class="msg-role">{{ message.role === "user" ? "你" : "助手" }}</span>
    </div>
    <div class="msg-body">
      <StreamingMarkdown v-if="message.role === 'assistant'" :text="message.content" />
      <div v-else class="user-text">{{ message.content }}</div>

      <div v-if="message.role === 'assistant' && message.sources?.length" class="sources-block">
        <button class="sources-toggle" @click="showSources = !showSources">
          来源 · {{ message.sources.length }} 条 {{ showSources ? "▾" : "▸" }}
        </button>
        <div v-if="showSources" class="sources anim-fade-in">
          <SourceCard v-for="s in message.sources" :key="s.id ?? s.content" :source="s" />
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.msg-row {
  display: flex;
  gap: 16px;
  animation: fade-up 0.3s cubic-bezier(0.16, 1, 0.3, 1) both;
}

.msg-meta {
  flex-shrink: 0;
  width: 42px;
  padding-top: 2px;
}
.msg-role {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  color: var(--text-3);
  text-transform: uppercase;
}
.msg-row.assistant .msg-role { color: var(--accent); }

.msg-body {
  flex: 1;
  min-width: 0;
}
.user-text {
  font-size: 14.5px;
  color: var(--text-1);
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 16px;
  display: inline-block;
  max-width: 100%;
}

.sources-block { margin-top: 10px; }
.sources-toggle {
  border: none;
  background: none;
  color: var(--text-3);
  font-size: 12px;
  cursor: pointer;
  padding: 2px 0;
  transition: color 0.15s ease;
}
.sources-toggle:hover { color: var(--accent-hover); }
.sources {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px;
  margin-top: 8px;
}
</style>
