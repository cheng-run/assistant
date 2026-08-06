<script setup lang="ts">
import { computed, nextTick, ref, watch } from "vue"
import { useChatStore } from "../../stores/chat"
import { renderMarkdown } from "../../utils/markdown"
import MessageItem from "./MessageItem.vue"
import StatusLine from "./StatusLine.vue"
import SourceCard from "./SourceCard.vue"

const props = defineProps<{ sessionId: string }>()
const chat = useChatStore()

const messages = computed(() => chat.localMessages(props.sessionId))
const listEl = ref<HTMLElement>()

function scrollToBottom() {
  nextTick(() => {
    listEl.value?.scrollTo({ top: listEl.value.scrollHeight })
  })
}

watch(
  () => [chat.answerText, chat.statuses.length, chat.messagesBySession[props.sessionId]?.length],
  () => scrollToBottom(),
  { deep: true },
)
</script>

<template>
  <div ref="listEl" class="msg-list">
    <div class="msg-column">
      <template v-for="m in messages" :key="m.id">
        <MessageItem :message="m" />
      </template>

      <!-- 流式中的 assistant 气泡 -->
      <div v-if="chat.streaming" class="anim-fade-up">
        <div v-if="chat.statuses.length" class="status-block">
          <StatusLine v-for="(s, i) in chat.statuses" :key="i" :status="s" />
        </div>
        <div class="msg-row assistant">
          <div class="msg-body">
            <div
              v-if="chat.answerText"
              class="md-body stream-caret"
              v-html="renderMarkdown(chat.answerText)"
            ></div>
            <div v-else class="thinking">
              <span class="thinking-dot"></span><span class="thinking-dot"></span><span class="thinking-dot"></span>
            </div>
            <div v-if="chat.sources.length" class="sources-block">
              <div class="sources anim-fade-in">
                <SourceCard v-for="(s, i) in chat.sources" :key="i" :source="s" />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div v-if="chat.streamError && !chat.answerText" class="stream-error">⚠ {{ chat.streamError }}</div>
    </div>
  </div>
</template>

<style scoped>
.msg-list {
  flex: 1;
  overflow-y: auto;
  padding: 28px 28px 12px;
}
.msg-column {
  max-width: 760px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: var(--message-gap);
}

.status-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 10px;
}

.thinking {
  display: flex;
  gap: 5px;
  padding: 4px 0;
}
.thinking-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-3);
  animation: pulse 1.2s ease-in-out infinite;
}
.thinking-dot:nth-child(2) { animation-delay: 0.15s; }
.thinking-dot:nth-child(3) { animation-delay: 0.3s; }
@keyframes pulse {
  0%, 100% { opacity: 0.25; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1); }
}

.sources {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px;
  margin-top: 12px;
}
.stream-error {
  color: #ef4444;
  font-size: 13px;
  text-align: center;
  padding: 8px;
}
</style>
