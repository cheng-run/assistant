<script setup lang="ts">
import { ref } from "vue"
import { useChatStore } from "../../stores/chat"

const chat = useChatStore()
const text = ref("")

function onKeydown(e: KeyboardEvent) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault()
    send()
  }
}

function send() {
  const t = text.value.trim()
  if (!t || chat.streaming) return
  text.value = ""
  void chat.sendMessage(t)
}

function autoResize(e: Event) {
  const el = e.target as HTMLTextAreaElement
  el.style.height = "auto"
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`
}
</script>

<template>
  <div class="input-wrap">
    <div class="chat-input" :class="{ disabled: chat.streaming }">
      <textarea
        v-model="text"
        rows="1"
        :disabled="chat.streaming"
        placeholder="输入问题，Enter 发送 · Shift+Enter 换行"
        @keydown="onKeydown"
        @input="autoResize"
      ></textarea>
      <button v-if="!chat.streaming" class="send-btn" :disabled="!text.trim()" @click="send">➤</button>
      <button v-else class="stop-btn" title="停止" @click="chat.abort()">■</button>
    </div>
    <div v-if="chat.streamError" class="input-error">⚠ {{ chat.streamError }}</div>
    <div class="input-hint">基于 LangGraph / Deep Agents 的多轮文档问答</div>
  </div>
</template>

<style scoped>
.input-wrap {
  padding: 12px 28px 20px;
}
.chat-input {
  max-width: 760px;
  margin: 0 auto;
  display: flex;
  align-items: flex-end;
  gap: 10px;
  background: var(--surface);
  border: 1px solid var(--border-strong);
  border-radius: 14px;
  padding: 10px 12px;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}
.chat-input:focus-within {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px var(--accent-soft);
}
.chat-input.disabled { opacity: 0.7; }
textarea {
  flex: 1;
  border: none;
  outline: none;
  background: transparent;
  color: var(--text-1);
  font-family: var(--sans);
  font-size: 14px;
  line-height: 1.5;
  resize: none;
  max-height: 180px;
}
textarea::placeholder { color: var(--text-3); }
.send-btn {
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  border: none;
  border-radius: 10px;
  background: var(--accent);
  color: #fff;
  font-size: 15px;
  cursor: pointer;
  transition: all 0.15s ease;
  display: grid;
  place-items: center;
}
.send-btn:hover:not(:disabled) { background: var(--accent-hover); }
.send-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.stop-btn {
  flex-shrink: 0;
  width: 34px;
  height: 34px;
  border: none;
  border-radius: 10px;
  background: var(--surface-3);
  color: var(--text-1);
  font-size: 14px;
  cursor: pointer;
}
.stop-btn:hover { background: #ef4444; color: #fff; }
.input-error {
  max-width: 760px;
  margin: 8px auto 0;
  color: #ef4444;
  font-size: 12.5px;
}
.input-hint {
  max-width: 760px;
  margin: 10px auto 0;
  text-align: center;
  font-size: 11px;
  color: var(--text-3);
  letter-spacing: 0.02em;
}
</style>
