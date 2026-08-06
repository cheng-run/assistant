<script setup lang="ts">
import { useRouter } from "vue-router"
import { useSessionStore } from "../../stores/session"

const session = useSessionStore()
const router = useRouter()

async function start() {
  const s = await session.createSession()
  router.push(`/s/${s.id}`)
}
</script>

<template>
  <div class="empty">
    <div class="empty-glyph">文</div>
    <h2 class="empty-title">开始你的文档问答</h2>
    <p class="empty-desc">
      在左侧上传文档，或直接开启对话。
      <br />Agent 会自动检索文档、调用工具并标注来源。
    </p>
    <button class="empty-btn" @click="start">＋ 新建对话</button>
  </div>
</template>

<style scoped>
.empty {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding-bottom: 80px;
  animation: fade-in 0.4s ease both;
}
.empty-glyph {
  width: 58px;
  height: 58px;
  border-radius: 16px;
  background: linear-gradient(135deg, var(--accent), #1d4ed8);
  color: #fff;
  font-family: var(--serif);
  font-size: 30px;
  display: grid;
  place-items: center;
  box-shadow: 0 8px 28px var(--accent-soft);
  margin-bottom: 10px;
}
.empty-title {
  font-family: var(--serif);
  font-size: 21px;
  font-weight: 600;
  margin: 0;
}
.empty-desc {
  color: var(--text-3);
  font-size: 13.5px;
  text-align: center;
  margin: 4px 0 18px;
  line-height: 1.7;
}
.empty-btn {
  padding: 9px 26px;
  border: none;
  border-radius: var(--radius);
  background: var(--accent);
  color: #fff;
  font-size: 14px;
  cursor: pointer;
  transition: background 0.15s ease;
}
.empty-btn:hover { background: var(--accent-hover); }
</style>
