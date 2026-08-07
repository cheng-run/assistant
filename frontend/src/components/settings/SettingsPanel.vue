<script setup lang="ts">
import { NButton, NDrawer, NDrawerContent, NSwitch, useDialog, useMessage } from "naive-ui"
import { useUiStore } from "../../stores/ui"
import { useSettingsStore } from "../../stores/settings"
import { useTheme } from "../../utils/theme"

const ui = useUiStore()
const settings = useSettingsStore()
const { isDark, toggleTheme } = useTheme()
const dialog = useDialog()
const message = useMessage()

async function clearDocs() {
  dialog.warning({
    title: "清空全部文档索引",
    content: "将删除所有已向量化的文档片段与知识图谱，此操作不可撤销。",
    positiveText: "清空",
    negativeText: "取消",
    onPositiveClick: async () => {
      try {
        await fetch("/api/docs", { method: "DELETE" })
        message.success("已清空文档索引")
      } catch (e) {
        message.error(`清空失败: ${String(e)}`)
      }
    },
  })
}
</script>

<template>
  <n-drawer v-model:show="ui.settingsOpen" :width="320" placement="right">
    <n-drawer-content title="设置">
      <div class="setting-group">
        <div class="group-title">检索开关</div>
        <div class="setting-row">
          <div class="setting-label">文档问答<span class="hint">基于文档内容回答（视觉/图谱自动路由）</span></div>
          <n-switch :value="settings.ragEnabled" @update:value="settings.setRag" />
        </div>
      </div>

      <div class="setting-group">
        <div class="group-title">运行时</div>
        <div class="setting-row static">
          <div class="setting-label">Agent 模式</div>
          <span class="static-value">{{ settings.agentMode }}</span>
        </div>
        <div class="setting-row static">
          <div class="setting-label">模型</div>
          <span class="static-value mono">{{ settings.model }}</span>
        </div>
        <div class="setting-row static">
          <div class="setting-label">多轮记忆（checkpoint）</div>
          <span class="static-value">{{ settings.checkpointEnabled ? "开" : "关" }}</span>
        </div>
      </div>

      <div class="setting-group">
        <div class="group-title">外观</div>
        <div class="setting-row">
          <div class="setting-label">深色模式</div>
          <n-switch :value="isDark" @update:value="toggleTheme" />
        </div>
      </div>

      <div class="setting-group danger">
        <n-button block type="error" secondary @click="clearDocs">清空全部文档索引</n-button>
      </div>
    </n-drawer-content>
  </n-drawer>
</template>

<style scoped>
.setting-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding-bottom: 18px;
  margin-bottom: 18px;
  border-bottom: 1px solid var(--border);
}
.setting-group:last-child { border-bottom: none; }
.group-title {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-3);
  letter-spacing: 0.04em;
  margin-bottom: 8px;
}
.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 0;
}
.setting-label {
  font-size: 13.5px;
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.hint { font-size: 11px; color: var(--text-3); }
.setting-row.static { padding: 6px 0; }
.static-value {
  font-size: 12px;
  color: var(--text-2);
  background: var(--surface-2);
  padding: 2px 9px;
  border-radius: 12px;
}
.mono { font-family: var(--mono); }
.danger { padding-bottom: 0; border-bottom: none; }
</style>
