<script setup lang="ts">
import { computed } from "vue"
import { NConfigProvider, NDialogProvider, NMessageProvider } from "naive-ui"
import Sidebar from "./components/layout/Sidebar.vue"
import SettingsPanel from "./components/settings/SettingsPanel.vue"
import {
  dateZhCN,
  lightThemeOverrides,
  naiveDarkTheme,
  themeOverrides,
  useTheme,
  zhCN,
} from "./utils/theme"

const { isDark } = useTheme()
const overrides = computed(() => (isDark.value ? themeOverrides : lightThemeOverrides))
</script>

<template>
  <n-config-provider
    :theme="isDark ? naiveDarkTheme : null"
    :theme-overrides="overrides"
    :locale="zhCN"
    :date-locale="dateZhCN"
  >
    <n-message-provider>
      <n-dialog-provider>
        <div class="app-shell">
          <Sidebar />
          <main class="app-main">
            <router-view />
          </main>
          <SettingsPanel />
        </div>
      </n-dialog-provider>
    </n-message-provider>
  </n-config-provider>
</template>

<style scoped>
.app-shell {
  display: flex;
  height: 100vh;
  overflow: hidden;
}
.app-main {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
}
</style>
