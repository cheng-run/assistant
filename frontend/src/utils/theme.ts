import { ref } from "vue"
import { darkTheme, dateZhCN, zhCN, type GlobalThemeOverrides } from "naive-ui"

const isDark = ref(localStorage.getItem("smartqa-theme") !== "light")

function applyClass() {
  document.documentElement.classList.toggle("light", !isDark.value)
}

applyClass()

export function toggleTheme() {
  isDark.value = !isDark.value
  localStorage.setItem("smartqa-theme", isDark.value ? "dark" : "light")
  applyClass()
}

export function useTheme() {
  return { isDark, toggleTheme }
}

const darkOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: "#3b82f6",
    primaryColorHover: "#60a5fa",
    primaryColorPressed: "#2563eb",
    primaryColorSuppl: "#3b82f6",
    borderRadius: "10px",
    borderRadiusSmall: "7px",
    bodyColor: "#0b0d12",
    cardColor: "#12151c",
    modalColor: "#12151c",
    popoverColor: "#151923",
    inputColor: "#171b24",
    textColorBase: "#e8eaed",
    textColor1: "#e8eaed",
    textColor2: "#b6bcc7",
    textColor3: "#8b93a1",
    borderColor: "rgba(255,255,255,0.08)",
    dividerColor: "rgba(255,255,255,0.06)",
    fontFamily: "'Manrope', 'PingFang SC', 'Microsoft YaHei', system-ui, sans-serif",
    fontFamilyMono: "'JetBrains Mono', 'Consolas', monospace",
  },
  Button: {
    borderRadiusMedium: "10px",
  },
}

const lightOverrides: GlobalThemeOverrides = {
  common: {
    primaryColor: "#2563eb",
    primaryColorHover: "#1d4ed8",
    primaryColorPressed: "#1e40af",
    primaryColorSuppl: "#2563eb",
    borderRadius: "10px",
    borderRadiusSmall: "7px",
    bodyColor: "#f6f7f9",
    cardColor: "#ffffff",
    modalColor: "#ffffff",
    popoverColor: "#ffffff",
    inputColor: "#f0f2f5",
    textColorBase: "#1a202c",
    textColor1: "#1a202c",
    textColor2: "#4a5568",
    textColor3: "#8a94a6",
    borderColor: "rgba(15,23,42,0.08)",
    dividerColor: "rgba(15,23,42,0.06)",
  },
}

export const themeOverrides: GlobalThemeOverrides = darkOverrides
export const lightThemeOverrides: GlobalThemeOverrides = lightOverrides
export const naiveDarkTheme = darkTheme
export { zhCN, dateZhCN }
