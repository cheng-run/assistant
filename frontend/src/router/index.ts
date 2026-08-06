import { createRouter, createWebHistory } from "vue-router"

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", component: () => import("../components/chat/ChatWindow.vue") },
    { path: "/s/:id", component: () => import("../components/chat/ChatWindow.vue") },
  ],
})

export { router }
