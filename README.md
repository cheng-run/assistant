# 智能文档问答助手

基于 **FastAPI + Vue 3** 的多轮智能文档问答系统。上传 PDF / DOCX / MD / TXT 文档后，Agent 自动检索、调用工具并标注来源回答。

- **检索引擎**（手写）：docling 解析 → 标题/Token 分块 → Ollama bge-m3 本地向量化 → SQLite 检索 → 知识图谱 / 视觉融合
- **Agent 编排**：LangChain 工具包装 + LangGraph ReAct 运行时 + Deep Agents 规划/子代理（可选）
- **前端**：Vue 3 + TypeScript + Naive UI，多会话侧栏、流式 markdown、来源卡片、暗色主题

## 快速开始

**前置条件**：
1. 启动本地 Ollama：`ollama serve`，并确保已拉取 `bge-m3`（嵌入）与 `qwen3-vl:4b`（视觉，可选）
2. 配置 `.env`（含 `DEEPSEEK_API_KEY`）

**一键启动**（首次会自动构建前端并打开浏览器）：

```bash
uv run --env-file .env python launch.py
```

浏览器访问 http://127.0.0.1:8000

> ⚠️ 必须用 `uv run --env-file .env` 启动 —— 它会在 Python 解释器启动前注入 `PYTHONUTF8=1`，否则 Windows 中文系统下 docling 会 GBK 崩溃。

## 开发模式（前端热更新）

开两个终端：

```bash
# 终端 1：后端
uv run --env-file .env uvicorn server.main:app --host 127.0.0.1 --port 8000 --reload

# 终端 2：前端 Vite（/api 自动代理到后端）
cd frontend && npm run dev
```

浏览器访问 http://127.0.0.1:5173

## 切换 Agent 模式

`.env` 中设置（改后重启）：

| 变量 | 值 | 说明 |
|---|---|---|
| `AGENT_MODE` | `reactive`（默认）/ `deep` / `legacy` | LangGraph / Deep Agents 规划子代理 / 旧直答 |
| `ENABLE_CHECKPOINT` | `1` | 启用 langgraph 多轮记忆持久化 |

## 项目结构

```
server/    FastAPI 后端（SSE 流式、会话管理、上传索引）
agent/     LangChain/LangGraph/Deep Agents 编排层
rag/       RAG 检索引擎（手写：解析/分块/向量化/图谱/视觉）
memory/    双层认知记忆系统
frontend/  Vue 3 + TS + Naive UI 前端
tests/     金集回归（run_golden.py）
```

## 测试

```bash
uv run --env-file .env python -m tests.run_golden --retrieve-only      # 只测检索
uv run --env-file .env python -m tests.run_golden --paths legacy,reactive  # 对比旧/新回答
```

详细设计见 [PROJECT_README.md](PROJECT_README.md)。
