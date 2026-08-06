"""agent/service.py — 编排层门面（FastAPI 后端结构化事件流 + Gradio 兼容层）

职责：
  1. 确定性种子检索（_seed_retrieve）：把重构前 app.py:on_message 的上下文拼装
     逻辑搬迁至此，保证即使 agent 工具调用失败，文档上下文也已注入。
  2. 结构化事件流（stream_events）：yield `{type: status|source|token|error}` 事件，
     供 FastAPI 的 SSE 端点消费；stream_chat 是把它重渲染回 Gradio 行的薄适配器
     （保证 tests/run_golden.py 零改动）。

开关约定：
  - ENABLE_CHECKPOINT=1：启用 langgraph-checkpoint-sqlite 持久化多轮记忆，
    以 session_id 作为 thread_id（历史由 checkpoint 重建，输入只传新 user 消息）；
  - AGENT_MODE=deep：走 Deep Agents harness（规划/子代理），否则走 LangGraph agent。
"""

import asyncio
import os
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from agent.tools import mem, rag
from memory.base import extract_text

MAX_HISTORY_TURNS = 20

# 依据开关组合缓存 agent（checkpoint 关闭时复用；开启时每次请求重建以绑定 saver）
_AGENT_CACHE: Dict[tuple, object] = {}

_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_db")


# ═══════════════════════════════════════════════════════════════
#  确定性种子检索
# ═══════════════════════════════════════════════════════════════

def _seed_retrieve(
    message: str,
    rag_enabled: bool,
    visual_enabled: bool,
    kg_enabled: bool,
) -> Tuple[List[Dict], List[Dict]]:
    """构建"确定性种子"系统消息 + 结构化来源片段。

    返回 (sys_msgs, sources)：
      - sys_msgs: 送给 agent 的系统消息（长期记忆 + 大纲 + 文档片段/图谱）
      - sources:  检索片段的结构化表示，供前端"来源卡片"事件
    """
    sys_msgs: List[Dict] = []
    sources: List[Dict] = []

    # 长期记忆注入（让沉睡的 memory/ 子系统在 UI 上生效）
    try:
        memory_ctx = mem.get_relevant_context(message, limit=3)
        if memory_ctx:
            lines = "\n".join(f"- {extract_text(c['content'])}" for c in memory_ctx)
            sys_msgs.append({
                "role": "system",
                "content": f"【系统资料】以下是系统检索到的与用户相关的历史记忆（非本次用户输入）：\n{lines}",
            })
    except Exception:
        pass

    if not rag_enabled:
        return sys_msgs, sources

    # 语料为空提示：避免 agent 反复调用空的检索工具
    try:
        if not rag.vector_store.list_sources():
            sys_msgs.append({
                "role": "system",
                "content": (
                    "（当前没有任何已索引文档。请如实告知用户尚未上传文档并引导其先上传，"
                    "不要反复调用检索工具，直接给出简短的引导回答。）"
                ),
            })
            return sys_msgs, sources
    except Exception:
        pass

    # 文档大纲注入（全局结构认知）
    try:
        outline_text = rag.get_all_outlines()
        if outline_text:
            sys_msgs.append({
                "role": "system",
                "content": f"【系统资料】以下为已索引文档的大纲结构（系统注入，非用户输入）：\n{outline_text}",
            })
    except Exception:
        pass

    # RAG 上下文注入
    try:
        if kg_enabled:
            kg_context, chunks = rag.retrieve_with_kg(
                message, top_k=3, enable_visual=visual_enabled
            )
            if kg_context:
                sys_msgs.append({"role": "system", "content": kg_context})
        else:
            chunks = rag.retrieve(message, top_k=3, enable_visual=visual_enabled)
    except Exception:
        sys_msgs.append({
            "role": "system",
            "content": "（注意：文档检索暂时失败，以下回答基于通用知识。）",
        })
        chunks = []

    # 结构化来源片段（供前端 source 事件）
    for c in chunks:
        sources.append({
            "id": c.get("id"),
            "source_file": c.get("source_file", ""),
            "heading_path": c.get("heading_path", []),
            "score": c.get("score"),
            "content": (c.get("content") or "")[:300],
            "modality": c.get("modality", "text"),
        })

    if chunks:
        parts = []
        for c in chunks:
            heading = " > ".join(c.get("heading_path", [])) or "文档"
            parts.append(f"[{heading}]\n{c['content']}")
        context = "\n\n---\n\n".join(parts)
        sys_msgs.append({
            "role": "system",
            "content": (
                "【系统参考资料，非用户输入】以下是从已索引文档中检索到的相关片段，"
                "仅作为回答依据。请直接回答用户最后提出的问题，如内容不足请如实说明：\n\n"
                + context
            ),
        })

    return sys_msgs, sources


def _build_seed_system_messages(
    message: str,
    rag_enabled: bool,
    visual_enabled: bool,
    kg_enabled: bool,
) -> List[Dict]:
    """兼容层：只返回系统消息（旧调用面 / 测试仍可用）。"""
    sys_msgs, _ = _seed_retrieve(message, rag_enabled, visual_enabled, kg_enabled)
    return sys_msgs


# ═══════════════════════════════════════════════════════════════
#  agent 构建
# ═══════════════════════════════════════════════════════════════

def _memo_key(rag_enabled: bool, visual_enabled: bool, kg_enabled: bool) -> tuple:
    return (bool(rag_enabled), bool(visual_enabled), bool(kg_enabled))


def _get_agent(rag_enabled: bool, visual_enabled: bool, kg_enabled: bool):
    """获取（缓存）按开关组合构建的 LangGraph agent。"""
    from agent.graph import build_agent

    key = _memo_key(rag_enabled, visual_enabled, kg_enabled)
    if key not in _AGENT_CACHE:
        _AGENT_CACHE[key] = build_agent(
            include_rag=rag_enabled,
            include_kg=kg_enabled,
            include_visual=visual_enabled,
        )
    return _AGENT_CACHE[key]


async def _build_deep_agent(rag_enabled: bool, visual_enabled: bool, kg_enabled: bool):
    """构建 Deep Agents harness（AGENT_MODE=deep 时使用）。"""
    from agent.deep import build_deep_agent

    return build_deep_agent(
        include_rag=rag_enabled,
        include_kg=kg_enabled,
        include_visual=visual_enabled,
    )


# ═══════════════════════════════════════════════════════════════
#  状态渲染
# ═══════════════════════════════════════════════════════════════

def _render_node_status(node: str, update: dict) -> Optional[str]:
    """把 agent 节点更新渲染为中文状态行（流式 UI 的中间行）。"""
    msgs = update.get("messages") or []
    if "tool" in node:  # create_agent 的 tools 节点 / prebuilt 的 tools 节点
        names = [getattr(m, "name", None) for m in msgs if hasattr(m, "name")]
        names = [n for n in names if n]
        return ("🔧 正在检索: " + ", ".join(names)) if names else "🔧 正在执行工具调用"
    if node in ("model", "agent"):  # create_agent 用 model，prebuilt 用 agent
        for m in msgs:
            tool_calls = getattr(m, "tool_calls", None) or []
            if tool_calls:
                calls = ", ".join(tc.get("name", "?") for tc in tool_calls)
                return f"🤔 计划调用: {calls}"
    return None


def _render_deep_update(update: dict) -> Optional[str]:
    """把 Deep Agents 的 stream DeltaChannel 渲染为状态行（规划/子代理）。"""
    stream = (update or {}).get("stream") or {}
    subs = stream.get("subagents") or []
    if subs:
        s = subs[-1]
        return f"👤 子代理 **{s.get('name', '?')}** 输出: {s.get('output', '')[:120]}…"
    actions = stream.get("actions") or []
    if actions:
        todo = actions[-1]
        return f"📋 计划: {todo.get('title', '')} —— {todo.get('status', '')}"
    return None


def _status_kind(node: str, update: dict, is_deep: bool) -> str:
    """给 status 事件打 kind 分类（tool / plan / subagent），供前端图标区分。"""
    if is_deep:
        return "subagent" if (update or {}).get("stream", {}).get("subagents") else "plan"
    return "tool" if "tool" in node else "plan"


# ═══════════════════════════════════════════════════════════════
#  结构化事件流（核心）
# ═══════════════════════════════════════════════════════════════

async def _stream_agent_events(
    agent,
    agent_input: Dict,
    config: Dict,
    is_deep: bool = False,
    fallback_messages: Optional[List[Dict]] = None,
) -> AsyncGenerator[Dict, None]:
    """agent 流式 → 结构化事件 dict：`{type: status|token|error}`。

    - 只累计**模型生成的 AI 文本**（过滤工具结果回显）；
    - 客户端断开时向上传播 CancelledError，交由路由清理；
    - 若 agent 只调工具没产出文本，用种子上下文做一次兜底综合。
    """
    answer, produced = "", False
    try:
        async for mode, chunk in agent.astream(
            agent_input, config=config, stream_mode=["updates", "messages"]
        ):
            if mode == "updates":
                for node, update in chunk.items():
                    if not isinstance(update, dict):
                        continue
                    status = _render_deep_update(update) if is_deep else _render_node_status(node, update)
                    if status:
                        yield {
                            "type": "status",
                            "kind": _status_kind(node, update, is_deep),
                            "text": status,
                        }
            elif mode == "messages":
                msg, _meta = chunk
                # 只累计 AI 消息文本（AIMessageChunk.type == "ai"）
                if getattr(msg, "type", "") != "ai":
                    continue
                content = getattr(msg, "content", None)
                if content:
                    produced = True
                    answer += content
                    yield {"type": "token", "delta": content, "text": answer}
    except asyncio.CancelledError:
        raise  # 客户端断开：上抛由路由清理
    except Exception as e:
        yield {"type": "error", "message": str(e)}

    # 兜底综合（同原逻辑）——注意用 is not None：空语料时 fallback_messages=[] 仍应兜底
    if not produced and fallback_messages is not None:
        last_msgs = agent_input.get("messages") or []
        last = last_msgs[-1] if last_msgs else {}
        last_user = last.get("content", "") if isinstance(last, dict) else ""
        ftext = ""
        try:
            for tok in chat_stream_lc(fallback_messages + [{"role": "user", "content": last_user}]):
                ftext += tok
                yield {"type": "token", "delta": tok, "text": ftext}
        except Exception:
            yield {"type": "error", "message": "回答生成失败，请重试。"}


async def stream_events(
    message: str,
    history: list,
    rag_enabled: bool,
    visual_enabled: bool = False,
    kg_enabled: bool = False,
    session_id: Optional[str] = None,
) -> AsyncGenerator[Dict, None]:
    """对外结构化事件流：source 卡片先到，随后 status/token/error。

    供 FastAPI `POST /api/chat` 的 SSE 生成器直接转发。
    """
    if not message:
        return

    user_row = {"role": "user", "content": message}
    seed, sources = _seed_retrieve(message, rag_enabled, visual_enabled, kg_enabled)

    # 来源卡片先于答案到达
    for s in sources:
        yield {"type": "source", **s}

    is_deep = os.getenv("AGENT_MODE", "reactive") == "deep"

    if _is_checkpoint_enabled():
        # 启用 checkpointer：历史由 checkpoint 重建，输入只传新 user 消息
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        db_path = os.path.join(_DB_DIR, "agent_checkpoints.db")
        config = {"configurable": {"thread_id": session_id or "default"}, "recursion_limit": 25}
        async with AsyncSqliteSaver.from_conn_string(db_path) as saver:
            if is_deep:
                from agent.deep import build_deep_agent

                agent = build_deep_agent(
                    include_rag=rag_enabled, include_kg=kg_enabled,
                    include_visual=visual_enabled, checkpointer=saver,
                )
            else:
                from agent.graph import build_agent

                agent = build_agent(
                    include_rag=rag_enabled, include_kg=kg_enabled,
                    include_visual=visual_enabled, checkpointer=saver,
                )
            agent_input = {"messages": seed + [user_row]}
            async for ev in _stream_agent_events(agent, agent_input, config, is_deep, fallback_messages=seed):
                yield ev
    else:
        # 默认：不启用 checkpointer，history 直接进 messages（与旧行为一致）
        history_msgs = history[-MAX_HISTORY_TURNS * 2:]
        agent_input = {"messages": seed + history_msgs + [user_row]}
        config = {"recursion_limit": 25}
        if is_deep:
            agent = await _build_deep_agent(rag_enabled, visual_enabled, kg_enabled)
        else:
            agent = _get_agent(rag_enabled, visual_enabled, kg_enabled)
        async for ev in _stream_agent_events(agent, agent_input, config, is_deep, fallback_messages=seed):
            yield ev


# ═══════════════════════════════════════════════════════════════
#  Gradio 兼容层（保证 tests/run_golden.py 零改动）
# ═══════════════════════════════════════════════════════════════

async def stream_chat(
    message: str,
    history: list,
    rag_enabled: bool,
    visual_enabled: bool = False,
    kg_enabled: bool = False,
) -> AsyncGenerator:
    """薄适配器：把结构化事件重渲染回 Gradio 的 (history + 新行) 形状。"""
    if not message:
        yield history
        return

    user_row = {"role": "user", "content": message}
    async for ev in stream_events(message, history, rag_enabled, visual_enabled, kg_enabled):
        if ev["type"] in ("status", "token"):
            yield history + [user_row, {"role": "assistant", "content": ev["text"]}]


def chat_stream_lc(messages: List[Dict]):
    """纯 ChatOpenAI 流式（兜底综合 / legacy + USE_LANGCHAIN 路径）。"""
    from agent.models import get_chat_model

    for chunk in get_chat_model().stream(messages):
        content = getattr(chunk, "content", None)
        if content:
            yield content


def _is_checkpoint_enabled() -> bool:
    return os.getenv("ENABLE_CHECKPOINT", "0") == "1"
