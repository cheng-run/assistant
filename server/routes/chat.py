"""server/routes/chat.py — POST /api/chat（SSE 流式 + 消息持久化）"""

import asyncio

from fastapi import APIRouter, HTTPException

from agent.service import stream_events
from server.schemas import ChatRequest
from server.sse import event_stream, sse
from server.store import store

router = APIRouter()


@router.post("/chat")
async def chat(body: ChatRequest):
    session = await store.get_session(body.session_id)
    if not session:
        # 会话不存在 → 自动创建并继续聊天（幂等，避免前端陈旧 session_id 报 404）
        session = await store.create_session_with_id(body.session_id)
        if not session:
            raise HTTPException(500, "创建会话失败")

    # 由 app.db 重建上下文（等效原 history[-40:]）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in await store.list_messages(body.session_id)
    ][-40:]
    await store.add_message(body.session_id, "user", body.message)

    async def gen():
        tokens: list[str] = []
        sources: list[dict] = []
        try:
            async for ev in stream_events(
                body.message, history,
                body.rag_enabled, body.visual_enabled, body.kg_enabled,
                body.session_id,
            ):
                t = ev["type"]
                if t == "token":
                    tokens.append(ev["delta"])
                    yield sse("token", {"delta": ev["delta"], "text": ev["text"]})
                elif t == "status":
                    yield sse("status", {"kind": ev["kind"], "text": ev["text"]})
                elif t == "source":
                    sources.append(ev)
                    yield sse("source", ev)
                elif t == "error":
                    yield sse("error", {"message": ev["message"]})

            # 正常结束：持久化 assistant 消息 + done
            if tokens:
                msg = await store.add_message(
                    body.session_id, "assistant", "".join(tokens), sources=sources
                )
                await store.auto_title(body.session_id)
                yield sse("done", {
                    "answer": "".join(tokens),
                    "sources": sources,
                    "message_id": msg["id"],
                    "session_id": body.session_id,
                })
        except asyncio.CancelledError:
            # 客户端断开：持久化已累积的部分，不 yield（连接已断）
            if tokens:
                await store.add_message(
                    body.session_id, "assistant", "".join(tokens), sources=sources
                )
                await store.auto_title(body.session_id)
            raise

    return event_stream(gen())
