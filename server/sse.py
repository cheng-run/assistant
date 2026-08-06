"""server/sse.py — SSE 帧与统一响应头（手写，零额外依赖）"""

import json

from fastapi.responses import StreamingResponse


def sse(event: str, data: dict) -> str:
    """把事件编码为 `event: <type>\\ndata: <json>\\n\\n` 帧。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def event_stream(gen) -> StreamingResponse:
    """包装异步生成器为标准 SSE 响应（含防缓冲头）。"""
    return StreamingResponse(
        gen,
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 防 nginx/proxy 缓冲 SSE
        },
    )
