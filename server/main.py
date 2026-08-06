"""server/main.py — FastAPI 应用装配

- 生产：uvicorn 托管 frontend/dist 静态文件 + SPA 兜底（Vue Router history 模式）
- 开发：Vite dev proxy 转发 /api（CORS 仅作双保险）
"""

from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

load_dotenv()

from server.routes import chat, config, sessions, upload  # noqa: E402

app = FastAPI(title="智能文档问答助手")

# 开发时由 Vite 代理，CORS 仅作双保险
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api")
app.include_router(sessions.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(config.router, prefix="/api")

_DIST = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@app.get("/{full_path:path}")
async def spa(full_path: str):
    """生产模式 SPA 兜底：静态文件优先，其余回落到 index.html。"""
    if full_path.startswith("api/"):
        raise HTTPException(404)
    if _DIST.is_dir():
        file = _DIST / full_path
        if full_path and file.is_file():
            return FileResponse(file)
        index = _DIST / "index.html"
        if index.is_file():
            return FileResponse(index)
    raise HTTPException(404, "前端未构建（生产模式需先 npm run build）")
