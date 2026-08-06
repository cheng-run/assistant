"""server/routes/upload.py — 文档上传（SSE 进度）+ 文档管理"""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, UploadFile

from agent.tools import rag
from server.service import run_upload
from server.sse import event_stream

router = APIRouter()

_UPLOAD_DIR = Path(__file__).resolve().parent.parent.parent / "data_db" / "uploads"


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    visual: bool = Form(False),
    kg: bool = Form(False),
):
    """接收文件 → 分块写盘（不占内存）→ SSE 进度流。"""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fname = (file.filename or "upload").replace("\\", "/").split("/")[-1]
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}_{fname}"
    with dest.open("wb") as f:
        while chunk := await file.read(1 << 20):  # 1MB 分块
            f.write(chunk)
    return event_stream(run_upload(str(dest), fname, visual, kg))


@router.get("/docs/sources")
async def list_sources():
    return {
        "sources": rag.vector_store.list_sources(),
        "outline": rag.get_all_outlines(),
    }


@router.delete("/docs")
async def clear_docs():
    rag.vector_store.clear()
    try:
        rag.kg_store.clear()
    except Exception:
        pass
    return {"cleared": True}


@router.delete("/docs/{source}")
async def remove_source(source: str):
    rag.vector_store.clear_by_source(source)
    return {"removed": True}
