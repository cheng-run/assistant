"""server/routes/sessions.py — 会话 CRUD + 消息列表"""

from fastapi import APIRouter, HTTPException

from server.schemas import MessageOut, SessionCreate, SessionOut
from server.store import store

router = APIRouter()


@router.post("/sessions", status_code=201, response_model=SessionOut)
async def create_session(body: SessionCreate):
    return await store.create_session(body.title)


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions():
    return await store.list_sessions()


@router.get("/sessions/{sid}", response_model=SessionOut)
async def get_session(sid: str):
    session = await store.get_session(sid)
    if not session:
        raise HTTPException(404, "会话不存在")
    return session


@router.get("/sessions/{sid}/messages", response_model=list[MessageOut])
async def list_messages(sid: str):
    session = await store.get_session(sid)
    if not session:
        raise HTTPException(404, "会话不存在")
    return await store.list_messages(sid)


@router.patch("/sessions/{sid}", response_model=SessionOut)
async def rename_session(sid: str, body: SessionCreate):
    title = (body.title or "新对话").strip() or "新对话"
    ok = await store.rename_session(sid, title)
    if not ok:
        raise HTTPException(404, "会话不存在")
    return await store.get_session(sid)


@router.delete("/sessions/{sid}", status_code=204)
async def delete_session(sid: str):
    await store.delete_session(sid)
