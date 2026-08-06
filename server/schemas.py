"""server/schemas.py — Pydantic 请求/响应模型"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str
    message: str
    rag_enabled: bool = True
    visual_enabled: bool = False
    kg_enabled: bool = False


class SessionCreate(BaseModel):
    title: Optional[str] = None


class SessionRename(BaseModel):
    title: str


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: int
    updated_at: int


class SourceOut(BaseModel):
    id: Optional[int] = None
    source_file: str = ""
    heading_path: list = Field(default_factory=list)
    score: Optional[float] = None
    content: str = ""
    modality: str = "text"


class MessageOut(BaseModel):
    id: int
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    sources: Optional[list[SourceOut]] = None
    created_at: int
