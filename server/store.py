"""server/store.py — 会话/消息 SQLite 存储

- 标准库 sqlite3（WAL），异步门面用 asyncio.to_thread 包装（单用户足够）；
- 与 data_db/rag_vectors.db、agent_checkpoints.db、episodic.db 分离；
- 职责：app.db 是 UI 会话/消息的唯一事实源；agent 内部多轮记忆由
  langgraph checkpoint（thread_id=session_id）负责。
"""

import asyncio
import json
import sqlite3
import time
import uuid
from pathlib import Path

_DB_DIR = Path(__file__).resolve().parent.parent / "data_db"
DB_PATH = _DB_DIR / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  id         TEXT PRIMARY KEY,
  title      TEXT NOT NULL DEFAULT '新对话',
  created_at INTEGER NOT NULL,
  updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role       TEXT NOT NULL CHECK (role IN ('user','assistant')),
  content    TEXT NOT NULL,
  sources    TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


def _row_to_session(row) -> dict:
    return {
        "id": row[0],
        "title": row[1],
        "created_at": row[2],
        "updated_at": row[3],
    }


class SessionStore:
    def __init__(self, db: str | Path = DB_PATH):
        self._db = str(db)
        _DB_DIR.mkdir(parents=True, exist_ok=True)
        self._init()

    # ── 同步实现（私有）──────────────────────────
    def _connect(self) -> sqlite3.Connection:
        """打开连接并启用外键级联（否则 ON DELETE CASCADE 被 SQLite 忽略）。"""
        conn = sqlite3.connect(self._db)
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            # 清理历史遗留的孤儿消息（会话已删但消息残留）
            conn.execute("DELETE FROM messages WHERE session_id NOT IN (SELECT id FROM sessions)")

    def _list_sessions_sync(self) -> list:
        with self._connect() as c:
            rows = c.execute(
                "SELECT id,title,created_at,updated_at FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        return [_row_to_session(r) for r in rows]

    def _create_session_sync(self, title: str | None) -> dict:
        sid = uuid.uuid4().hex
        now = int(time.time() * 1000)
        title = (title or "新对话").strip() or "新对话"
        with self._connect() as c:
            c.execute(
                "INSERT INTO sessions (id,title,created_at,updated_at) VALUES (?,?,?,?)",
                (sid, title, now, now),
            )
        return {"id": sid, "title": title, "created_at": now, "updated_at": now}

    def _get_session_sync(self, sid: str) -> dict | None:
        with self._connect() as c:
            row = c.execute(
                "SELECT id,title,created_at,updated_at FROM sessions WHERE id=?", (sid,)
            ).fetchone()
        return _row_to_session(row) if row else None

    def _delete_session_sync(self, sid: str) -> None:
        with self._connect() as c:
            c.execute("DELETE FROM sessions WHERE id=?", (sid,))

    def _rename_session_sync(self, sid: str, title: str) -> bool:
        now = int(time.time() * 1000)
        with self._connect() as c:
            cur = c.execute(
                "UPDATE sessions SET title=?, updated_at=? WHERE id=?",
                (title, now, sid),
            )
        return cur.rowcount > 0

    def _list_messages_sync(self, sid: str) -> list:
        with self._connect() as c:
            rows = c.execute(
                "SELECT id,session_id,role,content,sources,created_at "
                "FROM messages WHERE session_id=? ORDER BY id",
                (sid,),
            ).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "session_id": r[1],
                "role": r[2],
                "content": r[3],
                "sources": json.loads(r[4]) if r[4] else None,
                "created_at": r[5],
            })
        return out

    def _add_message_sync(self, sid: str, role: str, content: str, sources: list | None = None) -> dict:
        now = int(time.time() * 1000)
        src = json.dumps(sources, ensure_ascii=False) if sources is not None else None
        with self._connect() as c:
            cur = c.execute(
                "INSERT INTO messages (session_id,role,content,sources,created_at) VALUES (?,?,?,?,?)",
                (sid, role, content, src, now),
            )
            c.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now, sid))
            msg_id = cur.lastrowid
        return {
            "id": msg_id,
            "session_id": sid,
            "role": role,
            "content": content,
            "sources": sources,
            "created_at": now,
        }

    def _auto_title_sync(self, sid: str) -> None:
        """首条 user 消息截断 40 字作会话标题（若仍是默认标题）。"""
        with self._connect() as c:
            row = c.execute(
                "SELECT content FROM messages WHERE session_id=? AND role='user' ORDER BY id LIMIT 1",
                (sid,),
            ).fetchone()
            cur = c.execute("SELECT title FROM sessions WHERE id=?", (sid,)).fetchone()
        if not row or not cur or cur[0] != "新对话":
            return
        title = row[0].strip().replace("\n", " ")[:40]
        if title:
            with self._connect() as c:
                c.execute("UPDATE sessions SET title=? WHERE id=?", (title, sid))

    # ── 异步门面（public）──────────────────────
    async def list_sessions(self) -> list:
        return await asyncio.to_thread(self._list_sessions_sync)

    async def create_session(self, title: str | None = None) -> dict:
        return await asyncio.to_thread(self._create_session_sync, title)

    def _create_session_with_id_sync(self, sid: str, title: str | None) -> dict | None:
        """按指定 id 创建会话（幂等：已存在则直接返回）。"""
        now = int(time.time() * 1000)
        title = (title or "新对话").strip() or "新对话"
        with self._connect() as c:
            c.execute(
                "INSERT OR IGNORE INTO sessions (id,title,created_at,updated_at) VALUES (?,?,?,?)",
                (sid, title, now, now),
            )
        return self._get_session_sync(sid)

    async def create_session_with_id(self, sid: str, title: str | None = None) -> dict | None:
        return await asyncio.to_thread(self._create_session_with_id_sync, sid, title)

    async def get_session(self, sid: str) -> dict | None:
        return await asyncio.to_thread(self._get_session_sync, sid)

    async def delete_session(self, sid: str) -> None:
        await asyncio.to_thread(self._delete_session_sync, sid)

    async def rename_session(self, sid: str, title: str) -> bool:
        return await asyncio.to_thread(self._rename_session_sync, sid, title)

    async def list_messages(self, sid: str) -> list:
        return await asyncio.to_thread(self._list_messages_sync, sid)

    async def add_message(self, sid: str, role: str, content: str, sources: list | None = None) -> dict:
        return await asyncio.to_thread(self._add_message_sync, sid, role, content, sources)

    async def auto_title(self, sid: str) -> None:
        await asyncio.to_thread(self._auto_title_sync, sid)


store = SessionStore()  # 模块级单例
