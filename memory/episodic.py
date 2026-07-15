"""
Layer 3 — 情景记忆（长期记忆）
SQLite + FTS5 持久化存储

存储架构：
  SQLite 负责结构化存储与复杂查询
  FTS5  负责高效全文检索（类向量检索）
  embedding 列预留未来向量检索接入

数据库路径：{助手目录}/data_db/episodic.db

认知机制：
  - 基于重要性：低于阈值的记忆被清除
  - 基于时间：超过时限的记忆被淘汰
  - 基于容量：超软上限驱逐最不重要的记忆
"""
import json
import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timedelta
from .base import extract_text, is_cjk
from .episodic_retriever import EpisodicRetriever

# ── 数据库路径 ──────────────────────────
_ASSISTANT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_DB_DIR = os.path.join(_ASSISTANT_DIR, "data_db")
_DEFAULT_DB_PATH = os.path.join(_DEFAULT_DB_DIR, "episodic.db")


class EpisodicMemory:
    """
    长期情景记忆 — SQLite 持久化版

    表结构:
        episodic_memories  — 主数据表（结构化存储）
        episodic_fts       — FTS5 虚拟表（全文检索）

    用法:
        em = EpisodicMemory()
        em.add("用户叫小明", importance=0.9)
        results = em.search("小明")         # FTS5 全文检索
        top = em.get_top(10)                # 按有效重要性排序
        em.forget_by_time(90)              # 遗忘 90 天前的记忆
    """

    def __init__(self, db_path: str = None, soft_limit: int = 100):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._soft_limit = soft_limit
        self._lock = threading.Lock()
        self._retriever = EpisodicRetriever()
        self._init_db()

    # ═══════════════════════════════════════
    #  数据库初始化
    # ═══════════════════════════════════════

    def _init_db(self):
        """创建数据库目录、表结构、索引和 FTS5 全文检索引擎"""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        with self._get_conn() as conn:
            # 性能优化
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-8000")  # 8MB 缓存

            # 主数据表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodic_memories (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    content         TEXT    NOT NULL,
                    memory_type     TEXT    DEFAULT 'episodic',
                    importance      REAL    DEFAULT 0.7,
                    timestamp       TEXT    NOT NULL,
                    searchable_text TEXT    DEFAULT '',
                    embedding       BLOB    DEFAULT NULL,
                    created_at      TEXT    DEFAULT (datetime('now')),
                    updated_at      TEXT    DEFAULT (datetime('now'))
                )
            """)

            # FTS5 全文检索虚拟表
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS episodic_fts USING fts5(
                    searchable_text,
                    content='episodic_memories',
                    content_rowid='id'
                )
            """)

            # 索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_em_type "
                         "ON episodic_memories(memory_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_em_importance "
                         "ON episodic_memories(importance DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_em_timestamp "
                         "ON episodic_memories(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_em_type_importance "
                         "ON episodic_memories(memory_type, importance DESC)")

            # FTS 同步触发器：INSERT
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS em_fts_ai AFTER INSERT ON episodic_memories
                BEGIN
                    INSERT INTO episodic_fts(rowid, searchable_text)
                    VALUES (new.id, new.searchable_text);
                END
            """)
            # FTS 同步触发器：DELETE
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS em_fts_ad AFTER DELETE ON episodic_memories
                BEGIN
                    INSERT INTO episodic_fts(episodic_fts, rowid, searchable_text)
                    VALUES ('delete', old.id, old.searchable_text);
                END
            """)
            # FTS 同步触发器：UPDATE
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS em_fts_au AFTER UPDATE ON episodic_memories
                BEGIN
                    INSERT INTO episodic_fts(episodic_fts, rowid, searchable_text)
                    VALUES ('delete', old.id, old.searchable_text);
                    INSERT INTO episodic_fts(rowid, searchable_text)
                    VALUES (new.id, new.searchable_text);
                END
            """)

            conn.commit()

    def _get_conn(self):
        """获取数据库连接（WAL 模式，线程安全）。返回上下文管理器，退出时自动关闭。"""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return closing(conn)

    # ═══════════════════════════════════════
    #  写入
    # ═══════════════════════════════════════

    def add(self, content, memory_type: str = "episodic", importance: float = 0.7):
        """添加一条长期记忆"""
        content_json = self._serialize_content(content)
        searchable = self._extract_text(content)
        now = datetime.now().isoformat()

        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO episodic_memories
                       (content, memory_type, importance, timestamp, searchable_text)
                       VALUES (?, ?, ?, ?, ?)""",
                    (content_json, memory_type, importance, now, searchable),
                )
                conn.commit()

    def add_batch(self, entries: list, memory_type: str = "episodic"):
        """批量添加记忆（事务包装，高效写入）"""
        now = datetime.now().isoformat()
        rows = []
        for entry in entries:
            content = entry.get("content")
            content_json = self._serialize_content(content)
            searchable = self._extract_text(content)
            importance = entry.get("importance", 0.7)
            rows.append((content_json, memory_type, importance, now, searchable))

        with self._lock:
            with self._get_conn() as conn:
                conn.executemany(
                    """INSERT INTO episodic_memories
                       (content, memory_type, importance, timestamp, searchable_text)
                       VALUES (?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()

    # ═══════════════════════════════════════
    #  读取
    # ═══════════════════════════════════════

    def get_all(self) -> list:
        """返回所有长期记忆"""
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM episodic_memories ORDER BY timestamp DESC"
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_type(self, memory_type: str) -> list:
        """按类型筛选记忆"""
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM episodic_memories WHERE memory_type = ? "
                    "ORDER BY importance DESC",
                    (memory_type,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_by_id(self, memory_id: int) -> dict | None:
        """按 ID 获取单条记忆"""
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT * FROM episodic_memories WHERE id = ?", (memory_id,)
                ).fetchone()
        return self._row_to_dict(row) if row else None

    def search(self, query: str, limit: int = 5, **filters) -> list:
        """
        情景记忆混合检索：结构化过滤 + FTS5 召回 + 语义评分 + 综合排序。

        检索流程：
          1. SQL 层：结构化过滤 + FTS5 候选召回（limit × 5 扩大池）
          2. Python 层：TF-IDF 语义评分 + 时间近因性 + 重要性加权
          3. 综合排序返回 Top-K

        结构化过滤参数（可选）：
            min_importance: float  最低重要性阈值
            max_age_days:   int    最大时间跨度（天）
            memory_type:    str    记忆类型筛选

        评分公式：
            (tfidf × 0.8 + recency × 0.2) × (0.8 + importance × 0.4)
        """
        # ── 1. SQL 层：结构化过滤 + FTS5 召回（filter 在此完成） ──
        with self._lock:
            with self._get_conn() as conn:
                rows = self._search_rows(conn, query, limit * 5, **filters)

        if not rows:
            return []

        # ── 2. Python 层：语义评分 + 综合排序（不再重复过滤） ──
        memories = [self._row_to_dict(r) for r in rows]
        return self._retriever.retrieve(query, memories, limit)

    def _search_rows(self, conn, query: str, limit: int, **filters) -> list:
        """
        SQL 层候选检索：结构化 WHERE 过滤 + FTS5 MATCH。

        先做结构化过滤（重要性/类型/时间），再做 FTS5 语义匹配，
        大幅缩小候选集，提升后续 TF-IDF 语义评分的效率。
        """
        where = []
        params = []

        # 结构化过滤条件
        if "min_importance" in filters:
            where.append("em.importance >= ?")
            params.append(filters["min_importance"])
        if "memory_type" in filters:
            where.append("em.memory_type = ?")
            params.append(filters["memory_type"])
        if "max_age_days" in filters:
            cutoff = (datetime.now() - timedelta(days=filters["max_age_days"])).isoformat()
            where.append("em.timestamp >= ?")
            params.append(cutoff)

        # FTS5 语义匹配
        fts_query = self._build_fts_query(query)
        where.append("em.id IN (SELECT rowid FROM episodic_fts WHERE episodic_fts MATCH ?)")
        params.append(fts_query)

        try:
            rows = conn.execute(
                f"SELECT em.* FROM episodic_memories em WHERE {' AND '.join(where)} LIMIT ?",
                params + [limit],
            ).fetchall()
        except sqlite3.OperationalError:
            # FTS 失败 → LIKE 回退，保留结构化过滤
            where.pop()  # 移除 FTS 条件
            params.pop()
            like_clause = "em.searchable_text LIKE ?"
            where.append(like_clause)
            params.append(f"%{query}%")
            sql = "SELECT em.* FROM episodic_memories em"
            if where:
                sql += " WHERE " + " AND ".join(where)
            sql += " ORDER BY em.importance DESC LIMIT ?"
            rows = conn.execute(sql, params + [limit]).fetchall()

        return rows

    def get_top(self, limit: int = 10) -> list:
        """获取重要性最高的 N 条记忆"""
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM episodic_memories ORDER BY importance DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ═══════════════════════════════════════
    #  遗忘：策略一 — 基于重要性
    # ═══════════════════════════════════════

    def forget_by_importance(self, threshold: float = 0.3) -> int:
        """遗忘所有（原始）重要性 < threshold 的记忆"""
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM episodic_memories WHERE importance < ?",
                    (threshold,),
                )
                conn.commit()
                return cursor.rowcount

    # ═══════════════════════════════════════
    #  遗忘：策略二 — 基于时间
    # ═══════════════════════════════════════

    def forget_by_time(self, max_age_days: int = 90) -> int:
        """遗忘超过 max_age_days 天的记忆"""
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        with self._lock:
            with self._get_conn() as conn:
                cursor = conn.execute(
                    "DELETE FROM episodic_memories WHERE timestamp < ?",
                    (cutoff,),
                )
                conn.commit()
                return cursor.rowcount

    # ═══════════════════════════════════════
    #  遗忘：策略三 — 基于容量
    # ═══════════════════════════════════════

    def forget_by_capacity(self, soft_limit: int = None) -> int:
        """
        超过软上限时，驱逐重要性最低的记忆。
        返回被遗忘的条数。
        """
        if soft_limit is None:
            soft_limit = self._soft_limit

        with self._lock:
            with self._get_conn() as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM episodic_memories"
                ).fetchone()[0]
                if count <= soft_limit:
                    return 0

                excess = count - soft_limit
                cursor = conn.execute(
                    """DELETE FROM episodic_memories WHERE id IN (
                        SELECT id FROM episodic_memories
                        ORDER BY importance ASC LIMIT ?
                    )""",
                    (excess,),
                )
                conn.commit()
                return cursor.rowcount

    # ═══════════════════════════════════════
    #  整合
    # ═══════════════════════════════════════

    def extract_by_importance(self, threshold: float = 0.7) -> list:
        """
        提取并移除重要性 >= threshold 的记忆。
        用于记忆整合（episodic → semantic 等）。
        返回被提取的记忆列表。
        """
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM episodic_memories WHERE importance >= ?",
                    (threshold,),
                ).fetchall()
                extracted = [self._row_to_dict(r) for r in rows]
                if extracted:
                    ids = [(r["id"],) for r in rows]
                    conn.executemany(
                        "DELETE FROM episodic_memories WHERE id = ?", ids
                    )
                    conn.commit()
                return extracted

    # ═══════════════════════════════════════
    #  增强
    # ═══════════════════════════════════════

    def reinforce(self, memory_id: int, boost: float = 0.2):
        """
        增强记忆（模拟复述/回忆效应）：
        提升重要性并刷新时间戳（重新开始衰减曲线）。
        """
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE episodic_memories
                       SET importance = MIN(1.0, importance + ?),
                           timestamp = ?,
                           updated_at = datetime('now')
                       WHERE id = ?""",
                    (boost, datetime.now().isoformat(), memory_id),
                )
                conn.commit()

    def reinforce_by_query(self, query: str, boost: float = 0.15):
        """
        搜索并增强匹配的记忆（模拟主动回忆强化）。

        注意：当前通过 searchable_text LIKE 匹配，
              比 FTS 更宽泛，确保相关记忆都被强化。
        """
        with self._lock:
            with self._get_conn() as conn:
                conn.execute(
                    """UPDATE episodic_memories
                       SET importance = MIN(1.0, importance + ?),
                           timestamp = ?,
                           updated_at = datetime('now')
                       WHERE searchable_text LIKE ?""",
                    (boost, datetime.now().isoformat(), f"%{query}%"),
                )
                conn.commit()

    # ═══════════════════════════════════════
    #  维护
    # ═══════════════════════════════════════

    def clear(self):
        """清空所有长期记忆"""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("DELETE FROM episodic_memories")
                # FTS5 也需要清空（外部内容表的 delete 索引）
                conn.execute("DELETE FROM episodic_fts")
                conn.commit()

    def vacuum(self):
        """数据库压缩优化"""
        with self._lock:
            with self._get_conn() as conn:
                conn.execute("VACUUM")

    # ═══════════════════════════════════════
    #  统计查询
    # ═══════════════════════════════════════

    def count_by_type(self) -> dict:
        """按类型统计记忆数量"""
        with self._lock:
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT memory_type, COUNT(*) as cnt "
                    "FROM episodic_memories GROUP BY memory_type"
                ).fetchall()
        return {r["memory_type"]: r["cnt"] for r in rows}

    def avg_importance(self) -> float:
        """平均重要性"""
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute(
                    "SELECT AVG(importance) FROM episodic_memories"
                ).fetchone()
        return round(row[0], 3) if row[0] else 0.0

    # ═══════════════════════════════════════
    #  序列化 / 反序列化
    # ═══════════════════════════════════════

    @staticmethod
    def _serialize_content(content) -> str:
        """将 content 序列化为数据库存储格式"""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)

    @staticmethod
    def _deserialize_content(raw: str):
        """从数据库读取的 content 反序列化"""
        if not raw:
            return raw
        if raw.startswith("{") or raw.startswith("["):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                pass
        return raw

    @staticmethod
    def _extract_text(content) -> str:
        """从 content 提取纯文本，CJK 字符间插入空格以支持 FTS5 分词"""
        text = extract_text(content)
        return EpisodicMemory._space_cjk(text)

    @staticmethod
    def _space_cjk(text: str) -> str:
        """在 CJK 字符周围插入空格，使 FTS5 能按字分词。复用 base.is_cjk 统一判断。"""
        result = []
        for ch in text:
            if is_cjk(ch):
                result.append(f" {ch} ")
            else:
                result.append(ch)
        return "".join(result).strip()

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict | None:
        """将数据库行转换为 dict（兼容旧 API）"""
        if row is None:
            return None
        return {
            "id": row["id"],
            "content": self._deserialize_content(row["content"]),
            "type": row["memory_type"],
            "importance": row["importance"],
            "timestamp": row["timestamp"],
            "embedding": row["embedding"],
        }

    # ═══════════════════════════════════════
    #  FTS5 辅助
    # ═══════════════════════════════════════

    @staticmethod
    def _build_fts_query(query: str) -> str:
        """
        构建 FTS5 查询字符串。

        CJK 查询做空格分词处理（与 _extract_text 中的存储方式一致），
        使 FTS5 能按单字做 n-gram 匹配。
        """
        processed = EpisodicMemory._space_cjk(query)
        if not processed.strip():
            return query
        # 每个 token 加 * 支持前缀匹配
        terms = processed.split()
        return " OR ".join(f'"{t}"*' for t in terms)

    # ═══════════════════════════════════════
    #  元信息
    # ═══════════════════════════════════════

    def __len__(self) -> int:
        with self._lock:
            with self._get_conn() as conn:
                row = conn.execute("SELECT COUNT(*) FROM episodic_memories").fetchone()
        return row[0] if row else 0

    @property
    def soft_limit(self) -> int:
        return self._soft_limit

    @property
    def db_path(self) -> str:
        return self._db_path
