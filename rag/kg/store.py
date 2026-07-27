"""
知识图谱存储 — SQLite 持久化图结构

表结构:
  - kg_entities:      实体节点
  - kg_relations:     关系边
  - kg_communities:   社区（聚类后的实体群组）
  - kg_chunk_entity_refs: chunk-entity 交叉引用

设计模式与 embedding.py 的 VectorStore 保持一致:
  - 持久连接 + check_same_thread=False
  - threading.Lock 线程安全
  - WAL 模式 + synchronous=NORMAL
  - row_factory = sqlite3.Row
"""
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── 默认存储路径 ──────────────────────
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent.parent / "data_db"
_DEFAULT_DB_PATH = str(_DEFAULT_DB_DIR / "kg_graph.db")


class KnowledgeGraphStore:
    """
    SQLite 图存储 — 实体、关系、社区的 CRUD 操作。

    用法:
        store = KnowledgeGraphStore()
        ids = store.add_entities(entities, source_file="doc.pdf")
        count = store.add_relations(relations, source_file="doc.pdf")
        adj = store.build_adjacency()
        store.save_communities(communities)
    """

    def __init__(self, db_path: str = None):
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._conn_persist = None
        self._init_db()

    # ── 数据库初始化 ────────────────────

    def _init_db(self):
        import sqlite3
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_entities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    name            TEXT    NOT NULL,
                    canonical_name  TEXT    NOT NULL,
                    entity_type     TEXT    DEFAULT '',
                    description     TEXT    DEFAULT '',
                    chunk_refs      TEXT    DEFAULT '[]',
                    source_file     TEXT    DEFAULT '',
                    embedding       TEXT    DEFAULT '',
                    created_at      TEXT    DEFAULT (datetime('now'))
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_relations (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id       INTEGER NOT NULL,
                    target_id       INTEGER NOT NULL,
                    relation        TEXT    NOT NULL,
                    description     TEXT    DEFAULT '',
                    weight          REAL    DEFAULT 1.0,
                    chunk_refs      TEXT    DEFAULT '[]',
                    source_file     TEXT    DEFAULT '',
                    created_at      TEXT    DEFAULT (datetime('now')),
                    FOREIGN KEY (source_id) REFERENCES kg_entities(id) ON DELETE CASCADE,
                    FOREIGN KEY (target_id) REFERENCES kg_entities(id) ON DELETE CASCADE
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_communities (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    level           INTEGER DEFAULT 0,
                    label           TEXT    DEFAULT '',
                    entity_ids      TEXT    NOT NULL,
                    key_entities    TEXT    DEFAULT '[]',
                    summary         TEXT    DEFAULT '',
                    summary_embedding TEXT  DEFAULT '',
                    relation_count  INTEGER DEFAULT 0,
                    density         REAL    DEFAULT 0.0,
                    source_file     TEXT    DEFAULT '',
                    created_at      TEXT    DEFAULT (datetime('now'))
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS kg_chunk_entity_refs (
                    chunk_id        INTEGER NOT NULL,
                    entity_id       INTEGER NOT NULL,
                    source_file     TEXT    DEFAULT '',
                    PRIMARY KEY (chunk_id, entity_id)
                )
            """)

            # ── 索引 ──
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_entity_canonical "
                "ON kg_entities(canonical_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_entity_type "
                "ON kg_entities(entity_type)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_entity_source "
                "ON kg_entities(source_file)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_rel_source "
                "ON kg_relations(source_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_rel_target "
                "ON kg_relations(target_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_rel_source_file "
                "ON kg_relations(source_file)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_comm_level "
                "ON kg_communities(level)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_comm_source "
                "ON kg_communities(source_file)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_kg_cer_entity "
                "ON kg_chunk_entity_refs(entity_id)"
            )

            conn.commit()

    def _conn(self):
        """获取数据库连接（复用持久连接，线程安全）"""
        import sqlite3
        if self._conn_persist is None:
            self._conn_persist = sqlite3.connect(
                self._db_path, check_same_thread=False
            )
            self._conn_persist.row_factory = sqlite3.Row
        return self._conn_persist

    # ═══════════════════════════════════════════════════════════════
    #  实体操作
    # ═══════════════════════════════════════════════════════════════

    def add_entities(
        self, entities: List[Dict], source_file: str = ""
    ) -> List[int]:
        """
        批量插入实体。已存在的 canonical_name 则跳过（INSERT OR IGNORE）。

        参数:
            entities: [{"name": ..., "canonical_name": ..., "entity_type": ...,
                         "description": ..., "chunk_refs": [...], "embedding": [...]}, ...]
            source_file: 来源文档名
        返回:
            插入（或已存在）的实体 ID 列表
        """
        if not entities:
            return []

        ids = []
        with self._lock:
            with self._conn() as conn:
                for e in entities:
                    try:
                        conn.execute(
                            """INSERT OR IGNORE INTO kg_entities
                               (name, canonical_name, entity_type, description,
                                chunk_refs, source_file, embedding)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                e.get("name", e.get("canonical_name", "")),
                                e.get("canonical_name", e.get("name", "")),
                                e.get("entity_type", ""),
                                e.get("description", ""),
                                json.dumps(e.get("chunk_refs", [])),
                                source_file or e.get("source_file", ""),
                                json.dumps(e.get("embedding", []))
                                if e.get("embedding") else "",
                            ),
                        )
                    except Exception:
                        # INSERT OR IGNORE 对 UNIQUE 约束已足够，
                        # 但某些情况下可能因其他原因失败
                        pass

                # 获取所有相关实体的 ID
                for e in entities:
                    canonical = e.get("canonical_name", e.get("name", ""))
                    row = conn.execute(
                        "SELECT id FROM kg_entities WHERE canonical_name = ?",
                        (canonical,),
                    ).fetchone()
                    if row:
                        ids.append(row["id"])

        return ids

    def upsert_entity(
        self, entity: Dict, source_file: str = ""
    ) -> int:
        """
        插入或更新实体。如果 canonical_name 已存在，更新 chunk_refs 和 description。
        返回实体 ID。
        """
        canonical = entity.get("canonical_name", entity.get("name", ""))
        with self._lock:
            with self._conn() as conn:
                existing = conn.execute(
                    "SELECT id, chunk_refs FROM kg_entities WHERE canonical_name = ?",
                    (canonical,),
                ).fetchone()

                if existing:
                    # 合并 chunk_refs
                    old_refs = json.loads(existing["chunk_refs"]) if existing["chunk_refs"] else []
                    new_refs = entity.get("chunk_refs", [])
                    merged = list(set(old_refs + new_refs))
                    conn.execute(
                        """UPDATE kg_entities
                           SET chunk_refs = ?,
                               description = CASE WHEN description = '' THEN ? ELSE description END
                           WHERE id = ?""",
                        (json.dumps(merged), entity.get("description", ""), existing["id"]),
                    )
                    conn.commit()
                    return existing["id"]
                else:
                    cursor = conn.execute(
                        """INSERT INTO kg_entities
                           (name, canonical_name, entity_type, description,
                            chunk_refs, source_file, embedding)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entity.get("name", canonical),
                            canonical,
                            entity.get("entity_type", ""),
                            entity.get("description", ""),
                            json.dumps(entity.get("chunk_refs", [])),
                            source_file or entity.get("source_file", ""),
                            json.dumps(entity.get("embedding", []))
                            if entity.get("embedding") else "",
                        ),
                    )
                    conn.commit()
                    return cursor.lastrowid

    def get_entities(
        self, entity_ids: List[int] = None, source_file: str = None
    ) -> Dict[int, Dict]:
        """
        获取实体详情。

        参数:
            entity_ids:  实体 ID 列表（可选）
            source_file: 按来源文件筛选（可选）
        返回:
            {entity_id: {id, name, canonical_name, entity_type, description,
                         chunk_refs: [...], source_file, embedding: [...]}}
        """
        result = {}
        with self._lock:
            with self._conn() as conn:
                if entity_ids is not None:
                    placeholders = ",".join("?" * len(entity_ids))
                    rows = conn.execute(
                        f"SELECT * FROM kg_entities WHERE id IN ({placeholders})",
                        entity_ids,
                    ).fetchall()
                elif source_file:
                    rows = conn.execute(
                        "SELECT * FROM kg_entities WHERE source_file = ?",
                        (source_file,),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM kg_entities").fetchall()

                for row in rows:
                    d = dict(row)
                    d["chunk_refs"] = json.loads(d["chunk_refs"]) if d["chunk_refs"] else []
                    d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else []
                    result[d["id"]] = d

        return result

    def get_entity_by_name(self, name: str) -> Optional[Dict]:
        """按 canonical_name 查找实体"""
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT * FROM kg_entities WHERE canonical_name = ?",
                    (name,),
                ).fetchone()
                if row:
                    d = dict(row)
                    d["chunk_refs"] = json.loads(d["chunk_refs"]) if d["chunk_refs"] else []
                    d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else []
                    return d
        return None

    def fuzzy_search_entities(self, query: str, top_k: int = 10) -> List[Dict]:
        """按名称模糊搜索实体（LIKE 匹配）"""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM kg_entities WHERE name LIKE ? OR canonical_name LIKE ? "
                    "LIMIT ?",
                    (f"%{query}%", f"%{query}%", top_k),
                ).fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    d["chunk_refs"] = json.loads(d["chunk_refs"]) if d["chunk_refs"] else []
                    d["embedding"] = json.loads(d["embedding"]) if d["embedding"] else []
                    results.append(d)
        return results

    # ═══════════════════════════════════════════════════════════════
    #  关系操作
    # ═══════════════════════════════════════════════════════════════

    def add_relations(
        self, relations: List[Dict], source_file: str = ""
    ) -> int:
        """
        批量插入关系。自动将 source/target 实体名解析为 ID。

        参数:
            relations: [{"source": "entity_name", "target": "entity_name",
                          "relation": "...", "description": "...", "chunk_refs": [...]}, ...]
            source_file: 来源文档名
        返回:
            插入的关系数量
        """
        if not relations:
            return 0

        count = 0
        with self._lock:
            with self._conn() as conn:
                for r in relations:
                    src_name = r.get("source", "")
                    tgt_name = r.get("target", "")
                    if not src_name or not tgt_name:
                        continue

                    src = conn.execute(
                        "SELECT id FROM kg_entities WHERE canonical_name = ?",
                        (src_name,),
                    ).fetchone()
                    tgt = conn.execute(
                        "SELECT id FROM kg_entities WHERE canonical_name = ?",
                        (tgt_name,),
                    ).fetchone()

                    if not src or not tgt:
                        continue

                    # 检查是否已存在相同关系 → 加权重
                    existing = conn.execute(
                        "SELECT id, weight FROM kg_relations "
                        "WHERE source_id = ? AND target_id = ? AND relation = ?",
                        (src["id"], tgt["id"], r.get("relation", "")),
                    ).fetchone()

                    if existing:
                        conn.execute(
                            "UPDATE kg_relations SET weight = weight + 1.0 WHERE id = ?",
                            (existing["id"],),
                        )
                    else:
                        conn.execute(
                            """INSERT INTO kg_relations
                               (source_id, target_id, relation, description,
                                weight, chunk_refs, source_file)
                               VALUES (?, ?, ?, ?, 1.0, ?, ?)""",
                            (
                                src["id"], tgt["id"],
                                r.get("relation", ""),
                                r.get("description", ""),
                                json.dumps(r.get("chunk_refs", [])),
                                source_file or r.get("source_file", ""),
                            ),
                        )
                    count += 1

                conn.commit()

        return count

    def get_relations(
        self, source_file: str = None, entity_ids: List[int] = None
    ) -> List[Dict]:
        """
        获取关系。

        参数:
            source_file: 按来源文件筛选
            entity_ids:  只返回两端实体都在此列表中的关系（社区内部关系）
        返回:
            [{"id": ..., "source_id": ..., "target_id": ..., "relation": ...,
              "weight": ..., "chunk_refs": [...], ...}, ...]
        """
        with self._lock:
            with self._conn() as conn:
                if entity_ids is not None and len(entity_ids) > 0:
                    placeholders = ",".join("?" * len(entity_ids))
                    rows = conn.execute(
                        f"""SELECT * FROM kg_relations
                            WHERE source_id IN ({placeholders})
                              AND target_id IN ({placeholders})""",
                        entity_ids + entity_ids,
                    ).fetchall()
                elif source_file:
                    rows = conn.execute(
                        "SELECT * FROM kg_relations WHERE source_file = ?",
                        (source_file,),
                    ).fetchall()
                else:
                    rows = conn.execute("SELECT * FROM kg_relations").fetchall()

                results = []
                for row in rows:
                    d = dict(row)
                    d["chunk_refs"] = json.loads(d["chunk_refs"]) if d["chunk_refs"] else []
                    results.append(d)

        return results

    def get_relations_with_names(
        self, entity_ids: List[int] = None, limit: int = 50
    ) -> List[Dict]:
        """
        获取关系并附带实体名称（JOIN 查询），方便直接用于 LLM 上下文。
        """
        with self._lock:
            with self._conn() as conn:
                if entity_ids is not None and len(entity_ids) > 0:
                    placeholders = ",".join("?" * len(entity_ids))
                    rows = conn.execute(
                        f"""SELECT r.*, e1.name as source_name, e2.name as target_name
                            FROM kg_relations r
                            JOIN kg_entities e1 ON r.source_id = e1.id
                            JOIN kg_entities e2 ON r.target_id = e2.id
                            WHERE r.source_id IN ({placeholders})
                              AND r.target_id IN ({placeholders})
                            ORDER BY r.weight DESC
                            LIMIT ?""",
                        entity_ids + entity_ids + [limit],
                    ).fetchall()
                else:
                    rows = conn.execute(
                        """SELECT r.*, e1.name as source_name, e2.name as target_name
                           FROM kg_relations r
                           JOIN kg_entities e1 ON r.source_id = e1.id
                           JOIN kg_entities e2 ON r.target_id = e2.id
                           ORDER BY r.weight DESC
                           LIMIT ?""",
                        (limit,),
                    ).fetchall()

                results = []
                for row in rows:
                    d = dict(row)
                    d["chunk_refs"] = json.loads(d["chunk_refs"]) if d["chunk_refs"] else []
                    results.append(d)

        return results

    # ═══════════════════════════════════════════════════════════════
    #  图结构
    # ═══════════════════════════════════════════════════════════════

    def build_adjacency(
        self, source_file: str = None
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        从 kg_relations 构建邻接表（无向图）。

        参数:
            source_file: 可选，只构建特定文档的图
        返回:
            {entity_id: [(neighbor_entity_id, edge_weight), ...]}
        """
        relations = self.get_relations(source_file=source_file)
        adjacency: Dict[int, List[Tuple[int, float]]] = {}

        for r in relations:
            src = r["source_id"]
            tgt = r["target_id"]
            w = r["weight"]

            if src not in adjacency:
                adjacency[src] = []
            if tgt not in adjacency:
                adjacency[tgt] = []

            adjacency[src].append((tgt, w))
            adjacency[tgt].append((src, w))  # 无向边

        return adjacency

    # ═══════════════════════════════════════════════════════════════
    #  社区操作
    # ═══════════════════════════════════════════════════════════════

    def save_communities(
        self, communities: List[Dict], source_file: str = ""
    ) -> int:
        """
        批量保存社区。

        参数:
            communities: [{"level": ..., "entity_ids": [...],
                            "key_entities": [...], "summary": "...",
                            "summary_embedding": [...], "relation_count": ...,
                            "density": ...}, ...]
            source_file: 来源文档名
        返回:
            插入的社区数
        """
        if not communities:
            return 0

        with self._lock:
            with self._conn() as conn:
                for c in communities:
                    conn.execute(
                        """INSERT INTO kg_communities
                           (level, entity_ids, key_entities, summary,
                            summary_embedding, relation_count, density, source_file)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            c.get("level", 0),
                            json.dumps(c.get("entity_ids", [])),
                            json.dumps(c.get("key_entities", [])),
                            c.get("summary", ""),
                            json.dumps(c.get("summary_embedding", []))
                            if c.get("summary_embedding") else "",
                            c.get("relation_count", 0),
                            c.get("density", 0.0),
                            source_file or c.get("source_file", ""),
                        ),
                    )
                conn.commit()

        return len(communities)

    def get_all_communities(
        self, level: int = None, source_file: str = None
    ) -> List[Dict]:
        """
        获取社区列表。

        参数:
            level:       层级过滤（None = 所有层）
            source_file: 来源文件过滤
        返回:
            [{"id": ..., "level": ..., "entity_ids": [...],
              "key_entities": [...], "summary": ..., "summary_embedding": [...],
              "relation_count": ..., "density": ...}, ...]
        """
        conditions = []
        params = []

        if level is not None:
            conditions.append("level = ?")
            params.append(level)
        if source_file:
            conditions.append("source_file = ?")
            params.append(source_file)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    f"SELECT * FROM kg_communities {where} ORDER BY level, id",
                    params,
                ).fetchall()

                results = []
                for row in rows:
                    d = dict(row)
                    d["entity_ids"] = json.loads(d["entity_ids"]) if d["entity_ids"] else []
                    d["key_entities"] = json.loads(d["key_entities"]) if d["key_entities"] else []
                    d["summary_embedding"] = (
                        json.loads(d["summary_embedding"])
                        if d["summary_embedding"] else []
                    )
                    results.append(d)

        return results

    # ═══════════════════════════════════════════════════════════════
    #  交叉引用
    # ═══════════════════════════════════════════════════════════════

    def add_chunk_entity_refs(
        self, refs: List[Tuple[int, int]], source_file: str = ""
    ) -> None:
        """批量插入 chunk-entity 交叉引用（INSERT OR IGNORE）"""
        if not refs:
            return

        with self._lock:
            with self._conn() as conn:
                conn.executemany(
                    "INSERT OR IGNORE INTO kg_chunk_entity_refs "
                    "(chunk_id, entity_id, source_file) VALUES (?, ?, ?)",
                    [(chunk_id, entity_id, source_file) for chunk_id, entity_id in refs],
                )
                conn.commit()

    def get_chunks_for_entities(
        self, entity_ids: List[int]
    ) -> List[int]:
        """获取与给定实体关联的所有 chunk ID"""
        if not entity_ids:
            return []

        with self._lock:
            with self._conn() as conn:
                placeholders = ",".join("?" * len(entity_ids))
                rows = conn.execute(
                    f"SELECT DISTINCT chunk_id FROM kg_chunk_entity_refs "
                    f"WHERE entity_id IN ({placeholders})",
                    entity_ids,
                ).fetchall()

        return [r["chunk_id"] for r in rows]

    def get_entities_for_chunk(self, chunk_id: int) -> List[int]:
        """获取与给定 chunk 关联的实体 ID"""
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT entity_id FROM kg_chunk_entity_refs WHERE chunk_id = ?",
                    (chunk_id,),
                ).fetchall()

        return [r["entity_id"] for r in rows]

    # ═══════════════════════════════════════════════════════════════
    #  维护
    # ═══════════════════════════════════════════════════════════════

    def clear_by_source(self, source_file: str) -> Dict[str, int]:
        """删除指定来源文件的所有图数据"""
        counts = {}
        with self._lock:
            with self._conn() as conn:
                # 先删交叉引用
                cur = conn.execute(
                    "DELETE FROM kg_chunk_entity_refs WHERE source_file = ?",
                    (source_file,),
                )
                counts["chunk_refs"] = cur.rowcount

                # 删社区
                cur = conn.execute(
                    "DELETE FROM kg_communities WHERE source_file = ?",
                    (source_file,),
                )
                counts["communities"] = cur.rowcount

                # 删关系
                cur = conn.execute(
                    "DELETE FROM kg_relations WHERE source_file = ?",
                    (source_file,),
                )
                counts["relations"] = cur.rowcount

                # 删实体
                cur = conn.execute(
                    "DELETE FROM kg_entities WHERE source_file = ?",
                    (source_file,),
                )
                counts["entities"] = cur.rowcount

                conn.commit()

        return counts

    def clear(self) -> Dict[str, int]:
        """清空所有图数据"""
        counts = {}
        with self._lock:
            with self._conn() as conn:
                for table in ["kg_chunk_entity_refs", "kg_communities",
                              "kg_relations", "kg_entities"]:
                    cur = conn.execute(f"DELETE FROM {table}")
                    counts[table] = cur.rowcount
                conn.commit()

        return counts

    def stats(self) -> dict:
        """返回图统计信息"""
        with self._lock:
            with self._conn() as conn:
                entities = conn.execute(
                    "SELECT COUNT(*) FROM kg_entities"
                ).fetchone()[0]
                relations = conn.execute(
                    "SELECT COUNT(*) FROM kg_relations"
                ).fetchone()[0]
                communities = conn.execute(
                    "SELECT COUNT(*) FROM kg_communities"
                ).fetchone()[0]
                chunk_refs = conn.execute(
                    "SELECT COUNT(*) FROM kg_chunk_entity_refs"
                ).fetchone()[0]

        return {
            "entities": entities,
            "relations": relations,
            "communities": communities,
            "chunk_refs": chunk_refs,
        }

    def __len__(self) -> int:
        """实体总数"""
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM kg_entities"
                ).fetchone()
        return row[0] if row else 0

    @property
    def db_path(self) -> str:
        return self._db_path
