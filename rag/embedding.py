"""
本地嵌入服务 + 向量存储

嵌入引擎：fastembed（ONNX Runtime，无 PyTorch 依赖，AMD 友好）
中文模型：BAAI/bge-small-zh-v1.5（512 维）
英文模型：all-MiniLM-L6-v2（384 维）
存储引擎：SQLite（向量以 JSON 字符串存储）
检索算法：numpy 余弦相似度
"""
import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from memory.base import is_cjk

# ── 默认存储路径 ──────────────────────
_DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "data_db"
_DEFAULT_DB_PATH = str(_DEFAULT_DB_DIR / "rag_vectors.db")

# ── 模型配置 ──────────────────────────
_MODEL_CONFIG = {
    "zh": {
        "name": "BAAI/bge-small-zh-v1.5",
        "dim": 512,
        "desc": "中文嵌入",
    },
    "en": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dim": 384,
        "desc": "英文嵌入",
    },
}


# ═══════════════════════════════════════════════════════════════
#  语言检测
# ═══════════════════════════════════════════════════════════════

def detect_lang(text: str) -> str:
    """
    基于 CJK 字符占比检测语言。
    ratio > 0.3 → "zh"，否则 "en"
    """
    if not text:
        return "en"
    cjk = sum(1 for ch in text if is_cjk(ch))
    ratio = cjk / max(len(text), 1)
    return "zh" if ratio > 0.3 else "en"


def detect_lang_batch(texts: List[str]) -> str:
    """
    批量检测：采样前 200 条拼接，按总体 CJK 占比判断。
    避免多数投票被大量短英文片段（代码、符号）拉偏。
    """
    if not texts:
        return "en"
    # 采样部分文本拼接后统一判断 CJK 占比
    sample = "".join(texts[: min(200, len(texts))])
    return detect_lang(sample)


# ═══════════════════════════════════════════════════════════════
#  本地嵌入器
# ═══════════════════════════════════════════════════════════════

class LocalEmbedder:
    """
    本地双模型嵌入器（fastembed 引擎）。

    用法:
        embedder = LocalEmbedder()
        vec = embedder.embed("你好世界")       # 自动选中文模型
        vecs = embedder.embed_batch(["hello", "world"])  # 自动选英文模型
    """

    def __init__(self):
        self._models: dict = {}  # lang -> TextEmbedding (懒加载)

    # ── 模型懒加载 ──────────────────────

    def _get_model(self, lang: str):
        """懒加载指定语言的嵌入模型"""
        if lang not in self._models:
            from fastembed import TextEmbedding
            config = _MODEL_CONFIG[lang]
            self._models[lang] = TextEmbedding(model_name=config["name"])
        return self._models[lang]

    @property
    def zh_model(self):
        return self._get_model("zh")

    @property
    def en_model(self):
        return self._get_model("en")

    # ── 嵌入 ────────────────────────────

    def embed(self, text: str, lang: str = None) -> List[float]:
        """单条嵌入，返回 Python list[float]"""
        result = self.embed_batch([text], lang=lang)
        return result[0]

    def embed_batch(
        self, texts: List[str], lang: str = None
    ) -> List[List[float]]:
        """
        批量嵌入。

        参数:
            texts: 文本列表
            lang:  语言 "zh"/"en"，None 则自动检测
        返回:
            向量列表 List[List[float]]
        """
        if not texts:
            return []

        if lang is None:
            lang = detect_lang_batch(texts)

        model = self._get_model(lang)
        embeddings = list(model.embed(texts))
        return [vec.tolist() for vec in embeddings]

    def embed_with_model(
        self, texts: List[str], lang: str = None
    ) -> tuple:
        """
        嵌入并返回 (vectors, lang, model_name, dim)。
        用于 VectorStore 记录模型信息。
        """
        if lang is None:
            lang = detect_lang_batch(texts)

        config = _MODEL_CONFIG[lang]
        model = self._get_model(lang)
        embeddings = list(model.embed(texts))

        return (
            [vec.tolist() for vec in embeddings],
            lang,
            config["name"],
            config["dim"],
        )


# ═══════════════════════════════════════════════════════════════
#  向量存储（SQLite）
# ═══════════════════════════════════════════════════════════════

class VectorStore:
    """
    SQLite 向量存储 — 嵌入持久化 + 余弦相似度检索。

    用法:
        embedder = LocalEmbedder()
        store = VectorStore(embedder)
        store.add(chunks, source_file="report.md")
        results = store.search("query", top_k=5)
    """

    def __init__(self, embedder: LocalEmbedder, db_path: str = None):
        self.embedder = embedder
        self._db_path = db_path or _DEFAULT_DB_PATH
        self._lock = threading.Lock()
        self._conn_persist = None  # 持久连接，_init_db() 中初始化
        self._init_db()

    # ── 数据库初始化 ────────────────────

    def _init_db(self):
        import sqlite3
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)

        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    content         TEXT    NOT NULL,
                    heading_path    TEXT    DEFAULT '[]',
                    embedding       TEXT    NOT NULL,
                    model_name      TEXT    DEFAULT '',
                    model_dim       INTEGER DEFAULT 0,
                    token_count     INTEGER DEFAULT 0,
                    source_file     TEXT    DEFAULT '',
                    chunk_index     INTEGER DEFAULT 0,
                    created_at      TEXT    DEFAULT (datetime('now'))
                )
            """)

            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_source "
                "ON rag_chunks(source_file)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_model "
                "ON rag_chunks(model_name)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_rag_model_dim "
                "ON rag_chunks(model_dim)"
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

    # ── 写入 ────────────────────────────

    def add(
        self, chunks: List[Dict], source_file: str = "", lang: str = None
    ) -> int:
        """
        批量嵌入并存储。

        参数:
            chunks:      process() 输出
            source_file: 来源文件名
            lang:        语言，None 则自动检测
        返回:
            存储条数
        """
        if not chunks:
            return 0

        # ── 提取文本 ──
        texts = [c["content"] for c in chunks]

        # ── 批量嵌入（带回模型信息） ──
        vectors, lang, model_name, model_dim = self.embedder.embed_with_model(
            texts, lang=lang
        )

        # ── 写入 ──
        import sqlite3
        with self._lock:
            with self._conn() as conn:
                rows = [
                    (
                        chunks[i]["content"],
                        json.dumps(chunks[i].get("heading_path", [])),
                        json.dumps(vectors[i]),
                        model_name,
                        model_dim,
                        chunks[i].get("token_count", 0),
                        source_file or chunks[i].get("source_file", ""),
                        chunks[i].get("chunk_index", 0),
                    )
                    for i in range(len(chunks))
                ]
                conn.executemany(
                    """INSERT INTO rag_chunks
                       (content, heading_path, embedding, model_name, model_dim,
                        token_count, source_file, chunk_index)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()

        return len(chunks)

    # ── 检索 ────────────────────────────

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Dict]:
        """
        余弦相似度检索。

        1. 检测 query 语言 → 选模型 → embed
        2. 只检索同维度的向量（跳过维度不匹配的，避免 crash）
        3. 若无同维度向量，用存量最多的模型重新 embed 查询
        """
        if not query:
            return []

        # ── 查询嵌入 ──
        query_vec, q_lang, model_name, query_dim = self.embedder.embed_with_model(
            [query]
        )
        query_arr = np.array(query_vec[0], dtype=np.float32)
        q_norm = float(np.linalg.norm(query_arr))
        if q_norm == 0:
            return []

        # ── 加载所有向量 ──
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM rag_chunks ORDER BY id"
                ).fetchall()

        # ── 按维度分组 ──
        same_dim_rows = [r for r in rows if r["model_dim"] == query_dim]
        other_rows = [r for r in rows if r["model_dim"] != query_dim]

        # 若无同维度向量，改用存量最多的模型重新 embed
        if not same_dim_rows and other_rows:
            from collections import Counter
            dim_counts = Counter(r["model_dim"] for r in other_rows)
            best_dim = dim_counts.most_common(1)[0][0]
            # 找到该维度的模型名
            fallback_lang = None
            for lang, cfg in _MODEL_CONFIG.items():
                if cfg["dim"] == best_dim:
                    fallback_lang = lang
                    break
            if fallback_lang:
                query_vec2, _, model_name, query_dim2 = self.embedder.embed_with_model(
                    [query], lang=fallback_lang
                )
                query_arr = np.array(query_vec2[0], dtype=np.float32)
                q_norm = float(np.linalg.norm(query_arr))
                if q_norm == 0:
                    return []
                same_dim_rows = [r for r in rows if r["model_dim"] == query_dim2]
                other_rows = [r for r in rows if r["model_dim"] != query_dim2]

        # ── 计算余弦相似度（仅同维度） ──
        scored: List[Dict] = []
        for row in same_dim_rows:
            stored = np.array(json.loads(row["embedding"]), dtype=np.float32)
            s_norm = float(np.linalg.norm(stored))
            if s_norm == 0:
                continue

            sim = float(np.dot(query_arr, stored) / (q_norm * s_norm))
            # 同模型微小加分（优先同模型的向量，但不显著扭曲余弦相似度排序）
            same_model = row["model_name"] == model_name
            adjusted = sim * (1.01 if same_model else 0.99)

            if adjusted >= min_score:
                scored.append({
                    "id": row["id"],
                    "content": row["content"],
                    "heading_path": json.loads(row["heading_path"]),
                    "score": round(adjusted, 4),
                    "model_name": row["model_name"],
                    "token_count": row["token_count"],
                    "source_file": row["source_file"],
                    "chunk_index": row["chunk_index"],
                })

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    # ── 维护 ────────────────────────────

    def clear(self):
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM rag_chunks")
                conn.commit()

    def clear_by_source(self, source_file: str):
        with self._lock:
            with self._conn() as conn:
                conn.execute(
                    "DELETE FROM rag_chunks WHERE source_file = ?",
                    (source_file,),
                )
                conn.commit()

    def list_sources(self) -> List[str]:
        with self._lock:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT source_file FROM rag_chunks "
                    "WHERE source_file != ''"
                ).fetchall()
        return [r["source_file"] for r in rows]

    def stats(self) -> dict:
        with self._lock:
            with self._conn() as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM rag_chunks"
                ).fetchone()[0]
                models = conn.execute(
                    "SELECT model_name, COUNT(*) as cnt "
                    "FROM rag_chunks GROUP BY model_name"
                ).fetchall()
        return {
            "total": total,
            "by_model": {r["model_name"]: r["cnt"] for r in models},
        }

    def __len__(self) -> int:
        with self._lock:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM rag_chunks"
                ).fetchone()
        return row[0] if row else 0

    @property
    def db_path(self) -> str:
        return self._db_path
