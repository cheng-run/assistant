"""
本地嵌入服务 + 向量存储

文本嵌入引擎：Ollama（OpenAI 兼容接口，本地 GPU 加速，默认 bge-m3 1024 维）
  - 单模型中英通用，替代原 fastembed 双模型（bge-small-zh 512d / MiniLM 384d）
图像嵌入引擎：fastembed CLIP（Qdrant/clip-ViT-B-32-vision，512 维，仍为 ONNX 本地）
存储引擎：SQLite（向量以 BLOB 存储，含嵌入缓存表）
检索算法：numpy 批量余弦相似度
"""
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from openai import OpenAI

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
#  Ollama 本地嵌入器（OpenAI 兼容接口，GPU 加速）
# ═══════════════════════════════════════════════════════════════

class OllamaEmbedder:
    """
    基于 Ollama OpenAI 兼容接口的本地文本嵌入器（单一模型，中英通用）。

    默认模型 bge-m3（1024 维，多语言检索）。复用现有 openai 客户端，
    无需新依赖。在 AMD GPU（Ollama ROCm）上运行，速度远快于 CPU。

    用法:
        embedder = OllamaEmbedder()
        vec = embedder.embed("你好世界")
        vecs, lang, name, dim = embedder.embed_with_model(["hello", "world"])
    """

    # 常见模型维度，命中即可避免首次探测的额外 HTTP 调用
    _KNOWN_DIMS = {"bge-m3": 1024}

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        timeout: float = 120.0,
        max_retries: int = 3,
        batch_size: int = 128,
    ):
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "bge-m3")
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434/v1"
        )
        self.timeout = timeout
        self.max_retries = max_retries
        self.batch_size = batch_size
        self._client = OpenAI(
            api_key="ollama",       # Ollama 忽略 key，占位即可
            base_url=self.base_url,
            timeout=self.timeout,
        )
        self._dim: Optional[int] = self._KNOWN_DIMS.get(self.model)

    # ── 元信息 ──────────────────────────

    @property
    def model_name(self) -> str:
        """模型名（与写入 DB 的 model_name 保持一致）"""
        return self.model

    @property
    def dim(self) -> int:
        """向量维度。已知模型直接返回；未知模型首次探测一次并缓存"""
        if self._dim is None:
            v = self.embed("dimension probe")
            self._dim = len(v)
        return self._dim

    # ── 嵌入 ────────────────────────────

    def embed(self, text: str, lang: str = None) -> List[float]:
        """单条嵌入，返回 Python list[float]"""
        result = self.embed_batch([text], lang=lang)
        return result[0]

    def embed_batch(
        self, texts: List[str], lang: str = None
    ) -> List[List[float]]:
        """
        批量嵌入（分块调用 Ollama /v1/embeddings，带指数退避重试）。

        参数:
            texts: 文本列表
            lang:  语言参数（接口兼容，单一模型下忽略）
        返回:
            向量列表 List[List[float]]
        """
        if not texts:
            return []

        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            vectors.extend(self._embed_chunk(batch))

        # 首次嵌入时缓存维度
        if self._dim is None and vectors:
            self._dim = len(vectors[0])

        return vectors

    def _embed_chunk(self, texts: List[str]) -> List[List[float]]:
        """单块嵌入，带重试与友好错误提示"""
        last_error = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=texts,
                )
                # 按 index 排序（防御：Ollama 默认按输入顺序返回）
                data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
                return [list(d.embedding) for d in data]
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避：1s → 2s → 4s

        raise RuntimeError(
            f"无法连接 Ollama 嵌入服务（{self.base_url}）。"
            f"请确认已运行 'ollama serve' 且已拉取模型 'ollama pull {self.model}'。"
        ) from last_error

    def embed_with_model(
        self, texts: List[str], lang: str = None
    ) -> tuple:
        """
        嵌入并返回 (vectors, lang, model_name, dim)。
        用于 VectorStore 记录模型信息。
        """
        if lang is None:
            lang = detect_lang_batch(texts)  # 接口兼容，单一模型忽略
        vectors = self.embed_batch(texts, lang=lang)
        dim = len(vectors[0]) if vectors else (self._dim or 0)
        return (vectors, lang, self.model, dim)


# ═══════════════════════════════════════════════════════════════
#  本地图像嵌入器（多模态检索用）
# ═══════════════════════════════════════════════════════════════

class ImageEmbedder:
    """
    基于 fastembed 的本地图像嵌入器 (ONNX, CPU/AMD 友好)。

    模型: Qdrant/clip-ViT-B-32-vision (512d)
    备选: Qdrant/Unicom-ViT-B-16 (768d, 精度更高)

    用法:
        ie = ImageEmbedder()
        vec = ie.embed_image("page_0.png")
        vecs = ie.embed_batch(["p1.png", "p2.png"])
    """

    SUPPORTED_MODELS = [
        "Qdrant/clip-ViT-B-32-vision",     # 512d, 轻量
        "Qdrant/Unicom-ViT-B-16",          # 768d, 精度更高
        "jinaai/jina-clip-v1",             # 768d, 最新
    ]

    def __init__(self, model_name: str = "Qdrant/clip-ViT-B-32-vision"):
        self.model_name = model_name
        self._model = None  # 懒加载

    @property
    def model(self):
        if self._model is None:
            from fastembed import ImageEmbedding
            self._model = ImageEmbedding(model_name=self.model_name)
        return self._model

    @property
    def dim(self) -> int:
        """返回当前模型的向量维度"""
        dims = {
            "Qdrant/clip-ViT-B-32-vision": 512,
            "Qdrant/Unicom-ViT-B-16": 768,
            "jinaai/jina-clip-v1": 768,
        }
        return dims.get(self.model_name, 512)

    def embed_image(self, image_path: str) -> List[float]:
        """单张图片嵌入"""
        result = self.embed_batch([image_path])
        return result[0]

    def embed_batch(self, paths: List[str]) -> List[List[float]]:
        """批量图片嵌入"""
        if not paths:
            return []
        embeddings = list(self.model.embed(paths))
        return [vec.tolist() for vec in embeddings]


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
                    modality        TEXT    DEFAULT 'text',
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

            # 迁移: 为旧表添加 modality 列（如果不存在）
            try:
                conn.execute(
                    "ALTER TABLE rag_chunks ADD COLUMN modality TEXT DEFAULT 'text'"
                )
            except Exception:
                pass  # 列已存在

            # 迁移: 为旧表添加 content_hash 列（用于嵌入缓存去重）
            try:
                conn.execute(
                    "ALTER TABLE rag_chunks ADD COLUMN content_hash TEXT DEFAULT ''"
                )
            except Exception:
                pass  # 列已存在

            # 迁移: checkpoint 旧格式 JSON embedding → 新格式 BLOB
            # （后台自动处理，增量转换）
            try:
                conn.execute(
                    "ALTER TABLE rag_chunks ADD COLUMN embedding_blob BLOB DEFAULT NULL"
                )
            except Exception:
                pass

            # 迁移: 去掉 embedding 列的 NOT NULL 约束
            # （向量现存 embedding_blob，embedding 列仅用于旧格式兼容，需允许 NULL）
            try:
                # SQLite 不能直接改列约束，重建表
                conn.execute("PRAGMA foreign_keys=OFF")
                conn.executescript("""
                    BEGIN;
                    CREATE TABLE rag_chunks_new (
                        id              INTEGER PRIMARY KEY AUTOINCREMENT,
                        content         TEXT    NOT NULL,
                        heading_path    TEXT    DEFAULT '[]',
                        embedding       TEXT    DEFAULT NULL,
                        model_name      TEXT    DEFAULT '',
                        model_dim       INTEGER DEFAULT 0,
                        token_count     INTEGER DEFAULT 0,
                        source_file     TEXT    DEFAULT '',
                        chunk_index     INTEGER DEFAULT 0,
                        modality        TEXT    DEFAULT 'text',
                        created_at      TEXT    DEFAULT (datetime('now')),
                        content_hash    TEXT    DEFAULT '',
                        embedding_blob  BLOB    DEFAULT NULL
                    );
                    INSERT INTO rag_chunks_new (id, content, heading_path, embedding,
                        model_name, model_dim, token_count, source_file, chunk_index,
                        modality, created_at, content_hash, embedding_blob)
                    SELECT id, content, heading_path, embedding, model_name, model_dim,
                        token_count, source_file, chunk_index, modality, created_at,
                        content_hash, embedding_blob
                    FROM rag_chunks;
                    DROP TABLE rag_chunks;
                    ALTER TABLE rag_chunks_new RENAME TO rag_chunks;
                    CREATE INDEX IF NOT EXISTS idx_rag_source ON rag_chunks(source_file);
                    CREATE INDEX IF NOT EXISTS idx_rag_model ON rag_chunks(model_name);
                    CREATE INDEX IF NOT EXISTS idx_rag_model_dim ON rag_chunks(model_dim);
                    COMMIT;
                """)
            except Exception:
                pass  # 已重建或无需迁移

            # ── 嵌入缓存表 ──
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT PRIMARY KEY,
                    embedding    BLOB NOT NULL,
                    model_name   TEXT DEFAULT '',
                    model_dim    INTEGER DEFAULT 0,
                    created_at   TEXT DEFAULT (datetime('now'))
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_emb_cache_model "
                "ON embedding_cache(model_name, model_dim)"
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

    @staticmethod
    def _content_hash(text: str) -> str:
        """计算内容 SHA256 哈希（用于嵌入缓存去重）"""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]

    def add(
        self, chunks: List[Dict], source_file: str = "", lang: str = None,
        progress_callback=None, progress_offset: int = 0,
    ) -> int:
        """
        批量嵌入并存储（带嵌入缓存 + BLOB 存储）。

        参数:
            chunks:            process() 输出
            source_file:      来源文件名
            lang:             语言，None 则自动检测
            progress_callback: callable(completed, total) — 进度回调
            progress_offset:   进度起始偏移量（用于与其他阶段合并进度）
        返回:
            存储条数
        """
        if not chunks:
            return 0

        # 过滤近空 chunk（如图表轴标签碎片 "35,000"），避免检索噪音。
        # 有效内容 = 去掉 [Context:]/[上文:] 前缀后的正文长度。
        MIN_CONTENT_CHARS = 15

        def _meaningful_len(text: str) -> int:
            t = text or ""
            if t.startswith("[Context:"):
                end = t.find("]")
                if end >= 0:
                    t = t[end + 1:].lstrip("\n")
            idx = t.rfind("[上文:")
            if idx >= 0:
                end = t.find("]", idx)
                if end >= 0:
                    t = t[end + 1:].lstrip("\n")
            return len(t.strip())

        chunks = [
            c for c in chunks
            if _meaningful_len(c.get("content", "")) >= MIN_CONTENT_CHARS
        ]
        if not chunks:
            return 0

        # ── Step 1: 计算 content_hash ──
        for c in chunks:
            c["content_hash"] = self._content_hash(c.get("content", ""))

        # ── Step 1.5: 分离预置向量（CLIP 页面向量等，跳过重新嵌入） ──
        # 带非空 embedding 的 chunk 已由调用方预先嵌入（如 CLIP 图像向量），
        # 其 model_name/model_dim 也由调用方标注，直接转 BLOB 存储。
        preset_vectors = {}
        text_indices = []
        for i, c in enumerate(chunks):
            if c.get("embedding") is not None:
                vec = np.asarray(c["embedding"], dtype=np.float32)
                preset_vectors[i] = (
                    vec.tobytes(),
                    c.get("model_name") or getattr(self.embedder, "model_name", ""),
                    c.get("model_dim") or getattr(self.embedder, "dim", 0),
                )
            else:
                text_indices.append(i)

        # ── Step 2: 检查嵌入缓存（模型作用域） ──
        cache_hits = {}
        uncached_indices = []
        cache_model = getattr(self.embedder, "model_name", None)
        cache_dim = getattr(self.embedder, "dim", None)
        has_model_scope = cache_model is not None and cache_dim is not None
        with self._lock:
            with self._conn() as conn:
                for i in text_indices:
                    c = chunks[i]
                    args = [c["content_hash"]]
                    sql = (
                        "SELECT embedding, model_name, model_dim FROM embedding_cache "
                        "WHERE content_hash = ?"
                    )
                    if has_model_scope:
                        sql += " AND model_name = ? AND model_dim = ?"
                        args += [cache_model, cache_dim]
                    row = conn.execute(sql, tuple(args)).fetchone()
                    if row:
                        cache_hits[i] = (row["embedding"], row["model_name"], row["model_dim"])
                    else:
                        uncached_indices.append(i)

        # ── Step 3: 批量嵌入未缓存的 chunks ──
        new_vectors = {}
        if uncached_indices:
            uncached_texts = [chunks[i]["content"] for i in uncached_indices]
            vectors, detected_lang, model_name, model_dim = self.embedder.embed_with_model(
                uncached_texts, lang=lang
            )
            for idx, i in enumerate(uncached_indices):
                blob = np.array(vectors[idx], dtype=np.float32).tobytes()
                new_vectors[i] = (blob, model_name, model_dim)

            # 写入缓存（先清掉同内容旧模型条目，防止内容哈希撞键静默跳过缓存）
            with self._lock:
                with self._conn() as conn:
                    conn.executemany(
                        "DELETE FROM embedding_cache WHERE content_hash = ? "
                        "AND NOT (model_name = ? AND model_dim = ?)",
                        [
                            (chunks[i]["content_hash"], model_name, model_dim)
                            for i in uncached_indices
                        ],
                    )
                    conn.executemany(
                        "INSERT OR IGNORE INTO embedding_cache "
                        "(content_hash, embedding, model_name, model_dim) "
                        "VALUES (?, ?, ?, ?)",
                        [
                            (chunks[i]["content_hash"], blob, model_name, model_dim)
                            for i in uncached_indices
                        ],
                    )
                    conn.commit()

            lang = detected_lang

        # ── Step 4: 合并预置向量 + 缓存命中 + 新嵌入（统一原始索引空间） ──
        all_vectors = dict(preset_vectors)
        for i in text_indices:
            if i in cache_hits:
                all_vectors[i] = cache_hits[i]
            elif i in new_vectors:
                all_vectors[i] = new_vectors[i]

        if not all_vectors:
            return 0

        # ── Step 5: 写入数据库（BLOB 格式，每行用各自的模型信息） ──
        import sqlite3
        with self._lock:
            with self._conn() as conn:
                rows = [
                    (
                        chunks[i]["content"],
                        json.dumps(chunks[i].get("heading_path", [])),
                        chunks[i]["content_hash"],
                        embedding_blob,
                        model_name,
                        model_dim,
                        chunks[i].get("token_count", 0),
                        source_file or chunks[i].get("source_file", ""),
                        chunks[i].get("chunk_index", 0),
                        chunks[i].get("modality", "text"),
                    )
                    for i, (embedding_blob, model_name, model_dim) in all_vectors.items()
                ]
                conn.executemany(
                    """INSERT INTO rag_chunks
                       (content, heading_path, content_hash, embedding_blob,
                        model_name, model_dim, token_count, source_file,
                        chunk_index, modality)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    rows,
                )
                conn.commit()

        if progress_callback:
            progress_callback(progress_offset + len(chunks), progress_offset + len(chunks))

        return len(chunks)

    # ── 检索 ────────────────────────────

    def search(
        self, query: str, top_k: int = 5, min_score: float = 0.0,
        modality_filter: str = None, image_embedder=None,
    ) -> List[Dict]:
        """
        余弦相似度检索。

        参数:
            query:           查询文本
            top_k:           返回数量
            min_score:       最低相似度阈值
            modality_filter: 只返回该模态的结果 ('text' / 'page_image' / ...)
                            None 表示不过滤
            image_embedder:  传入 ImageEmbedder 实例时，用 CLIP 嵌入查询
                            （用于视觉检索路径）

        返回:
            [{"id": ..., "content": ..., "score": ..., ...}, ...]
        """
        if not query:
            return []

        # ── 查询嵌入 ──
        if image_embedder is not None:
            # 视觉路径：用 CLIP 把文本查询映射到图像空间
            query_vec_np = image_embedder.model.embed([query])
            query_arr = np.array(list(query_vec_np)[0], dtype=np.float32)
            q_norm = float(np.linalg.norm(query_arr))
            if q_norm == 0:
                return []
            # 视觉路径不需要 model_name/model_dim 对齐
            model_name = image_embedder.model_name
            query_dim = image_embedder.dim
        else:
            # 文本路径：原有逻辑
            query_vec, q_lang, model_name, query_dim = self.embedder.embed_with_model(
                [query]
            )
            query_arr = np.array(query_vec[0], dtype=np.float32)
            q_norm = float(np.linalg.norm(query_arr))
            if q_norm == 0:
                return []

        # ── 加载同维度向量（文本路径按查询维度过滤；视觉路径用 CLIP 512d） ──
        with self._lock:
            with self._conn() as conn:
                # 兼容旧格式：优先使用 BLOB，fallback 到 JSON
                rows = conn.execute(
                    "SELECT id, content, heading_path, embedding_blob, embedding, "
                    "model_name, model_dim, token_count, source_file, chunk_index, "
                    "modality FROM rag_chunks WHERE model_dim = ? ORDER BY id",
                    (query_dim,),
                ).fetchall()

        if not rows:
            return []

        # ── 一次 gather：向量 → 矩阵，元数据 → 并行列表（O(n) 无数学运算） ──
        n = len(rows)
        matrix = np.empty((n, query_dim), dtype=np.float32)
        row_ids = np.empty(n, dtype=np.int64)
        model_names = []
        modalities = []
        valid = np.ones(n, dtype=bool)

        for idx, row in enumerate(rows):
            try:
                if row["embedding_blob"] is not None:
                    stored = np.frombuffer(row["embedding_blob"], dtype=np.float32)
                elif row["embedding"]:
                    stored = np.array(json.loads(row["embedding"]), dtype=np.float32)
                else:
                    valid[idx] = False
                    continue
            except (json.JSONDecodeError, TypeError, ValueError):
                valid[idx] = False
                continue

            if stored.shape[0] != query_dim or stored.shape[0] == 0:
                valid[idx] = False
                continue

            matrix[idx] = stored
            row_ids[idx] = row["id"]
            model_names.append(row["model_name"])
            modalities.append(row["modality"] if "modality" in row.keys() else "text")

        # ── 批量余弦相似度 ──
        norms = np.linalg.norm(matrix, axis=1)
        valid &= norms > 0
        sims = (matrix @ query_arr) / (q_norm * norms)

        # 同模型微小加分
        same_model = np.array([mn == model_name for mn in model_names])
        adjusted = sims * np.where(same_model, 1.01, 0.99)

        # 模态过滤（page_image / text / ...）
        if modality_filter is not None:
            mod_arr = np.array(modalities)
            valid &= mod_arr == modality_filter

        # 无效项置为最低分（过滤掉零向量/坏行/模态不符）
        adjusted[~valid] = -np.inf

        # ── 稳定排序取 top_k（按分数降序，同分按 id 升序，与旧逻辑一致） ──
        candidates = np.nonzero(adjusted >= min_score)[0]
        if candidates.size == 0:
            return []
        order = candidates[np.lexsort((row_ids[candidates], -adjusted[candidates]))]

        scored: List[Dict] = []
        for idx in order[:top_k]:
            row = rows[idx]
            scored.append({
                "id": row["id"],
                "content": row["content"],
                "heading_path": json.loads(row["heading_path"]),
                "score": round(float(adjusted[idx]), 4),
                "model_name": row["model_name"],
                "token_count": row["token_count"],
                "source_file": row["source_file"],
                "chunk_index": row["chunk_index"],
                "modality": row["modality"] if "modality" in row.keys() else "text",
            })

        return scored

    # ── 维护 ────────────────────────────

    def clear(self):
        with self._lock:
            with self._conn() as conn:
                conn.execute("DELETE FROM rag_chunks")
                conn.execute("DELETE FROM embedding_cache")
                conn.commit()
                # 回收碎片空间
                conn.execute("VACUUM")

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
