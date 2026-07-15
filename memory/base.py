"""
共享基础工具：供 WorkingMemory 和 EpisodicMemory 的检索器共用

包含：
  - CJK 感知分词器
  - 文本提取 / 时间解析
  - 向量数学（点积、L2 范数、余弦相似度）
  - TF-IDF 计算引擎（纯 Python，无外部依赖）
"""
import math
from collections import Counter
from datetime import datetime


# ═══════════════════════════════════════════════════════════════
#  字符检测
# ═══════════════════════════════════════════════════════════════

def is_cjk(ch: str) -> bool:
    """判断字符是否为 CJK（中日韩统一表意文字）"""
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF   # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF  # CJK Unified Ideographs Extension A
            or 0xF900 <= cp <= 0xFAFF  # CJK Compatibility Ideographs
            )


# ═══════════════════════════════════════════════════════════════
#  分词
# ═══════════════════════════════════════════════════════════════

def tokenize(text: str) -> list[str]:
    """
    CJK 感知分词：
    - CJK 字符：每个字独立为 token
    - 英文单词：按字母/数字聚合，统一小写
    - 标点/空白：分隔符，不产出 token
    """
    if not text:
        return []

    tokens: list[str] = []
    buf: list[str] = []

    for ch in text:
        if is_cjk(ch):
            if buf:
                tokens.append("".join(buf).lower())
                buf.clear()
            tokens.append(ch)
        elif ch.isalnum():
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf).lower())
                buf.clear()

    if buf:
        tokens.append("".join(buf).lower())

    # 过滤空白 token
    return [t for t in tokens if t.strip()]


# ═══════════════════════════════════════════════════════════════
#  文本提取 / 时间解析
# ═══════════════════════════════════════════════════════════════

def extract_text(content) -> str:
    """从 memory content（str 或 dict）提取纯文本"""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return content.get("content", str(content))
    return str(content)


def parse_time(timestamp: str) -> datetime:
    """解析 ISO 时间戳，解析失败返回 datetime.min"""
    try:
        return datetime.fromisoformat(timestamp)
    except (ValueError, TypeError):
        return datetime.min


# ═══════════════════════════════════════════════════════════════
#  向量数学
# ═══════════════════════════════════════════════════════════════

def dot(v1: dict[str, float], v2: dict[str, float]) -> float:
    """稀疏向量点积"""
    # 在较小的 dict 上迭代以减少比较次数
    if len(v1) > len(v2):
        v1, v2 = v2, v1
    return sum(v1.get(k, 0) * v2[k] for k in v1 if k in v2)


def l2_norm(v: dict[str, float]) -> float:
    """L2 范数"""
    return math.sqrt(sum(val ** 2 for val in v.values()))


def cosine_similarity(
    v1: dict[str, float], v2: dict[str, float]
) -> float:
    """稀疏向量的余弦相似度"""
    n1 = l2_norm(v1)
    n2 = l2_norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dot(v1, v2) / (n1 * n2)


# ═══════════════════════════════════════════════════════════════
#  TF-IDF 引擎
# ═══════════════════════════════════════════════════════════════

class TFIDFEngine:
    """
    自包含 TF-IDF 引擎，无外部依赖。

    用法:
        engine = TFIDFEngine()
        scores = engine.similarity("查询", ["文档1", "文档2"])
        # → {0: 0.85, 1: 0.12}
    """

    def __init__(self, smooth_idf: bool = True):
        self.smooth_idf = smooth_idf

    def similarity(
        self, query: str, documents: list[str]
    ) -> dict[int, float]:
        """
        计算 query 与每个 document 的 TF-IDF 余弦相似度。

        返回 {doc_index: similarity_score}，仅包含 > 0 的结果。
        """
        if not documents:
            return {}

        query_tokens = tokenize(query)
        doc_tokens_list = [tokenize(d) for d in documents]

        # ── TF ──
        query_tf = self._tf(query_tokens)
        doc_tfs = [self._tf(dt) for dt in doc_tokens_list]

        # ── IDF（懒加载缓存） ──
        N = len(documents)
        idf_cache: dict[str, float] = {}

        def get_idf(term: str) -> float:
            if term not in idf_cache:
                df = sum(1 for tf in doc_tfs if term in tf)
                idf_cache[term] = math.log(
                    (N + 1) / (df + 1)
                ) + (1.0 if self.smooth_idf else 0.0)
            return idf_cache[term]

        # ── TF-IDF 向量 ──
        def to_tfidf(tf: dict[str, float]) -> dict[str, float]:
            return {t: tf_val * get_idf(t) for t, tf_val in tf.items()}

        query_vec = to_tfidf(query_tf)
        doc_vecs = [to_tfidf(tf) for tf in doc_tfs]

        # ── 余弦相似度 ──
        scores: dict[int, float] = {}
        for i, dv in enumerate(doc_vecs):
            sim = cosine_similarity(query_vec, dv)
            if sim > 0.001:
                scores[i] = sim

        return scores

    @staticmethod
    def _tf(tokens: list[str]) -> dict[str, float]:
        """词频：term_count / total_tokens"""
        if not tokens:
            return {}
        total = len(tokens)
        return {term: count / total for term, count in Counter(tokens).items()}
