"""
情景记忆专用检索器：结构化过滤 + 语义检索 + 综合评分

检索流程：
  1. 结构化预过滤 — 按时间范围、重要性、类型缩小候选集
  2. TF-IDF 语义检索 — 余弦相似度，取 top (limit × 5) 扩大候选池
  3. 综合评分     — (语义 × 0.8 + 近因性 × 0.2) × 重要性加权
  4. 排序返回 Top-K

与工作记忆 HybridRetriever 的关键差异：
  - 情景记忆更重语义（0.8 vs 0.7），不依赖关键词匹配
  - 用"时间近因性"代替"时间衰减"：近期记忆加分，旧记忆不惩罚
  - 结构化预过滤先于语义检索，减少无效计算
"""
import math
from datetime import datetime, timedelta
from .base import TFIDFEngine, extract_text, parse_time


class EpisodicRetriever:
    """
    情景记忆检索器：语义 + 近因 + 重要性综合评分。

    用法:
        retriever = EpisodicRetriever()
        results = retriever.retrieve("小明喜欢什么", memories, limit=5)
        results = retriever.retrieve("火锅", memories, limit=5,
                                     min_importance=0.5, max_age_days=60)
    """

    def __init__(self, semantic_weight: float = 0.8, recency_weight: float = 0.2):
        self.semantic_weight = semantic_weight
        self.recency_weight = recency_weight
        self._tfidf = TFIDFEngine()

    # ═══════════════════════════════════════
    #  主检索入口
    # ═══════════════════════════════════════

    def retrieve(self, query: str, memories: list, limit: int = 5, **filters) -> list:
        """
        情景记忆混合检索。filters 支持 min_importance / max_age_days / memory_type。
        """
        if not memories:
            return []

        # ── 1. 结构化预过滤 ──
        candidates = self._filter(memories, **filters)
        if not candidates:
            return []

        # ── 2. 语义检索 → 扩大候选池 ──
        documents = [extract_text(m.get("content", "")) for m in candidates]
        tfidf_scores = self._tfidf.similarity(query, documents)
        hits = sorted(
            ((candidates[i], s) for i, s in tfidf_scores.items() if s > 0.001),
            key=lambda x: x[1], reverse=True,
        )[: limit * 5]

        # ── 3. 综合评分 ──
        scored = self._score_all(hits)

        # ── 4. 排序返回 Top-K ──
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    # ═══════════════════════════════════════
    #  评分
    # ═══════════════════════════════════════

    def _score_all(self, hits: list[tuple[dict, float]]) -> list[tuple[float, dict]]:
        """
        综合评分公式：(语义 × 0.8 + 近因性 × 0.2) × (0.8 + importance × 0.4)
        """
        results: list[tuple[float, dict]] = []
        for memory, vec_score in hits:
            recency = self._recency_score(memory.get("timestamp", ""))
            importance = memory.get("importance", 0.5)

            base = vec_score * self.semantic_weight + recency * self.recency_weight
            final = base * (0.8 + importance * 0.4)
            results.append((final, memory))
        return results

    # ═══════════════════════════════════════
    #  结构化预过滤
    # ═══════════════════════════════════════

    def _filter(self, memories: list, **filters) -> list:
        """
        逐条件过滤，缩小候选集。

        注意：当从 EpisodicMemory.search() 调用时，_search_rows() 已在 SQL 层完成
        结构化过滤（min_importance / memory_type / max_age_days），此处的 _filter
        作为保险层二次确认。外部直接调用 retrieve() 时，_filter 是主要过滤层。
        """
        results = memories

        t = filters.get("min_importance")
        if t is not None:
            results = [m for m in results if m.get("importance", 0) >= t]

        mt = filters.get("memory_type")
        if mt is not None:
            results = [m for m in results if m.get("type") == mt]

        days = filters.get("max_age_days")
        if days is not None:
            cutoff = datetime.now() - timedelta(days=days)
            results = [
                m for m in results
                if parse_time(m.get("timestamp", "")) >= cutoff
            ]

        return results

    # ═══════════════════════════════════════
    #  时间近因性
    # ═══════════════════════════════════════

    def _recency_score(self, timestamp: str) -> float:
        """
        近因性评分：1 / (1 + age_days / 30)
        当天 → 1.00，7天前 → 0.81，30天前 → 0.50，365天前 → 0.08
        不会衰减到零，旧信息仍有被检索的可能。
        """
        ts = parse_time(timestamp)
        age_days = (datetime.now() - ts).total_seconds() / 86400.0
        return 1.0 / (1.0 + age_days / 30.0)
