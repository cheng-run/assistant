"""
混合检索器：TF-IDF 向量化 + 关键词匹配 + 时间衰减 + 重要性加权

检索流程：
  1. TF-IDF 向量化 → 余弦相似度（权重 0.7）
  2. 关键词匹配   → Jaccard 重叠度（权重 0.3）
  3. 时间衰减     → 指数衰减，越旧越低
  4. 重要性加权   → 重要性越高分越高
  5. 综合评分排序 → 返回 Top-K
"""
import math
from datetime import datetime
from .base import TFIDFEngine, tokenize, extract_text, parse_time


class HybridRetriever:
    """
    混合检索器：工作记忆专用。

    用法:
        retriever = HybridRetriever()
        results = retriever.retrieve("小明喜欢什么", memories, limit=5)
    """

    def __init__(
        self,
        tfidf_weight: float = 0.7,
        keyword_weight: float = 0.3,
        time_half_life_hours: float = 24.0,
    ):
        self.tfidf_weight = tfidf_weight
        self.keyword_weight = keyword_weight
        self.time_half_life_hours = time_half_life_hours
        self._tfidf = TFIDFEngine()

    # ═══════════════════════════════════════
    #  主检索入口
    # ═══════════════════════════════════════

    def retrieve(self, query: str, memories: list, limit: int = 5) -> list:
        """
        混合检索主入口。

        参数:
            query:    查询文本
            memories: 记忆条目列表
            limit:    返回数量上限

        返回:
            按综合评分降序排列的记忆列表（只读，不修改原数据）
        """
        if not memories:
            return []

        documents = [extract_text(m.get("content", "")) for m in memories]

        # ── TF-IDF 向量相似度 ──
        vector_scores = self._tfidf.similarity(query, documents)

        # ── 综合评分 ──
        scored = self._score_all(memories, vector_scores, query)

        # ── 排序返回 Top-K ──
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:limit]]

    # ═══════════════════════════════════════
    #  评分
    # ═══════════════════════════════════════

    def _score_all(
        self,
        memories: list,
        vector_scores: dict[int, float],
        query: str,
    ) -> list[tuple[float, dict]]:
        """为所有记忆计算综合评分，返回 [(score, memory), ...]"""
        results: list[tuple[float, dict]] = []
        for i, memory in enumerate(memories):
            relevance = vector_scores.get(i, 0.0)
            keyword = self._keyword_score(query, memory)

            if relevance > 0:
                base = relevance * self.tfidf_weight + keyword * self.keyword_weight
            else:
                base = keyword

            if base <= 0:
                continue

            decay = self._time_decay(memory.get("timestamp", ""))
            imp_weight = 0.8 + memory.get("importance", 0.5) * 0.4
            final = base * decay * imp_weight
            results.append((final, memory))
        return results

    # ═══════════════════════════════════════
    #  关键词匹配
    # ═══════════════════════════════════════

    def _keyword_score(self, query: str, memory: dict) -> float:
        """
        Token 重叠 Jaccard 系数 + 精确子串奖励。
        返回 0.0 ~ 1.0
        """
        text = extract_text(memory.get("content", ""))
        q_tokens = set(tokenize(query))
        t_tokens = set(tokenize(text))

        if not q_tokens:
            return 0.0

        exact = 0.8 if query.lower() in text.lower() else 0.0
        union = q_tokens | t_tokens
        jaccard = len(q_tokens & t_tokens) / len(union) if union else 0.0
        return max(jaccard, exact)

    # ═══════════════════════════════════════
    #  时间衰减
    # ═══════════════════════════════════════

    def _time_decay(self, timestamp: str) -> float:
        """
        指数衰减：decay = e^(-age_hours / half_life_hours)
        半衰期默认 24h：1h → ~96%，12h → ~61%，24h → ~37%
        """
        ts = parse_time(timestamp)
        age_hours = (datetime.now() - ts).total_seconds() / 3600.0
        return math.exp(-age_hours / self.time_half_life_hours)
