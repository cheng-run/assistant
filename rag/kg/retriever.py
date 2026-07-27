"""
知识图谱检索器 — 基于社区摘要匹配的图检索 + 向量/图谱融合

检索流程:
  1. 嵌入用户查询
  2. 与所有社区摘要嵌入做余弦相似度匹配
  3. 展开匹配的社区 → 实体详情 + 关系
  4. 格式化输出为 LLM 可用的上下文
"""
import json
from typing import Dict, List, Optional, Tuple

import numpy as np


KgRetrievalResult = Dict
# {"community_id": int, "score": float, "summary": str,
#  "entities": [...], "relations": [...]}


class KnowledgeGraphRetriever:
    """
    基于社区摘要嵌入的图检索器。

    用法:
        retriever = KnowledgeGraphRetriever(store, embedder)
        results = retriever.retrieve("查询文本", top_k=3)
        context = retriever.format_context(results)
    """

    def __init__(self, store, embedder):
        """
        参数:
            store:    KnowledgeGraphStore 实例
            embedder: LocalEmbedder 实例
        """
        self.store = store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
        source_file: str = None,
    ) -> List[KgRetrievalResult]:
        """
        查询时分三步走。

        参数:
            query:       用户查询文本
            top_k:       返回社区数
            min_score:   最低相似度阈值
            source_file: 可选，限制特定文档
        返回:
            按相似度排序的检索结果列表
        """
        if not query:
            return []

        # ── Step 1: 嵌入查询 ──
        try:
            query_vec = self.embedder.embed(query)
        except Exception:
            return []
        query_arr = np.array(query_vec, dtype=np.float32)
        q_norm = float(np.linalg.norm(query_arr))
        if q_norm == 0:
            return []

        # ── Step 2: 加载社区摘要嵌入 ──
        communities = self.store.get_all_communities(
            level=0, source_file=source_file
        )
        if not communities:
            return []

        # ── Step 3: 计算余弦相似度 ──
        scored = []
        for comm in communities:
            emb = comm.get("summary_embedding", [])
            if not emb:
                continue

            comm_arr = np.array(emb, dtype=np.float32)
            c_norm = float(np.linalg.norm(comm_arr))
            if c_norm == 0:
                continue

            # 确保维度一致
            if len(comm_arr) != len(query_arr):
                continue

            sim = float(np.dot(query_arr, comm_arr) / (q_norm * c_norm))
            if sim >= min_score:
                scored.append((comm, sim))

        # ── Step 4: 排序 ──
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]

        # ── Step 5: 展开社区上下文 ──
        return [self._expand_community(comm, sim) for comm, sim in top]

    def _expand_community(
        self, community: dict, score: float
    ) -> KgRetrievalResult:
        """
        展开社区为详细结果：实体 + 关系。
        """
        entity_ids = community.get("entity_ids", [])

        # 获取实体详情
        entity_map = self.store.get_entities(entity_ids=entity_ids) if entity_ids else {}
        entities = list(entity_map.values())

        # 获取内部关系（带名称）
        relations = self.store.get_relations_with_names(
            entity_ids=entity_ids, limit=20
        ) if entity_ids else []

        # 转换实体格式
        entity_list = []
        for e in entities[:20]:
            entity_list.append({
                "id": e["id"],
                "name": e.get("name", ""),
                "entity_type": e.get("entity_type", ""),
                "description": e.get("description", ""),
            })

        # 转换关系格式
        rel_list = []
        for r in relations[:20]:
            rel_list.append({
                "id": r["id"],
                "source": r.get("source_name", ""),
                "target": r.get("target_name", ""),
                "relation": r.get("relation", ""),
                "weight": r.get("weight", 1.0),
            })

        return {
            "community_id": community.get("id", 0),
            "score": round(score, 4),
            "summary": community.get("summary", ""),
            "entities": entity_list,
            "relations": rel_list,
        }

    def retrieve_entities_and_chunks(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_communities: int = 3,
    ) -> Tuple[List[Dict], List[int], List[KgRetrievalResult]]:
        """
        增强检索：返回匹配的实体、关联 chunk ID、社区上下文。

        返回:
            (matched_entities, chunk_ids, community_results)
        """
        community_results = self.retrieve(query, top_k=top_k_communities)

        # 收集所有命中实体
        all_entities = []
        seen_ids = set()
        for cr in community_results:
            for e in cr.get("entities", []):
                if e["id"] not in seen_ids:
                    seen_ids.add(e["id"])
                    all_entities.append(e)

        # 收集关联 chunks
        entity_ids = [e["id"] for e in all_entities]
        chunk_ids = self.store.get_chunks_for_entities(entity_ids)

        return all_entities[:top_k_entities], chunk_ids, community_results

    @staticmethod
    def format_context(results: List[KgRetrievalResult]) -> str:
        """
        将 KG 检索结果格式化为 LLM 注入上下文。

        输出格式:
            ## 文档知识图谱分析
            ### 主题组 1: <摘要>
            关键实体: 实体A（类型）、实体B（类型）
            关键关系: 实体A → 关系 → 实体B
            ...
        """
        if not results:
            return ""

        parts = ["## 文档知识图谱分析\n"]

        for i, r in enumerate(results, 1):
            summary = r.get("summary", "")
            entities = r.get("entities", [])
            relations = r.get("relations", [])

            # 主题组标题
            if summary:
                parts.append(f"### 主题组 {i}: {summary[:120]}")
            else:
                parts.append(f"### 主题组 {i}")

            # 关键实体
            top_entities = entities[:8]
            if top_entities:
                entity_desc = "、".join(
                    f"{e['name']}（{e.get('entity_type', '未知')}）"
                    for e in top_entities
                )
                parts.append(f"关键实体: {entity_desc}")

            # 关键关系
            top_relations = relations[:8]
            if top_relations:
                parts.append("关键关系:")
                for rel in top_relations:
                    parts.append(
                        f"- {rel['source']} → {rel['relation']} → {rel['target']}"
                    )

            parts.append("")  # 空行分隔

        return "\n".join(parts)

    @staticmethod
    def format_compact(results: List[KgRetrievalResult]) -> str:
        """
        紧凑格式：适合作为简短上下文注入。
        """
        if not results:
            return ""

        lines = []
        for r in results:
            summary = r.get("summary", "")
            relations = r.get("relations", [])[:5]

            if summary:
                lines.append(f"📌 {summary[:100]}")

            for rel in relations:
                lines.append(
                    f"  · {rel['source']} —{rel['relation']}→ {rel['target']}"
                )

        return "\n".join(lines) if lines else ""


def fuse_vector_and_graph(
    vector_results: List[Dict],
    graph_results: List[KgRetrievalResult],
    top_k_vector: int = 3,
) -> Tuple[str, List[Dict]]:
    """
    向量检索 + 图谱检索的融合策略。

    策略: 分层上下文 (Layered Context)
      - KG 上下文作为"结构化概览"，放在向量块之前
      - 向量块保留作为"具体证据"
      - KG 命中的实体用于微调向量结果排序（启发式加分）

    参数:
        vector_results:  rag.retrieve() 的返回
        graph_results:   kg_retriever.retrieve() 的返回
        top_k_vector:    保留的向量结果数
    返回:
        (kg_context_str, enhanced_vector_chunks)
    """
    # ── Step 1: 生成 KG 上下文 ──
    kg_context = ""
    if graph_results:
        kg_context = KnowledgeGraphRetriever.format_context(graph_results)

    # ── Step 2: 可选增强：KG 命中实体提及的 chunk 加分 ──
    if graph_results and vector_results:
        # 收集 KG 命中的实体名
        kg_entity_names = set()
        for cr in graph_results:
            for e in cr.get("entities", []):
                kg_entity_names.add(e.get("name", ""))

        # 对包含 KG 实体的向量结果轻微加分
        for vr in vector_results:
            content = vr.get("content", "")
            bonus = 0.0
            for name in kg_entity_names:
                if name and name in content:
                    bonus += 0.03  # 每个命中实体 +0.03

            if bonus > 0:
                vr["score"] = round(min(vr.get("score", 0) + bonus, 1.0), 4)

        # 重新排序
        vector_results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return kg_context, vector_results[:top_k_vector]
