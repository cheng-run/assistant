"""agent/router.py — 检索路由引擎（企业级落地）

分层：
  L0 规则意图识别 classify_intent()        ← 确定性、零 token、第一道防线
  L1 LLM 意图识别 llm_intent()（可选）      ← ROUTER=llm 时启用
  StrategyRegistry 意图→策略 配置化映射
  multi_retrieve() 多路执行 + RRF 排名融合

设计原则：
  - 可回退：任何模式无数据自动降级 vector；seed 始终含 vector 保险
  - 可组合：combined 意图 → 多模式并行（kg+visual+vector）
  - RRF 融合：跨模式 score 不可比（bge 文本 vs CLIP 视觉），只看排名，稳健
"""

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List

from langchain_core.tools import tool


# ═══════════════════════════════════════════════════════════════
#  意图识别（L0 规则）
# ═══════════════════════════════════════════════════════════════

# 关键词表（新增意图/关键词只需在此维护）
_KG_HINTS = ("关系", "实体", "图谱", "联系", "依赖", "属于", "分类", "谁负责", "哪些", "关联")
_VISUAL_HINTS = (
    "图表", "图", "曲线", "趋势", "折线", "柱状", "饼图", "图片",
    "增长", "多少", "数字", "stars", "可视化", "对比图", "数量",
)
_SUMMARY_HINTS = ("总结", "概述", "摘要", "对比", "综合", "归纳", "梳理", "整体")
_LONG_QUERY_CHARS = 30  # 长句视为复杂问题 → summary/expanded


def classify_intent(query: str) -> str:
    """L0 规则：返回意图名。命中多类 → combined（多模式并行）。"""
    q = (query or "").lower()
    hits = []
    if any(k in q for k in _KG_HINTS):
        hits.append("relation")
    if any(k in q for k in _VISUAL_HINTS):
        hits.append("visual")
    if len(q) > _LONG_QUERY_CHARS or any(k in q for k in _SUMMARY_HINTS):
        hits.append("summary")

    if len(hits) >= 2:
        return "combined"
    if hits:
        return hits[0]
    return "concept"  # 默认概念类（纯向量）


# ═══════════════════════════════════════════════════════════════
#  策略注册表（意图 → 检索策略 配置化映射）
# ═══════════════════════════════════════════════════════════════

@dataclass
class RetrievalStrategy:
    modes: tuple  # 参与检索的模式，如 ("kg", "vector")
    fusion: str = "rrf"
    parallel: bool = True


class StrategyRegistry:
    """意图 → 策略 的配置化映射（新增意图只需加一行）。"""

    RULES: Dict[str, RetrievalStrategy] = {
        "concept":  RetrievalStrategy(("vector",)),
        "relation": RetrievalStrategy(("kg", "vector")),        # 关系类：图谱+向量并行
        "visual":   RetrievalStrategy(("visual", "vector")),    # 图表类：视觉+向量
        "summary":  RetrievalStrategy(("expanded", "vector")),  # 总结类：多查询+向量
        "combined": RetrievalStrategy(("kg", "visual", "vector")),  # 多模式全开
        "default":  RetrievalStrategy(("vector",)),
    }

    @classmethod
    def resolve(cls, intent: str) -> RetrievalStrategy:
        return cls.RULES.get(intent, cls.RULES["default"])


# ═══════════════════════════════════════════════════════════════
#  多路执行 + RRF 融合
# ═══════════════════════════════════════════════════════════════

def _rrf_fuse(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """Reciprocal Rank Fusion：跨检索器只看排名，score 不可比也能融合。"""
    rrf: Dict[int, float] = {}
    source: Dict[int, Dict] = {}
    for results in ranked_lists:
        for rank, r in enumerate(results):
            cid = r.get("id")
            if cid is None:
                continue
            rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in source:
                source[cid] = r
    merged = []
    for cid, score in sorted(rrf.items(), key=lambda x: x[1], reverse=True):
        item = dict(source[cid])
        item["score"] = round(score, 4)
        merged.append(item)
    return merged


def multi_retrieve(
    rag, query: str, modes: tuple, top_k: int = 3,
) -> tuple:
    """按 mode 列表多路检索，RRF 融合，取 top_k。

    返回 (chunks, kg_context)：
      - chunks: RRF 融合后的检索片段
      - kg_context: 若含 kg 模式，返回图谱分析上下文（供单独注入）；否则 None

    降级：某模式抛异常/无结果自动跳过；始终依赖 vector 兜底。
    """
    ranked_lists: List[List[Dict]] = []
    kg_context = None
    for mode in modes:
        try:
            if mode == "kg":
                kg_context, chunks = rag.retrieve_with_kg(query, top_k=top_k)
            elif mode == "visual":
                chunks = rag.retrieve(query, top_k=top_k, enable_visual=True)
            elif mode == "expanded":
                chunks = rag.retrieve_expanded(
                    query, top_k=top_k, enable_mqe=True, enable_hyde=True
                )
            else:  # vector
                chunks = rag.retrieve(query, top_k=top_k)
        except Exception:
            chunks = []
        if chunks:
            ranked_lists.append(chunks)
    if not ranked_lists:
        return [], kg_context
    return _rrf_fuse(ranked_lists)[:top_k], kg_context


# ═══════════════════════════════════════════════════════════════
#  路由入口
# ═══════════════════════════════════════════════════════════════

def route_query(query: str) -> RetrievalStrategy:
    """完整路由：意图识别 → 解析策略。返回 (intent, strategy, modes)。"""
    intent = classify_intent(query)
    return intent, StrategyRegistry.resolve(intent)


# ═══════════════════════════════════════════════════════════════
#  L1 LLM 意图识别（可选，ROUTER=llm 时启用）
# ═══════════════════════════════════════════════════════════════

_LLM_INTENT_PROMPT = (
    "判断下面用户问题最合适的文档检索策略，只输出一个词：\n"
    "concept=概念/事实查询；relation=实体关系/图谱查询；visual=图表/图片/数据查询；"
    "summary=总结/概述/综合；combined=问题复杂需多策略并行。\n"
    "问题：{query}\n输出："
)


def llm_intent(query: str) -> str:
    """L1：LLM 意图分类（比规则准，但每次 +1 次调用）。失败回退规则。"""
    from llm import ask_llm

    try:
        resp = ask_llm([{"role": "user", "content": _LLM_INTENT_PROMPT.format(query=query)}])
        resp = (resp or "").strip().lower()
        if resp in StrategyRegistry.RULES:
            return resp
    except Exception:
        pass
    return classify_intent(query)


def get_intent(query: str) -> str:
    """根据 ROUTER 环境变量选择 L0/L1 意图识别。"""
    if os.getenv("ROUTER", "rule") == "llm":
        return llm_intent(query)
    return classify_intent(query)


# ═══════════════════════════════════════════════════════════════
#  网页搜索工具（TAVILY）
# ═══════════════════════════════════════════════════════════════

@tool
def web_search(query: str, max_results: int = 4) -> str:
    """搜索互联网获取最新信息（当文档中没有答案时使用，返回标题/摘要/来源链接）。"""
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return "（未配置 TAVILY_API_KEY，无法联网搜索）"
    try:
        body = json.dumps({
            "api_key": api_key, "query": query, "max_results": max_results,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.tavily.com/search", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        return f"（联网搜索失败: {e}）"
    results = data.get("results") or []
    if not results:
        return "（未搜索到相关结果）"
    parts = []
    for r in results[:max_results]:
        title = r.get("title", "")
        content = (r.get("content") or "")[:300]
        url = r.get("url", "")
        parts.append(f"### {title}\n{content}\n来源: {url}")
    return "\n\n".join(parts)
