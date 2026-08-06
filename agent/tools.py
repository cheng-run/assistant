"""agent/tools.py — 把现有 RAG / 记忆引擎包装成 LangChain 工具

设计要点：
  - 一个"富工具" search_documents（mode 参数控制检索策略）而非四个散工具，
    减少模型做工具选择的失败率；mode 由 args_schema 的 Literal 枚举约束。
  - get_outline / list_sources / memory_recall / memory_remember 为辅助工具。
  - 本文件持有 RAGTool / MemoryTool 单例（与 app.py 共用同一批 SQLite 库）。
"""

from typing import Any, Dict, List, Literal, Tuple

from langchain_core.tools import StructuredTool, tool
from pydantic import Field, create_model

from rag import RAGTool
from memory.base import extract_text
from memory.tool import MemoryTool

# ── 共享单例 ───────────────────────────────
rag = RAGTool()
mem = MemoryTool(working_capacity=20)

# 全部检索模式（顺序即 schema 里 Literal 的展示顺序）
ALL_MODES: Tuple[str, ...] = ("vector", "kg", "expanded", "visual")

_MODE_DESC = {
    "vector": "语义向量检索",
    "kg": "向量+知识图谱融合",
    "expanded": "多查询+假设文档扩展",
    "visual": "文本+图片跨模态检索",
}


def allowed_modes(include_kg: bool, include_visual: bool) -> Tuple[str, ...]:
    """由 UI 开关决定 search_documents 的允许模式（保序去重）。"""
    modes = ["vector"]
    if include_kg:
        modes.append("kg")
    if include_visual:
        modes.append("visual")
    modes.append("expanded")
    return tuple(dict.fromkeys(modes))


# ═══════════════════════════════════════════════════════════════
#  工具实现
# ═══════════════════════════════════════════════════════════════

def _fmt_chunks(chunks: List[Dict]) -> str:
    """与重构前 app.py:on_message 相同的片段渲染：`[标题路径] 内容`。"""
    if not chunks:
        return "（未检索到相关片段）"
    parts = []
    for c in chunks:
        heading = " > ".join(c.get("heading_path", [])) or "文档"
        parts.append(f"[{heading}]\n{c['content']}")
    return "\n\n---\n\n".join(parts)


def _search_impl(query: str, top_k: int = 3, mode: str = "vector") -> str:
    """search_documents 的实现体：按 mode 分发到 RAGTool 的检索路径。"""
    if mode == "kg":
        kg_context, chunks = rag.retrieve_with_kg(query, top_k=top_k)
        if kg_context:
            return kg_context + "\n\n" + _fmt_chunks(chunks)
        return _fmt_chunks(chunks)
    if mode == "expanded":
        chunks = rag.retrieve_expanded(
            query, top_k=top_k, enable_mqe=True, enable_hyde=True
        )
        return _fmt_chunks(chunks)
    # vector / visual
    chunks = rag.retrieve(query, top_k=top_k, enable_visual=(mode == "visual"))
    return _fmt_chunks(chunks)


def _build_search_schema(allowed_modes: Tuple[str, ...]) -> Any:
    """按允许模式动态构造 args_schema —— 在 schema 层禁止非法 mode。"""
    modes_literal = Literal[tuple(allowed_modes)]
    return create_model(
        "SearchDocumentsArgs",
        query=(str, Field(description="要检索的问题或关键词")),
        top_k=(int, Field(default=3, ge=1, le=10, description="返回的片段数量（1-10）")),
        mode=(
            modes_literal,
            Field(
                default=allowed_modes[0],
                description="检索模式：" + "；".join(
                    f"{m}={_MODE_DESC[m]}" for m in allowed_modes
                ),
            ),
        ),
    )


def make_search_documents(allowed_modes: Tuple[str, ...] = ALL_MODES) -> StructuredTool:
    """构造 search_documents 工具（允许模式由 UI 开关决定）。"""
    allowed = tuple(m for m in ALL_MODES if m in allowed_modes) or ("vector",)
    return StructuredTool.from_function(
        func=_search_impl,
        name="search_documents",
        description=(
            "搜索已索引的文档，返回与问题相关的片段（含来源标题路径）。"
            "回答文档相关问题前必须先调用此工具。"
        ),
        args_schema=_build_search_schema(allowed),
    )


@tool
def get_outline() -> str:
    """获取所有已索引文档的大纲/章节结构（全局结构认知）。"""
    return rag.get_all_outlines()


@tool
def list_sources() -> str:
    """列出所有已索引的文档文件名。"""
    sources = rag.vector_store.list_sources()
    return ", ".join(sources) if sources else "（尚未上传文档）"


@tool
def memory_recall(query: str, limit: int = 3) -> str:
    """从长期记忆检索与用户相关的历史事实/偏好（如名字、要求、前文结论）。"""
    items = mem.get_relevant_context(query, limit)
    if not items:
        return "（暂无相关记忆）"
    return "\n".join(f"- {extract_text(c['content'])}" for c in items)


@tool
def memory_remember(content: str, importance: float = 0.7) -> str:
    """把关于用户的重要事实（如名字、偏好、要求）存入长期记忆。"""
    mem.add(content, importance=importance, memory_type="episodic")
    return "已记住。"
