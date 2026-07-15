"""
RAG 文档处理子系统

文档 → Markdown → 标题分块 → Token 分块（自适应重叠）
→ 本地嵌入（fastembed） → 向量存储（SQLite） → 相似度检索
"""
from .ragtool import RAGTool
from .embedding import LocalEmbedder, VectorStore

__all__ = ["RAGTool", "LocalEmbedder", "VectorStore"]
