"""
RAG 文档处理子系统

文档 → Markdown → 标题分块 → Token 分块（自适应重叠）
→ Ollama 本地嵌入（bge-m3） → 向量存储（SQLite） → 相似度检索
→ 知识图谱构建 → 实体关系检索 → 融合增强
"""
from .ragtool import RAGTool
from .embedding import (
    ImageEmbedder,
    LocalEmbedder,
    OllamaEmbedder,
    VectorStore,
)
from .kg import (
    CommunityDetector,
    CommunitySummarizer,
    EntityRelationExtractor,
    KnowledgeGraphRetriever,
    KnowledgeGraphStore,
    fuse_vector_and_graph,
)

__all__ = [
    "CommunityDetector",
    "CommunitySummarizer",
    "EntityRelationExtractor",
    "ImageEmbedder",
    "KnowledgeGraphRetriever",
    "KnowledgeGraphStore",
    "LocalEmbedder",
    "OllamaEmbedder",
    "RAGTool",
    "VectorStore",
    "fuse_vector_and_graph",
]
