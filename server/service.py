"""server/service.py — 上传索引流水线（SSE 生成器）

复用现有 RAGTool 流水线：docling 转换 → 标题分块 → token 分块 → 分批向量化
→ 可选视觉索引 / KG 构建。全部阻塞调用包 asyncio.to_thread，避免卡死事件循环。
"""

import asyncio

from agent.tools import rag
from server.sse import sse


async def run_upload(path: str, fname: str, visual: bool, kg: bool):
    """按阶段产出 SSE 帧：stage / progress / result / error。"""
    try:
        yield sse("stage", {
            "stage": "extract", "pct": 0.05,
            "label": f"📖 [1/4] 正在提取文本 ({fname})...",
        })
        markdown = await asyncio.to_thread(rag.convert_to_markdown, path)

        yield sse("stage", {"stage": "split", "pct": 0.10, "label": "✂️ [2/4] 正在按标题分块..."})
        sections = rag.split_by_headings(markdown)

        yield sse("stage", {"stage": "chunk", "pct": 0.20, "label": "📝 [3/4] 正在 Token 分块..."})
        chunks = rag.chunk_by_tokens(sections)

        # 分批向量化（0.30 → 0.70）
        total = len(chunks)
        batch = max(32, total // 5) if total else 1
        count = 0
        for i in range(0, total, batch):
            b = chunks[i:i + batch]
            count += await asyncio.to_thread(rag.vector_store.add, b, fname)
            done = min(i + batch, total)
            yield sse("progress", {
                "stage": "vectorize",
                "pct": 0.30 + 0.40 * (done / total),
                "done": done, "total": total,
                "label": f"🧠 [4/4] 向量化 ({done}/{total})...",
            })

        kg_note, visual_flag = "", False

        # 视觉索引（可选，仅 PDF）
        if visual and fname.lower().endswith(".pdf"):
            yield sse("stage", {"stage": "visual", "pct": 0.70, "label": "🎨 [视觉] 生成页面图片并索引..."})
            try:
                count += await asyncio.to_thread(rag._index_visual_pages, fname, path)
                visual_flag = True
            except Exception:
                pass  # 视觉索引失败不影响文本路径

        # 知识图谱构建（可选）
        if kg:
            yield sse("stage", {"stage": "kg", "pct": 0.75, "label": "🧠 [5/5] 构建知识图谱..."})
            try:
                stats = await asyncio.to_thread(
                    rag.build_knowledge_graph, chunks, fname, path
                )
                kg_note = (
                    f"{stats['entities']}实体/{stats['relations']}关系"
                    f"/{stats['communities']}社区"
                )
            except Exception:
                pass

        yield sse("result", {
            "source_file": fname,
            "chunks": count,
            "kg": kg_note,
            "visual": visual_flag,
            "sources": rag.vector_store.list_sources(),
        })
    except Exception as e:
        yield sse("error", {"message": f"索引失败: {e}"})
