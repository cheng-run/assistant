"""
tests/legacy_on_message.py — 重构前的旧对话逻辑（回归基准）

阶段 B 会从 app.py 删除手写拼装逻辑。此模块在删除前冻结旧实现，
供 tests/run_golden.py 作 OLD 路径与新版 agent 做对比回归。
逻辑与 git 历史中 app.py:on_message 完全一致，禁止在此"顺手优化"。
"""

from llm import ask_llm_stream
from rag import RAGTool

SYSTEM_PROMPT = (
    "你是一个智能文档问答助手，可以基于上传的文档内容回答用户问题。"
    "回答时请引用文档中的具体信息，并标注来源标题路径。"
    "如果文档中没有相关信息，请如实告知用户。"
)

rag = RAGTool()

MAX_HISTORY_TURNS = 20


def build_legacy_messages(message, history, rag_enabled, visual_enabled=False, kg_enabled=False):
    """
    按旧逻辑构建 LLM messages（与重构前 app.py:on_message 逐行对应）。

    返回 (llm_messages, chunks)：
      - llm_messages: 送给 LLM 的完整消息列表
      - chunks:       检索到的片段（供金集评估检索质量 P@3）
    """
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    chunks = []

    # 文档大纲注入（全局结构认知）
    if rag_enabled:
        outline_text = rag.get_all_outlines()
        if outline_text:
            llm_messages.append({"role": "system", "content": outline_text})

    # RAG 上下文注入
    if rag_enabled:
        try:
            if kg_enabled:
                kg_context, chunks = rag.retrieve_with_kg(
                    message, top_k=3, enable_visual=visual_enabled
                )
                if kg_context:
                    llm_messages.append({"role": "system", "content": kg_context})
            else:
                chunks = rag.retrieve(message, top_k=3, enable_visual=visual_enabled)

            if chunks:
                parts = []
                for c in chunks:
                    heading = " > ".join(c.get("heading_path", [])) or "文档"
                    parts.append(f"[{heading}]\n{c['content']}")
                context = "\n\n---\n\n".join(parts)
                llm_messages.append({
                    "role": "system",
                    "content": (
                        "以下是参考文档的相关内容，请基于这些内容回答用户问题。"
                        "如果内容不足以回答问题，请如实说明：\n\n"
                        f"{context}"
                    ),
                })
        except Exception:
            llm_messages.append({
                "role": "system",
                "content": "（注意：文档检索暂时失败，以下回答基于通用知识。）",
            })

    # 对话历史（最近 20 轮）
    for h in history[-MAX_HISTORY_TURNS * 2:]:
        llm_messages.append(h)

    # 当前问题
    llm_messages.append({"role": "user", "content": message})

    return llm_messages, chunks


def legacy_stream_tokens(message, history, rag_enabled, visual_enabled=False, kg_enabled=False):
    """逐 token 产出旧路径的回答（等价于重构前 on_message 的生成段）。"""
    llm_messages, _ = build_legacy_messages(
        message, history, rag_enabled, visual_enabled, kg_enabled
    )
    yield from ask_llm_stream(llm_messages)
