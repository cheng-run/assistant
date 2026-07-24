"""
智能文档问答助手 — Gradio WebUI

功能：
  - 多轮对话（流式输出）
  - 文档上传 → 自动 RAG 索引
  - RAG 开关（可切换纯对话 / 文档问答模式）
  - 清空对话
"""

import gradio as gr
from rag import RAGTool
from llm import ask_llm_stream

# ── 系统提示词 ──────────────────────────
SYSTEM_PROMPT = (
    "你是一个智能文档问答助手，可以基于上传的文档内容回答用户问题。"
    "回答时请引用文档中的具体信息，并标注来源标题路径。"
    "如果文档中没有相关信息，请如实告知用户。"
)

# ── 全局 RAG 实例 ───────────────────────
rag = RAGTool()


# ═══════════════════════════════════════════════════════════════
#  回调函数
# ═══════════════════════════════════════════════════════════════

def on_upload(file, visual_enabled=False, progress: gr.Progress = gr.Progress()):
    """文档上传 → 索引（分阶段进度可视化）"""
    if file is None:
        sources = rag.vector_store.list_sources()
        if sources:
            return f"📊 已索引 {len(sources)} 个文档: {', '.join(sources)}"
        return "📊 尚未上传文档"

    try:
        fname = file.name.replace("\\", "/").split("/")[-1]

        # ── 阶段 1：文本提取 ──
        progress(0.05, desc=f"📖 [1/4] 正在提取文本 ({fname})...")
        markdown = rag.convert_to_markdown(file.name)

        # ── 阶段 2：标题分块 ──
        progress(0.10, desc="✂️ [2/4] 正在按标题分块...")
        sections = rag.split_by_headings(markdown)

        # ── 阶段 3：Token 分块 ──
        progress(0.20, desc="📝 [3/4] 正在 Token 分块...")
        chunks = rag.chunk_by_tokens(sections)

        # ── 阶段 4：向量化（最耗时） ──
        progress(0.30, desc=f"🧠 [4/4] 正在向量化（共 {len(chunks)} 块，约需 1-2 分钟）...")
        count = rag.vector_store.add(chunks, source_file=fname)

        # ── 视觉索引（可选） ──
        if visual_enabled and file.name.lower().endswith('.pdf'):
            progress(0.70, desc=f"🎨 [视觉] 正在生成页面图片并索引视觉层...")
            try:
                visual_count = rag._index_visual_pages(fname, file.name)
                count += visual_count
            except Exception as ve:
                pass  # 视觉索引失败不影响文本路径

        # ── 完成 ──
        progress(1.0, desc="✅ 完成！")
        sources = rag.vector_store.list_sources()
        visual_note = " (含视觉索引)" if visual_enabled else ""
        return f"✅ 已索引 **{fname}**（{count} 个片段）{visual_note}\n\n📊 全部文档: {', '.join(sources)}"
    except Exception as e:
        return f"❌ 索引失败: {e}"


def on_message(message, history, rag_enabled, visual_enabled=False):
    """
    多轮对话回调（流式输出）。

    参数:
        message:       用户输入文本
        history:       Chatbot 的当前消息列表 [{"role": ..., "content": ...}, ...]
        rag_enabled:   RAG 开关状态
        visual_enabled: 视觉检索开关
    """
    if not message:
        yield history
        return

    # ── 构建 LLM messages ────────────────
    llm_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 文档大纲注入（全局结构认知）
    if rag_enabled:
        outline_text = rag.get_all_outlines()
        if outline_text:
            llm_messages.append({
                "role": "system",
                "content": outline_text,
            })

    # RAG 上下文注入
    if rag_enabled:
        try:
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

    # 对话历史（限制最近 20 轮，避免超出上下文窗口）
    MAX_HISTORY_TURNS = 20
    for h in history[-MAX_HISTORY_TURNS * 2:]:
        llm_messages.append(h)

    # 当前问题
    llm_messages.append({"role": "user", "content": message})

    # ── 流式生成 ─────────────────────────
    response_text = ""
    try:
        for token in ask_llm_stream(llm_messages):
            response_text += token
            yield history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response_text},
            ]
    except Exception as e:
        response_text += f"\n\n[流式生成中断: {e}]"
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response_text},
        ]


def on_clear():
    """清空对话 + 重置状态"""
    return [], "📊 尚未上传文档"


def on_clear_docs():
    """清空所有已索引的文档"""
    rag.vector_store.clear()
    return "📊 已清空全部索引的文档"


# ═══════════════════════════════════════════════════════════════
#  UI 布局
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(title="智能文档问答助手") as demo:
    # ── 标题 ─────────────────────────────
    gr.Markdown(
        "# 🤖 智能文档问答助手\n"
        "基于 RAG（检索增强生成）的智能文档问答 Demo — 上传文档，提问即可获得基于文档内容的回答。"
    )

    # ── 对话区 ───────────────────────────
    chatbot = gr.Chatbot(
        height=520,
        placeholder="👋 你好！上传文档后即可开始提问...",
        label="对话",
    )

    # ── 工具栏 ───────────────────────────
    with gr.Row(equal_height=True):
        file_upload = gr.File(
            label="📄 上传文档",
            file_types=[".md", ".txt", ".docx", ".pdf"],
            file_count="single",
            scale=3,
        )
        rag_toggle = gr.Checkbox(
            label="🔍 RAG 检索",
            value=True,
            scale=1,
            info="基于文档回答",
        )
        visual_toggle = gr.Checkbox(
            label="👁 视觉检索",
            value=False,
            scale=1,
            info="CLIP+DeepSeek 图表理解",
        )
        clear_btn = gr.Button(
            "🗑 清空对话",
            variant="secondary",
            scale=1,
        )
        clear_docs_btn = gr.Button(
            "🧹 清空文档",
            variant="stop",
            scale=1,
        )

    # ── 状态提示 ─────────────────────────
    status = gr.Markdown("📊 尚未上传文档")

    # ── 输入框 ───────────────────────────
    chat_input = gr.Textbox(
        placeholder="输入问题，按 Enter 发送...",
        label="输入",
        autofocus=True,
    )

    # ── 示例 ─────────────────────────────
    gr.Examples(
        examples=[
            "请总结一下文档的主要内容",
            "文档中提到了哪些关键概念？",
            "根据文档，XXX 是如何实现的？",
        ],
        inputs=chat_input,
    )

    # ═════════════════════════════════════
    #  事件绑定
    # ═════════════════════════════════════

    # 文档上传
    file_upload.change(
        on_upload,
        inputs=[file_upload, visual_toggle],
        outputs=[status],
    )

    # 对话提交（流式）
    msg_event = chat_input.submit(
        on_message,
        inputs=[chat_input, chatbot, rag_toggle, visual_toggle],
        outputs=[chatbot],
    )
    # 发送后清空输入框
    msg_event.then(lambda: "", None, [chat_input])

    # 清空对话
    clear_btn.click(
        on_clear,
        inputs=None,
        outputs=[chatbot, status],
    )

    # 清空文档索引
    clear_docs_btn.click(
        on_clear_docs,
        inputs=None,
        outputs=[status],
    )

    # 页面加载时刷新状态
    demo.load(
        lambda: on_upload(None),
        inputs=None,
        outputs=[status],
    )


# ═══════════════════════════════════════════════════════════════
#  启动
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
