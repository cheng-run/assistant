"""
RAG 文档处理工具

文档 → Markdown → 标题分块 → Token 分块（自适应重叠）→ 向量嵌入 → 存储检索
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from memory.base import is_cjk


# ── 辅助函数 ──────────────────────────

def _has_body_text(text: str) -> bool:
    """检查文本是否包含非标题的正文内容（至少有 1 行非 Markdown 标题行有实质内容）"""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        # 跳过 Markdown 标题行（# 开头）
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        # 跳过分隔线、纯符号行
        if re.match(r"^[-=_*]{3,}$", stripped):
            continue
        return True
    return False


class RAGTool:
    """
    RAG 文档处理工具 — 三步流水线：转换 → 标题分块 → Token 分块

    用法:
        rag = RAGTool(knowledge_base_path="./data_db")
        text = rag.convert_to_markdown("docs/report.docx")
        sections = rag.split_by_headings(text)
        chunks = rag.chunk_by_tokens(sections, chunk_tokens=512, overlap_tokens=64)
    """

    def __init__(
        self,
        knowledge_base_path: str = "./data_db",
    ):
        self.knowledge_base_path = Path(knowledge_base_path)
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)

        # 懒加载嵌入器和向量存储
        self._embedder = None
        self._vector_store = None

    @property
    def embedder(self):
        if self._embedder is None:
            from .embedding import LocalEmbedder
            self._embedder = LocalEmbedder()
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            from .embedding import VectorStore
            db_path = str(self.knowledge_base_path / "rag_vectors.db")
            self._vector_store = VectorStore(self.embedder, db_path=db_path)
        return self._vector_store

    # ═══════════════════════════════════════════════════════════════
    #  第零步：文档 → Markdown
    # ═══════════════════════════════════════════════════════════════

    def convert_to_markdown(self, path: str) -> str:
        """
        将文档转换为 markdown 文本。

        支持格式:
            .md   — 直接读取
            .txt  — 直接读取
            .docx — 需要 python-docx（可选）
            .pdf  — 需要 pymupdf / pdfplumber（可选）

        参数:
            path: 文件路径或目录路径（目录则递归处理所有支持的文件）
        返回:
            markdown 文本（多文件以分隔符连接）
        """
        file_path = Path(path)

        if file_path.is_dir():
            return self._convert_directory(file_path)
        return self._convert_single(file_path)

    def _convert_single(self, file_path: Path) -> str:
        """转换单个文件"""
        suffix = file_path.suffix.lower()

        if suffix == ".md":
            return file_path.read_text(encoding="utf-8")
        elif suffix == ".txt":
            return file_path.read_text(encoding="utf-8")
        elif suffix == ".docx":
            return self._convert_docx(file_path)
        elif suffix == ".pdf":
            return self._convert_pdf(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}（支持: .md, .txt, .docx, .pdf）")

    def _convert_directory(self, dir_path: Path) -> str:
        """递归处理目录下所有支持的文档，以分隔符拼接"""
        supported = {".md", ".txt", ".docx", ".pdf"}
        files = sorted(
            f for f in dir_path.rglob("*") if f.suffix.lower() in supported
        )
        if not files:
            raise ValueError(f"目录 {dir_path} 中未找到支持的文档")

        parts: list[str] = []
        for f in files:
            try:
                text = self._convert_single(f)
                parts.append(f"<!-- source: {f.name} -->\n\n{text}")
            except Exception as e:
                parts.append(f"<!-- ERROR reading {f.name}: {e} -->")
        return "\n\n---\n\n".join(parts)

    def _convert_docx(self, file_path: Path) -> str:
        """docx → markdown（需安装 python-docx）"""
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "处理 .docx 需要 python-docx，请运行: uv add python-docx"
            )

        doc = Document(str(file_path))
        md_lines: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                md_lines.append("")
                continue

            style = para.style.name if para.style else ""
            if style.startswith("Heading"):
                level = style.split()[-1]
                try:
                    level = int(level)
                except ValueError:
                    level = 1
                md_lines.append(f"{'#' * level} {text}")
            elif style == "Title":
                md_lines.append(f"# {text}")
            else:
                # 处理行内格式（粗体、斜体）
                text = self._format_docx_runs(para)
                md_lines.append(text)

        return "\n\n".join(md_lines)

    @staticmethod
    def _format_docx_runs(para) -> str:
        """处理 docx 段落中的行内格式 run"""
        result: list[str] = []
        for run in para.runs:
            t = run.text
            if run.bold:
                t = f"**{t}**"
            if run.italic:
                t = f"*{t}*"
            result.append(t)
        return "".join(result)

    def _convert_pdf(self, file_path: Path) -> str:
        """pdf → markdown（需安装 pymupdf 或 pdfplumber）"""
        # 优先尝试 pymupdf (fitz)
        fitz_available = True
        try:
            import fitz
        except ImportError:
            fitz_available = False

        if fitz_available:
            try:
                doc = fitz.open(str(file_path))
                md_lines: list[str] = []
                for page in doc:
                    text = page.get_text("text")
                    if text:
                        md_lines.append(text)
                doc.close()
                return "\n\n".join(md_lines)
            except Exception as e:
                # pymupdf 处理失败（如损坏的 PDF），回退到 pdfplumber
                pass

        # 回退到 pdfplumber
        pdfplumber_available = True
        try:
            import pdfplumber
        except ImportError:
            pdfplumber_available = False

        if pdfplumber_available:
            try:
                with pdfplumber.open(str(file_path)) as pdf:
                    md_lines = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            md_lines.append(text)
                    return "\n\n".join(md_lines)
            except Exception as e:
                pass

        if not fitz_available and not pdfplumber_available:
            raise ImportError(
                "处理 .pdf 需要 pymupdf 或 pdfplumber，请运行: uv add pymupdf"
            )
        raise ImportError(
            "处理 .pdf 失败。pymupdf 和 pdfplumber 均已尝试但均无法解析该文件。"
        )

    # ═══════════════════════════════════════════════════════════════
    #  第一步：按标题分块（保持语义完整性）
    # ═══════════════════════════════════════════════════════════════

    def split_by_headings(self, text: str) -> List[Dict]:
        """
        按 Markdown 标题层级 + 段落边界分割文本（参照教程 8.3.4 节）。

        改造要点：
        - 标题匹配时：保存当前段落 → 更新标题栈 → 标题行加入缓冲区
        - 空行匹配时：保存当前段落（段落级拆分）
        - 每个段落携带当前 heading_path，天然保留层级上下文

        返回:
            [
                {
                    "heading_path": ["第1章", "1.1 概述"],
                    "heading_level": 2,
                    "content": "这是该标题下的某一段落...",
                },
                ...
            ]
        """
        if not text:
            return []

        lines = text.splitlines()
        heading_stack: List[Dict] = []
        sections: List[Dict] = []
        buf: List[str] = []

        for line in lines:
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # ── 保存当前段落 ──
                if buf:
                    content = "\n".join(buf).strip()
                    if content and _has_body_text(content):
                        sections.append({
                            "heading_path": [h["title"] for h in heading_stack],
                            "heading_level": heading_stack[-1]["level"] if heading_stack else 0,
                            "content": content,
                        })
                    buf = []

                # ── 更新标题栈 ──
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                while heading_stack and heading_stack[-1]["level"] >= level:
                    heading_stack.pop()
                heading_stack.append({"level": level, "title": title})

                # ── 标题行加入缓冲区（保留在内容中） ──
                buf.append(line)

            elif line.strip() == "":
                # ── 空行 → 段落边界，保存当前段落 ──
                if buf:
                    content = "\n".join(buf).strip()
                    if content and _has_body_text(content):
                        sections.append({
                            "heading_path": [h["title"] for h in heading_stack],
                            "heading_level": heading_stack[-1]["level"] if heading_stack else 0,
                            "content": content,
                        })
                    buf = []

            else:
                buf.append(line)

        # ── 最后一个段落 ──
        if buf:
            content = "\n".join(buf).strip()
            if content and _has_body_text(content):
                sections.append({
                    "heading_path": [h["title"] for h in heading_stack],
                    "heading_level": heading_stack[-1]["level"] if heading_stack else 0,
                    "content": content,
                })

        return sections

    # ═══════════════════════════════════════════════════════════════
    #  第二步：按 Token 数切块（自适应重叠 + 上下文前缀）
    # ═══════════════════════════════════════════════════════════════

    def chunk_by_tokens(
        self,
        sections: List[Dict],
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        adaptive_overlap: bool = True,
    ) -> List[Dict]:
        """
        基于 Token 的智能分块，带自适应重叠和上下文前缀注入。

        参数:
            sections:         split_by_headings 输出
            chunk_tokens:     每块最大 token 数
            overlap_tokens:   基础重叠 token 数
            adaptive_overlap: 启用自适应重叠率
        返回:
            [{"content": ..., "heading_path": ..., "token_count": ..., "chunk_index": ..., "cross_section_bridge": ...}, ...]
        """
        all_chunks: List[Dict] = []
        prev_section_tail: str = None  # 前一个 section 的尾句，用作引桥

        for section in sections:
            content = section["content"]
            heading_path = section["heading_path"]

            # ── 构建上下文前缀 ──
            prefix = self._build_context_prefix(heading_path)

            content_tokens = self._approx_token_len(content)

            if content_tokens <= chunk_tokens:
                # 短 section，无需再切
                bridge = prev_section_tail if (prev_section_tail and adaptive_overlap) else None
                full_content, is_bridge = self._build_chunk_content(
                    content, prefix, bridge
                )
                chunk = {
                    "heading_path": heading_path,
                    "content": full_content,
                    "token_count": self._approx_token_len(full_content),
                    "chunk_index": 0,
                    "cross_section_bridge": is_bridge,
                }
                all_chunks.append(chunk)
                prev_section_tail = self._last_sentence(content)
                continue

            # ── 需要切分 ──
            sub_chunks = self._split_long_section(
                content, chunk_tokens, overlap_tokens
            )

            for i, sub_content in enumerate(sub_chunks):
                # 跨 section 引桥（仅第一个子块）
                bridge = prev_section_tail if (i == 0 and prev_section_tail and adaptive_overlap) else None
                full_content, is_bridge = self._build_chunk_content(
                    sub_content, prefix, bridge
                )

                all_chunks.append({
                    "heading_path": heading_path,
                    "content": full_content,
                    "token_count": self._approx_token_len(full_content),
                    "chunk_index": i,
                    "cross_section_bridge": is_bridge,
                })

            # 当前 section 尾句留给下一个 section
            prev_section_tail = self._last_sentence(content)

        return all_chunks

    # ── 上下文前缀 ──────────────────────

    @staticmethod
    def _build_context_prefix(heading_path: List[str]) -> str:
        """构建 [Context: H1 > H2 > H3] 前缀"""
        if not heading_path:
            return ""
        return "[Context: " + " > ".join(heading_path) + "]"

    @staticmethod
    def _build_chunk_content(content: str, prefix: str, bridge_tail: str = None) -> tuple:
        """构建 chunk 内容，统一处理前缀和跨 section 引桥。返回 (content, is_bridge)。"""
        is_bridge = False
        if bridge_tail:
            bridge = bridge_tail[:80]
            full = (
                f"{prefix}\n\n[上文: {bridge}...]\n\n{content}"
                if prefix else
                f"[上文: {bridge}...]\n\n{content}"
            )
            is_bridge = True
        else:
            full = f"{prefix}\n\n{content}" if prefix else content
        return full, is_bridge

    @staticmethod
    def _last_sentence(text: str) -> str:
        """提取文本最后一句（以 。！？.!? 结尾）"""
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        clean = [s.strip() for s in sentences if s.strip()]
        return clean[-1] if clean else ""

    # ── 长 section 切分 ─────────────────

    def _split_long_section(
        self,
        text: str,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> List[str]:
        """
        将过长文本按自然段落切分，带句子边界对齐和自适应重叠。
        同 section 内连续 chunk 用基础重叠（10%），首个 chunk 用锚点重叠（20%）。
        """
        paragraphs = re.split(r"\n\n+", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        if not paragraphs:
            return [text]

        # 自适应重叠量
        base_overlap = max(4, min(overlap_tokens, int(chunk_tokens * 0.10)))
        anchor_overlap = max(8, min(overlap_tokens * 2, int(chunk_tokens * 0.20)))

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0
        chunk_count = 0

        def finish_chunk(overlap_for_next: int):
            """输出当前积累的 chunk，并为下一个 chunk 保留重叠"""
            nonlocal current, current_len, chunk_count
            chunk_count += 1

            body = "\n\n".join(current)
            chunks.append(body)

            # 句子边界对齐：重叠从最后一个完整句子开始
            target = base_overlap if chunk_count > 1 else anchor_overlap
            actual_overlap = min(overlap_for_next, target)

            overlap_parts: List[str] = []
            overlap_len = 0
            for p in reversed(current):
                pl = self._approx_token_len(p)
                if overlap_len + pl > actual_overlap:
                    # 句子边界对齐：找到第一个句子边界
                    if overlap_parts:
                        first = overlap_parts[0]
                        sent_boundary = re.search(
                            r"(?<=[。！？.!?])\s+", first
                        )
                        if sent_boundary:
                            overlap_parts[0] = first[sent_boundary.end():]
                            if not overlap_parts[0].strip():
                                overlap_parts.pop(0)
                    break
                overlap_parts.insert(0, p)
                overlap_len += pl

            current = overlap_parts
            current_len = overlap_len

        for para in paragraphs:
            para_len = self._approx_token_len(para)

            # 单段超限 → 硬切
            if para_len > chunk_tokens:
                if current:
                    finish_chunk(overlap_tokens)
                hard_chunks = self._hard_split(
                    para, chunk_tokens, overlap_tokens
                )
                chunks.extend(hard_chunks)
                current = []
                current_len = 0
                chunk_count += len(hard_chunks)
                continue

            if current_len + para_len > chunk_tokens and current:
                finish_chunk(overlap_tokens)

            current.append(para)
            current_len += para_len

        if current:
            chunks.append("\n\n".join(current))

        return chunks if chunks else [text]

    def _hard_split(
        self,
        text: str,
        chunk_tokens: int,
        overlap_tokens: int,
    ) -> List[str]:
        """对单个超长段落按句子边界硬切，带重叠"""
        sentences = re.split(r"(?<=[。！？.!?])\s*", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return [text]

        chunks: List[str] = []
        current: List[str] = []
        current_len = 0

        for sent in sentences:
            sent_len = self._approx_token_len(sent)

            if sent_len > chunk_tokens:
                if current:
                    chunks.append("".join(current))
                    current = []
                    current_len = 0
                chunks.extend(self._force_split(sent, chunk_tokens, overlap_tokens))
                continue

            if current_len + sent_len > chunk_tokens:
                chunks.append("".join(current))
                # 句子边界对齐的重叠：用 current 中句子的平均 token 长度计算保留句数
                avg_sent_len = max(
                    sum(self._approx_token_len(s) for s in current) // max(len(current), 1),
                    1,
                )
                overlap_count = max(1, min(len(current), overlap_tokens // avg_sent_len))
                current = current[-overlap_count:] if overlap_count < len(current) else current
                current_len = sum(self._approx_token_len(s) for s in current)

            current.append(sent)
            current_len += sent_len

        if current:
            chunks.append("".join(current))

        return chunks if chunks else [text]

    @staticmethod
    def _force_split(text: str, chunk_tokens: int, overlap_tokens: int = 16) -> List[str]:
        """最后手段：按字符数硬切，带最小重叠窗口"""
        char_limit = int(chunk_tokens * 1.5)
        overlap_chars = max(4, int(overlap_tokens * 1.5))
        if char_limit <= overlap_chars or len(text) <= char_limit:
            return [text]

        chunks = []
        start = 0
        while start < len(text):
            end = min(start + char_limit, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - overlap_chars  # 重叠区域
        return chunks

    # ═══════════════════════════════════════════════════════════════
    #  Token 估算（中英文混合）
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def _approx_token_len(text: str) -> int:
        """
        近似估计 Token 长度，支持中英文混合。

        规则（经验值，基于 GPT 系列 tokenizer）:
          - CJK 字符：约 1.5 token/字（多数常用汉字 1~2 token）
          - 英文单词：约 1.3 token/词
          - 数字/符号：约 1 token/个
          - 空白：不计

        返回估算 token 数（整数）。
        """
        if not text:
            return 0

        cjk_count = 0
        word_count = 0
        symbol_count = 0
        in_word = False

        for ch in text:
            if is_cjk(ch):
                cjk_count += 1
                if in_word:
                    word_count += 1
                    in_word = False
            elif ch.isalpha():
                in_word = True
            elif ch.isdigit():
                symbol_count += 1
                if in_word:
                    word_count += 1
                    in_word = False
            else:
                if in_word:
                    word_count += 1
                    in_word = False
                if not ch.isspace():
                    symbol_count += 1

        if in_word:
            word_count += 1

        return round(cjk_count * 1.5 + word_count * 1.3 + symbol_count * 1.0)

    # ═══════════════════════════════════════════════════════════════
    #  便捷流水线
    # ═══════════════════════════════════════════════════════════════

    def process(
        self,
        path: str,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        adaptive_overlap: bool = True,
    ) -> List[Dict]:
        """
        文档分块流水线：文档 → markdown → 标题分块 → token 分块。

        参数:
            path:             文件或目录路径
            chunk_tokens:     每块最大 token 数
            overlap_tokens:   块间重叠 token 数
            adaptive_overlap: 启用自适应重叠（上下文前缀 + 跨 section 引桥）
        返回:
            最终的 chunk 列表
        """
        markdown = self.convert_to_markdown(path)
        sections = self.split_by_headings(markdown)
        chunks = self.chunk_by_tokens(
            sections, chunk_tokens, overlap_tokens, adaptive_overlap
        )
        return chunks

    # ═══════════════════════════════════════════════════════════════
    #  嵌入 + 检索
    # ═══════════════════════════════════════════════════════════════

    def index(
        self,
        path: str,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        adaptive_overlap: bool = True,
    ) -> int:
        """
        完整索引流水线：分块 → 嵌入 → 存储。

        参数:
            path:             文件或目录路径
            chunk_tokens:     每块最大 token 数
            overlap_tokens:   块间重叠 token 数
            adaptive_overlap: 启用自适应重叠
        返回:
            存储的 chunk 数量
        """
        source_file = Path(path).name if Path(path).is_file() else ""
        chunks = self.process(path, chunk_tokens, overlap_tokens, adaptive_overlap)
        count = self.vector_store.add(chunks, source_file=source_file)

        # 自动提取并缓存文档大纲
        if source_file:
            markdown = self.convert_to_markdown(path)
            outline = self.extract_outline(markdown)
            if outline:
                self._save_outline(source_file, outline)

        return count

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float = 0.0
    ) -> List[Dict]:
        """
        在已索引的文档中检索最相关的 chunks。

        参数:
            query:     查询文本
            top_k:     返回数量
            min_score: 最低相似度阈值
        返回:
            [{"content": ..., "score": ..., "heading_path": ..., ...}, ...]
        """
        return self.vector_store.search(query, top_k=top_k, min_score=min_score)

    # ═══════════════════════════════════════════════════════════════
    #  高级检索：MQE（多查询扩展）+ HyDE（假设文档嵌入）
    # ═══════════════════════════════════════════════════════════════

    def _expand_mqe(self, query: str, n: int) -> List[str]:
        """
        多查询扩展 (MQE)：LLM 生成 N 个语义等价/互补的变体查询。

        参数:
            query: 原始查询文本
            n:     期望生成的变体数量
        返回:
            [原始查询, 变体1, ...] — LLM 调用失败时返回 [原始查询]
        """
        if n <= 0 or not query:
            return [query]

        try:
            from llm import ask_llm

            prompt = [
                {
                    "role": "system",
                    "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。",
                },
                {
                    "role": "user",
                    "content": (
                        f"原始查询：{query}\n"
                        f"请给出{n}个不同表述的查询，每行一个。"
                    ),
                },
            ]
            text = ask_llm(messages=prompt)
            lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
            expanded = [
                ln for ln in lines if ln and ln != query
            ]
            # 去重（保持顺序）
            seen = {query}
            unique = []
            for q in expanded[:n]:
                if q not in seen:
                    seen.add(q)
                    unique.append(q)
            return [query] + unique
        except Exception:
            import traceback
            traceback.print_exc()
            return [query]

    def _generate_hyde(self, query: str) -> Optional[str]:
        """
        假设文档嵌入 (HyDE)：LLM 生成假设答案段落，
        用于代替原始查询进行向量检索。

        参数:
            query: 原始查询文本
        返回:
            假设性答案文本，LLM 调用失败时返回 None
        """
        if not query:
            return None

        try:
            from llm import ask_llm

            prompt = [
                {
                    "role": "system",
                    "content": (
                        "根据用户问题，先写一段可能的答案性段落，"
                        "用于向量检索的查询文档（不要分析过程）。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"问题：{query}\n"
                        "请直接写一段中等长度、客观、包含关键术语的段落。"
                    ),
                },
            ]
            return ask_llm(messages=prompt)
        except Exception:
            import traceback
            traceback.print_exc()
            return None

    def retrieve_expanded(
        self,
        query: str,
        top_k: int = 5,
        enable_mqe: bool = False,
        mqe_expansions: int = 2,
        enable_hyde: bool = False,
        candidate_pool_multiplier: int = 4,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """
        增强检索：支持 MQE 多查询扩展和 HyDE 假设文档嵌入。

        流程：扩展查询 → 各自检索 → 合并去重 → 排序 → Top-K

        参数:
            query:                    原始查询文本
            top_k:                    最终返回数量
            enable_mqe:               启用多查询扩展
            mqe_expansions:           MQE 生成的变体查询数
            enable_hyde:              启用假设文档嵌入
            candidate_pool_multiplier: 候选池倍数（扩大每查询检索数）
            min_score:                最低相似度阈值
        返回:
            [{"id": ..., "content": ..., "score": ..., ...}, ...]
            与 retrieve() 返回格式完全兼容
        """
        if not query:
            return []

        # ── Step 1: 扩展查询 ──────────────────
        expansions: List[str] = [query]

        if enable_mqe:
            mqe_results = self._expand_mqe(query, mqe_expansions)
            # _expand_mqe 已包含原始 query，替换 expansions
            expansions = mqe_results

        if enable_hyde:
            hyde_text = self._generate_hyde(query)
            if hyde_text and hyde_text not in expansions:
                expansions.append(hyde_text)

        # ── Step 2: 各自检索（候选池分配） ────
        total_slots = max(top_k * candidate_pool_multiplier, top_k)
        per_query_k = max(1, total_slots // len(expansions))

        merged: dict = {}  # chunk_id → result (保留最高分)

        for q in expansions:
            results = self.vector_store.search(
                q, top_k=per_query_k, min_score=min_score
            )
            for r in results:
                cid = r["id"]
                if cid not in merged or r["score"] > merged[cid]["score"]:
                    merged[cid] = r

        # ── Step 3: 排序返回 ──────────────────
        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

    # ═══════════════════════════════════════════════════════════════
    #  文档大纲提取与缓存
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def extract_outline(text: str, max_level: int = 3) -> str:
        """
        从 Markdown 文本中提取完整标题层级，构建文档大纲。

        参数:
            text:      Markdown 文本
            max_level: 最大标题层级（默认 H3）

        智能过滤：只保留章/部分/节等结构性标题，跳过内容级标题。
        """
        if not text:
            return ""

        lines = text.splitlines()
        outline_lines = []
        heading_re = re.compile(r"^(#{1,6})\s+(.+)$")
        # 隐式章节模式
        implicit_chapter_re = re.compile(
            r"^(第[一二三四五六七八九十\d]+[章部分节编]|"
            r"[一二三四五六七八九十\d]+[\.、．]\s*)\s*(.+)?"
        )
        # 结构性标题关键词（只有这些才纳入大纲）
        structural_keywords = re.compile(
            r"第[一二三四五六七八九十\d]+[部章节日]|"
            r"^[一二三四五六七八九十]+[、．\s]|"
            r"^\d+[\.、．]\s*\d*|"
            r"前言|目录|附录|参考|总结|展望|习题|致谢"
        )

        for line in lines:
            m = heading_re.match(line)
            if m:
                level = len(m.group(1))
                if level > max_level:
                    continue
                title = m.group(2).strip()
                # 只保留结构性标题
                if not structural_keywords.search(title):
                    continue
                indent = "  " * (level - 1)
                entry = f"{indent}- {title}"
                if not outline_lines or outline_lines[-1] != entry:
                    outline_lines.append(entry)
            elif implicit_chapter_re.match(line.strip()):
                stripped = line.strip()
                if len(stripped) <= 40 and not stripped.startswith("#"):
                    if any('一' <= c <= '鿿' for c in stripped):
                        entry = f"- {stripped}"
                        if not outline_lines or outline_lines[-1] != entry:
                            outline_lines.append(entry)

        return "\n".join(outline_lines)

    def _get_outline_path(self) -> Path:
        """大纲缓存文件路径"""
        return self.knowledge_base_path / "rag_outlines.json"

    def _load_outlines(self) -> dict:
        """加载已保存的所有文档大纲"""
        path = self._get_outline_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    def _save_outline(self, source_file: str, outline: str):
        """保存单个文档大纲（以 source_file 为 key）"""
        outlines = self._load_outlines()
        outlines[source_file] = outline
        path = self._get_outline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(outlines, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def get_all_outlines(self) -> str:
        """
        获取所有已索引文档的大纲，拼接为紧凑的上下文块。
        """
        outlines = self._load_outlines()
        if not outlines:
            return ""

        parts = []
        for source, outline in outlines.items():
            if outline.strip():
                parts.append(f"[{source}]\n{outline}")

        if not parts:
            return ""
        return "## 已索引文档结构\n" + "\n\n".join(parts)
