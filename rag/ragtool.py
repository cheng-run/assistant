"""
RAG 文档处理工具

文档 → Markdown → 标题分块 → Token 分块（自适应重叠）→ 向量嵌入 → 存储检索

文本嵌入：Ollama 本地模型（默认 bge-m3，OpenAI 兼容接口，GPU 加速）
图像嵌入：fastembed CLIP（视觉检索路径，保持不变）
"""
import json
import os
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


# ── 表格检测与转换 ────────────────────

def _is_pipe_table(text: str) -> bool:
    """
    检测文本块是否为 Markdown 管道表。

    判断条件:
      1. 至少 2 行
      2. 存在分隔行 (|---|---|)
      3. 所有非空行都以 | 开头
    """
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return False

    # 存在分隔行：包含 |---| 模式（允许冒号对齐标记 :---:）
    has_separator = any(
        re.match(r"^\|[\s\-:]+\|", ln) for ln in lines
    )
    if not has_separator:
        return False

    # 所有非空行以 | 开头
    return all(
        ln.strip().startswith("|") for ln in lines if ln.strip()
    )


def _pipe_table_to_json(text: str) -> str:
    """
    将 Markdown 管道表转为 JSON 对象列表。

    示例:
        输入:
            | 姓名 | 年龄 |
            |------|------|
            | 张三 | 30   |

        输出:
            [{"姓名": "张三", "年龄": "30"}]

    返回:
        JSON 字符串（ensure_ascii=False, indent=2）
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 过掉分隔行 (|---|:---:|---|)
    data_lines = [
        ln for ln in lines
        if not re.match(r"^\|[\s\-:|\s]+\|$", ln)
    ]
    if len(data_lines) < 1:
        return "[]"

    # 解析表头
    headers = [h.strip() for h in data_lines[0].split("|")[1:-1]]
    if not headers:
        return "[]"

    # 解析数据行
    rows: list[dict] = []
    for line in data_lines[1:]:
        cells = [c.strip() for c in line.split("|")[1:-1]]
        # 补齐缺失列（合并单元格可能导致列数不一致）
        while len(cells) < len(headers):
            cells.append("")
        row = {headers[i]: cells[i] for i in range(len(headers))}
        rows.append(row)

    return json.dumps(rows, ensure_ascii=False, indent=2)


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

        # 懒加载文档转换器
        self.__docling_converter = None

        # 懒加载图像嵌入器
        self._image_embedder = None

        # 懒加载知识图谱组件
        self._kg_store = None
        self._kg_extractor = None
        self._kg_detector = None
        self._kg_summarizer = None
        self._kg_retriever = None

    @property
    def embedder(self):
        """文本嵌入器：Ollama 本地模型（默认 bge-m3，OpenAI 兼容接口）"""
        if self._embedder is None:
            from .embedding import OllamaEmbedder
            self._embedder = OllamaEmbedder()
        return self._embedder

    @property
    def vector_store(self):
        if self._vector_store is None:
            from .embedding import VectorStore
            db_path = str(self.knowledge_base_path / "rag_vectors.db")
            self._vector_store = VectorStore(self.embedder, db_path=db_path)
        return self._vector_store

    # ── 文档转换器懒加载 ─────────────────

    def _new_pdf_converter(self, pipeline_options=None):
        """
        创建 Docling PDF 转换器（Windows 兼容配置）。

        修复三个 Windows 下的已知问题:
          1. docling-parse 的 C 扩展资源加载 bug（additional.dat 找不到）
             → 改用 PyPdfiumDocumentBackend 后端
          2. layout 模型加载遇 GBK 编码错误 → 需 PYTHONUTF8=1
          3. torch.compile 需要 MSVC cl 编译器（Windows 通常没有）
             → 通过 DOCLING_INFERENCE_COMPILE_TORCH_MODELS=false 关闭
        """
        # 确保 UTF-8（解决 docling 模型加载在 Windows GBK 编码下的崩溃）
        os.environ["PYTHONUTF8"] = "1"
        # 关闭 torch.compile（Windows 无 MSVC 时 torch.compile 会失败）
        os.environ["DOCLING_INFERENCE_COMPILE_TORCH_MODELS"] = "false"

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

        pipeline_options = pipeline_options or PdfPipelineOptions()
        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                    backend=PyPdfiumDocumentBackend,
                )
            }
        )

    @property
    def _docling_converter(self):
        """懒加载 Docling 文档转换器（PDF 走 PyPdfium 后端）"""
        if self.__docling_converter is None:
            try:
                self.__docling_converter = self._new_pdf_converter()
            except ImportError:
                raise ImportError(
                    "处理文档需要 docling，请运行: uv add docling"
                )
        return self.__docling_converter

    # ── 文档转换缓存 ──────────────────────

    def _get_conversion_cache_path(self, file_path: str) -> Path:
        """获取转换缓存目录"""
        cache_dir = self.knowledge_base_path / "conversion_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    def _convert_with_cache(self, file_path: str) -> tuple:
        """
        带 MD5 缓存的文档转换。
        同一文件只转换一次，结果缓存为 .md 文件。

        返回: (markdown_text, is_cache_hit)
        """
        import hashlib

        fp = Path(file_path)
        # 计算文件 MD5
        file_hash = hashlib.md5(fp.read_bytes()).hexdigest()
        cache_path = self._get_conversion_cache_path(file_path) / f"{file_hash}.md"

        if cache_path.exists():
            return cache_path.read_text(encoding="utf-8"), True

        markdown = self._convert_single(fp)
        cache_path.write_text(markdown, encoding="utf-8")
        return markdown, False

    # ═══════════════════════════════════════════════════════════════
    #  第零步：文档 → Markdown
    # ═══════════════════════════════════════════════════════════════

    def convert_to_markdown(self, path: str) -> str:
        """
        将文档转换为 markdown 文本（带 MD5 缓存）。

        支持格式:
            .md   — 直接读取
            .txt  — 直接读取
            .docx — 需要 docling（可选）
            .pdf  — 需要 docling（可选）

        参数:
            path: 文件路径或目录路径（目录则递归处理所有支持的文件）
        返回:
            markdown 文本（多文件以分隔符连接）
        """
        file_path = Path(path)

        if file_path.is_dir():
            return self._convert_directory(file_path)

        # 单文件：对 .docx/.pdf 使用缓存
        suffix = file_path.suffix.lower()
        if suffix in (".docx", ".pdf"):
            markdown, _ = self._convert_with_cache(path)
            return markdown
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
        """docx → markdown（使用 Docling）"""
        try:
            converter = self._docling_converter
        except ImportError:
            raise ImportError(
                "处理 .docx 需要 docling，请运行: uv add docling"
            )

        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()

    def _convert_pdf(self, file_path: Path) -> str:
        """pdf → markdown（使用 Docling）"""
        try:
            converter = self._docling_converter
        except ImportError:
            raise ImportError(
                "处理 .pdf 需要 docling，请运行: uv add docling"
            )

        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()

    # ── 表格处理 ──────────────────────────

    @staticmethod
    def _process_tables_in_text(text: str) -> str:
        """
        在文本中查找 Markdown 管道表，替换为 JSON 代码块。

        处理方式:
          1. 按 \\n\\n+ 分割段落
          2. 检测每个段落是否为管道表
          3. 是 → 替换为 ```table-json ``` 代码块
          4. 否 → 保持原样

        返回:
            处理后的文本
        """
        # 按段落边界分割，保留分隔符
        parts = re.split(r"(\n\n+)", text)
        result: list[str] = []

        for part in parts:
            if _is_pipe_table(part):
                json_str = _pipe_table_to_json(part)
                result.append(f"```table-json\n{json_str}\n```")
            else:
                result.append(part)

        return "".join(result)

    # ── 图像嵌入器懒加载 ─────────────────

    @property
    def image_embedder(self):
        """懒加载 CLIP 图像嵌入器"""
        if self._image_embedder is None:
            from .embedding import ImageEmbedder
            self._image_embedder = ImageEmbedder()
        return self._image_embedder

    # ── 知识图谱组件懒加载 ──────────────

    @property
    def kg_store(self):
        """懒加载知识图谱存储"""
        if self._kg_store is None:
            from .kg.store import KnowledgeGraphStore
            db_path = str(self.knowledge_base_path / "kg_graph.db")
            self._kg_store = KnowledgeGraphStore(db_path=db_path)
        return self._kg_store

    @property
    def kg_extractor(self):
        """懒加载实体关系提取器"""
        if self._kg_extractor is None:
            from .kg.extractor import EntityRelationExtractor
            self._kg_extractor = EntityRelationExtractor()
        return self._kg_extractor

    @property
    def kg_detector(self):
        """懒加载社区检测器"""
        if self._kg_detector is None:
            from .kg.community import CommunityDetector
            self._kg_detector = CommunityDetector()
        return self._kg_detector

    @property
    def kg_summarizer(self):
        """懒加载社区摘要生成器"""
        if self._kg_summarizer is None:
            from .kg.community import CommunitySummarizer
            self._kg_summarizer = CommunitySummarizer()
        return self._kg_summarizer

    @property
    def kg_retriever(self):
        """懒加载图检索器"""
        if self._kg_retriever is None:
            from .kg.retriever import KnowledgeGraphRetriever
            self._kg_retriever = KnowledgeGraphRetriever(
                store=self.kg_store,
                embedder=self.embedder,
            )
        return self._kg_retriever

    # ═══════════════════════════════════════════════════════════════
    #  视觉页面索引
    # ═══════════════════════════════════════════════════════════════

    def _generate_page_images(self, path: str, dpi: float = 2.0) -> tuple:
        """
        用 Docling 生成每页截图，保存到 data_db/page_images/。

        返回: (page_paths: List[str], doc: DoclingDocument)
              返回 document 对象以避免重复转换
        """
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        # 配置页面图片生成
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_page_images = True
        pipeline_options.images_scale = dpi

        img_converter = self._new_pdf_converter(pipeline_options)
        result = img_converter.convert(path)
        doc = result.document

        # 保存页面图片
        img_dir = self.knowledge_base_path / "page_images"
        img_dir.mkdir(parents=True, exist_ok=True)

        source_name = Path(path).stem
        page_paths = []
        for page_no, page in doc.pages.items():
            if page.image is not None:
                img_path = img_dir / f"{source_name}_p{page_no}.png"
                pil_img = page.image.pil_image  # ImageRef → PIL Image
                pil_img.save(str(img_path), "PNG")
                page_paths.append(str(img_path))

        return page_paths, doc

    @staticmethod
    def _detect_visual_pages_from_doc(doc) -> set:
        """
        从已转换的 Docling document 检测哪些页码包含图表/表格。

        返回: 包含图表的页码集合 {page_no, ...}
        """
        visual_pages = set()
        for item, _ in doc.iterate_items():
            # item.label 是 DocItemLabel 枚举，需取 .value 与字符串比较
            label = getattr(item, "label", None)
            label_str = getattr(label, "value", "") if label is not None else ""
            if label_str not in ("picture", "figure", "table"):
                continue
            prov = getattr(item, "prov", None)
            if not prov:
                continue
            # prov[0] 是 ProvenanceItem（不可哈希），页码在 .page_no
            page_no = getattr(prov[0], "page_no", None)
            if page_no is not None:
                visual_pages.add(page_no)
        return visual_pages

    def _generate_page_description(self, image_path: str) -> str:
        """
        用 DeepSeek v4-pro 解读单页图片，返回中文描述。
        """
        import base64
        from llm import ask_llm

        with open(image_path, "rb") as f:
            b64_data = base64.b64encode(f.read()).decode("utf-8")

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_data}"},
                },
                {
                    "type": "text",
                    "text": (
                        "这张页面经检测包含图表或图片。请详细描述图中内容："
                        "图表类型、坐标轴/图例含义、关键数值、数据趋势与对比关系。"
                        "列出你看到的具体数字和结论，只描述画面内容，不要评价，不要编造不存在的细节。"
                    ),
                },
            ],
        }]
        return ask_llm(messages=messages) or ""

    def _index_visual_pages(self, source_file: str, doc_path: str) -> int:
        """
        索引文档的视觉层：
        1. Docling 生成页面图片（与检测共享一次转换）
        2. CLIP 嵌入每页 → SQLite (modality="page_image")
        3. 含图表页 → DeepSeek 解读 → BGE 嵌入 → SQLite (modality="visual_description")

        返回: 索引的视觉 chunk 数量
        """
        # ── Step 1: 生成页面图片 + 获取 document（一次转换） ──
        try:
            page_paths, doc = self._generate_page_images(doc_path)
        except Exception:
            return 0

        if not page_paths:
            return 0

        # ── Step 2: CLIP 嵌入每页 ──
        page_chunks = []
        for pp in page_paths:
            page_no = Path(pp).stem.rsplit("_p", 1)[-1]
            page_chunks.append({
                "content": f"[Page {page_no} of {source_file}]",
                "heading_path": [],
                "token_count": self._approx_token_len(f"Page {page_no}"),
                "chunk_index": 0,
                "source_file": source_file,
                "modality": "page_image",
                "embedding": self.image_embedder.embed_image(pp),
                # 预置向量的模型标识（VectorStore.add 会按此存 BLOB）
                "model_name": self.image_embedder.model_name,
                "model_dim": self.image_embedder.dim,
            })

        # ── Step 3: 检测含图表页（复用已转换的 doc，无需再转换） ──
        try:
            visual_page_nos = self._detect_visual_pages_from_doc(doc)
        except Exception:
            visual_page_nos = set()

        desc_chunks = []
        for pp in page_paths:
            page_no = Path(pp).stem.rsplit("_p", 1)[-1]
            if int(page_no) in visual_page_nos:
                try:
                    desc = self._generate_page_description(pp)
                    if desc and "纯文字页面" not in desc:
                        desc_chunks.append({
                            "content": f"[{source_file} Page {page_no} 图表描述]\n{desc}",
                            "heading_path": [],
                            "token_count": self._approx_token_len(desc),
                            "chunk_index": 0,
                            "source_file": source_file,
                            "modality": "text",
                        })
                except Exception:
                    pass

        # ── Step 4: 写入 SQLite ──
        count = 0
        if page_chunks:
            count += self.vector_store.add(page_chunks, source_file=source_file)
        if desc_chunks:
            count += self.vector_store.add(desc_chunks, source_file=source_file)

        return count

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
            # 表格预处理：管道表 → JSON
            content = self._process_tables_in_text(content)
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
                chunk = self._enrich_table_chunk({
                    "heading_path": heading_path,
                    "content": full_content,
                    "token_count": self._approx_token_len(full_content),
                    "chunk_index": 0,
                    "cross_section_bridge": is_bridge,
                })
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

                all_chunks.append(self._enrich_table_chunk({
                    "heading_path": heading_path,
                    "content": full_content,
                    "token_count": self._approx_token_len(full_content),
                    "chunk_index": i,
                    "cross_section_bridge": is_bridge,
                }))

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

    # ── 表格元数据 ────────────────────────

    @staticmethod
    def _enrich_table_chunk(chunk: Dict) -> Dict:
        """
        如果 chunk 内容是 JSON 表格块，提取元数据附加到 chunk dict。

        添加字段:
          - is_table: True
          - table_columns: [列名列表]
          - table_rows: 行数
        """
        m = re.search(r"```table-json\n(.*?)\n```", chunk.get("content", ""), re.DOTALL)
        if not m:
            return chunk
        try:
            data = json.loads(m.group(1))
            if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                chunk["is_table"] = True
                chunk["table_columns"] = list(data[0].keys())
                chunk["table_rows"] = len(data)
        except (json.JSONDecodeError, TypeError):
            pass
        return chunk

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
                # JSON 表格块保持完整，不切割
                if para.strip().startswith("```table-json"):
                    chunks.append(para)
                    current = []
                    current_len = 0
                    chunk_count += 1
                    continue
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
        # JSON 表格块不切割（兜底保护）
        if text.strip().startswith("```table-json"):
            return [text]

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
        # JSON 表格块不切割（兜底保护）
        if text.strip().startswith("```table-json"):
            return [text]

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
        enable_visual: bool = False,
    ) -> int:
        """
        完整索引流水线：分块 → 嵌入 → 存储。

        参数:
            path:             文件或目录路径
            chunk_tokens:     每块最大 token 数
            overlap_tokens:   块间重叠 token 数
            adaptive_overlap: 启用自适应重叠
            enable_visual:    启用视觉层索引（页面图片 + CLIP + DeepSeek 解读）
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

        # 视觉层索引
        if enable_visual and Path(path).is_file():
            try:
                visual_count = self._index_visual_pages(source_file, path)
                count += visual_count
            except Exception:
                pass  # 视觉索引失败不影响文本路径

        return count

    def retrieve(
        self, query: str, top_k: int = 5, min_score: float = 0.0,
        enable_visual: bool = False,
    ) -> List[Dict]:
        """
        在已索引的文档中检索最相关的 chunks。

        参数:
            query:         查询文本
            top_k:         返回数量
            min_score:     最低相似度阈值
            enable_visual: 启用视觉检索（CLIP + 图表描述）
        返回:
            [{"content": ..., "score": ..., "heading_path": ..., ...}, ...]
        """
        if not enable_visual:
            return self.vector_store.search(
                query, top_k=top_k, min_score=min_score
            )

        # ── 双路检索 ──
        # 文本路径
        text_results = self.vector_store.search(
            query, top_k=top_k * 2, min_score=min_score,
            modality_filter="text",
        )

        # 视觉路径（CLIP 嵌入查询 → 匹配页面图片）
        try:
            visual_results = self.vector_store.search(
                query, top_k=top_k, min_score=0.0,
                modality_filter="page_image",
                image_embedder=self.image_embedder,
            )
        except Exception:
            visual_results = []

        # ── 融合 ──
        if not visual_results:
            return text_results[:top_k]

        return self._fusion_results(text_results, visual_results, top_k)

    @staticmethod
    def _fusion_results(
        text_results: List[Dict],
        visual_results: List[Dict],
        top_k: int,
    ) -> List[Dict]:
        """
        融合文本检索和视觉检索结果。

        策略：视觉命中的页面 → 提升同 source_file 的文本 chunk 分数。
        """
        # 视觉命中的 source_file
        visual_sources = {v.get("source_file", "") for v in visual_results}

        # 给命中视觉页面的文本结果加分
        fused = {}
        for r in text_results:
            cid = r["id"]
            bonus = 1.15 if r.get("source_file", "") in visual_sources else 1.0
            if cid not in fused or r["score"] * bonus > fused[cid]["score"]:
                r_copy = dict(r)
                r_copy["score"] = round(r["score"] * bonus, 4)
                fused[cid] = r_copy

        # 视觉描述直接加入（DeepSeek 解读的高价值内容）
        # 注：visual_description 在文本检索路径中也会被 BGE 搜到，
        # 因为它们的 embedding 是用 BGE 生成的。这里主要是加分逻辑。
        ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]

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
    #  知识图谱构建
    # ═══════════════════════════════════════════════════════════════

    def build_knowledge_graph(
        self,
        chunks: List[Dict],
        source_file: str,
        source_file_path: str = None,
    ) -> dict:
        """
        从 chunks 构建知识图谱：
        1. 实体/关系提取（LLM）
        2. 实体去重
        3. 写入 SQLite
        4. 社区检测（Louvain）
        5. 社区摘要（LLM）+ 嵌入
        6. chunk-entity 交叉引用

        参数:
            chunks:           chunk_by_tokens() 的输出
            source_file:      来源文件名
            source_file_path: 源文件完整路径（预留，用于未来扩展）
        返回:
            {"entities": N, "relations": M, "communities": C}
        """
        if not chunks:
            return {"entities": 0, "relations": 0, "communities": 0}

        # ── Step 1: 实体/关系提取 ──
        try:
            entities, relations = self.kg_extractor.extract_from_chunks(
                chunks, source_file=source_file
            )
        except Exception:
            return {"entities": 0, "relations": 0, "communities": 0}

        if not entities:
            return {"entities": 0, "relations": 0, "communities": 0}

        # ── Step 2: 写入实体 ──
        entity_ids = self.kg_store.add_entities(
            entities, source_file=source_file
        )

        # ── Step 3: 写入关系 ──
        rel_count = self.kg_store.add_relations(
            relations, source_file=source_file
        )

        # ── Step 4: chunk-entity 交叉引用 ──
        # 构建 name → entity_id 映射
        name_to_id = {}
        for e in entities:
            canonical = e.get("canonical_name", e.get("name", ""))
            row = self.kg_store.get_entity_by_name(canonical)
            if row:
                name_to_id[canonical] = row["id"]

        refs = []
        for e in entities:
            eid = name_to_id.get(e.get("canonical_name", e.get("name", "")))
            if eid:
                for chunk_idx in e.get("chunk_refs", []):
                    if chunk_idx < len(chunks):
                        chunk_id = chunks[chunk_idx].get("id", chunk_idx)
                        refs.append((chunk_id, eid))

        if refs:
            self.kg_store.add_chunk_entity_refs(refs, source_file=source_file)

        # ── Step 5: 社区检测 ──
        adjacency = self.kg_store.build_adjacency(source_file=source_file)
        communities = []
        if adjacency:
            communities = self.kg_detector.detect(adjacency)

        # ── Step 6: 社区摘要 + 嵌入 ──
        if communities:
            entity_map = self.kg_store.get_entities(source_file=source_file)
            communities = self.kg_summarizer.summarize_all(
                communities, entity_map, self.kg_store, self.embedder
            )

        # ── Step 7: 保存社区 ──
        comm_count = self.kg_store.save_communities(
            communities, source_file=source_file
        )

        return {
            "entities": len(entity_ids),
            "relations": rel_count,
            "communities": comm_count,
        }

    # ═══════════════════════════════════════════════════════════════
    #  KG 增强检索
    # ═══════════════════════════════════════════════════════════════

    def retrieve_with_kg(
        self,
        query: str,
        top_k: int = 3,
        enable_visual: bool = False,
    ) -> tuple:
        """
        双路检索融合：向量检索 + 知识图谱检索。

        参数:
            query:          查询文本
            top_k:          向量检索返回数
            enable_visual:  启用视觉检索
        返回:
            (kg_context: str, vector_chunks: List[Dict])
        """
        from .kg.retriever import fuse_vector_and_graph

        # ── 路径 1: 向量检索 ──
        vector_results = self.retrieve(
            query, top_k=top_k, enable_visual=enable_visual
        )

        # ── 路径 2: KG 图检索 ──
        kg_results = []
        try:
            kg_results = self.kg_retriever.retrieve(query, top_k=2)
        except Exception:
            kg_results = []

        # ── 融合 ──
        kg_context, chunks = fuse_vector_and_graph(
            vector_results, kg_results, top_k_vector=top_k
        )

        # ── 图谱证据跳转：命中实体 → 关联 chunk 原文（kg_chunk_entity_refs 接线） ──
        try:
            evidence = self._kg_evidence(query, limit=3)
            if evidence:
                kg_context = (kg_context + "\n\n" + evidence) if kg_context else evidence
        except Exception:
            pass

        return kg_context, chunks

    def _kg_evidence(self, query: str, limit: int = 3) -> str:
        """图谱 → 证据 chunk 跳转：返回命中实体关联的原文片段。

        复用 kg_chunk_entity_refs（build_knowledge_graph 已写入）与
        VectorStore.get_by_ids 取原文，让图谱回答有据可依。
        """
        try:
            _, chunk_ids, _ = self.kg_retriever.retrieve_entities_and_chunks(
                query, top_k_entities=5
            )
        except Exception:
            return ""
        if not chunk_ids:
            return ""
        chunks = self.vector_store.get_by_ids(chunk_ids[:limit])
        if not chunks:
            return ""
        parts = ["## 相关原文证据（来自知识图谱关联）"]
        for c in chunks:
            heading = " > ".join(c.get("heading_path", [])) or c.get("source_file", "文档")
            parts.append(f"[{heading}]\n{c['content'][:200]}")
        return "\n\n".join(parts)

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

        只返回向量库中真实存在的 source —— 即使大纲缓存残留了已删除文档的
        旧条目，也不会再注入（清空文档后不会继续出现"幽灵"大纲）。
        """
        outlines = self._load_outlines()
        if not outlines:
            return ""

        try:
            active = set(self.vector_store.list_sources())
        except Exception:
            active = set(outlines.keys())

        parts = []
        for source, outline in outlines.items():
            if source in active and outline.strip():
                parts.append(f"[{source}]\n{outline}")

        if not parts:
            return ""
        return "## 已索引文档结构\n" + "\n\n".join(parts)

    def clear_outlines(self) -> None:
        """清空全部大纲缓存（配合向量库清空，避免残留旧文档结构）。"""
        path = self._get_outline_path()
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass

    def remove_outline(self, source_file: str) -> None:
        """从大纲缓存移除单个文档条目。"""
        outlines = self._load_outlines()
        if source_file not in outlines:
            return
        del outlines[source_file]
        path = self._get_outline_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(outlines, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
