"""
实体与关系提取器 — 使用 LLM 从文档片段中提取知识三元组

提取流程:
  1. 将 chunks 按 batch_size 分批
  2. 每批调用 LLM 做结构化 JSON 提取
  3. 跨批次合并与实体去重（Entity Resolution）
  4. 返回 (entities, relations)
"""
import json
import re
from typing import Dict, List, Tuple

import numpy as np


# ── 实体/关系类型别名 ──────────────────
EntityDict = Dict        # {"name": ..., "canonical_name": ..., "entity_type": ..., "description": ..., "chunk_refs": [...]}
RelationDict = Dict      # {"source": ..., "target": ..., "relation": ..., "description": ..., "chunk_refs": [...]}


# ── 实体类型定义 ──────────────────────
_VALID_ENTITY_TYPES = {
    "PERSON", "ORGANIZATION", "CONCEPT", "LOCATION",
    "EVENT", "PRODUCT", "TECH", "DATE", "OTHER",
}

# ── 提取 Prompt 模板 ─────────────────

_EXTRACTION_SYSTEM = """你是知识图谱构建专家。你的任务是阅读文档片段，从中提取实体和关系。

实体类型定义：
- PERSON: 人物（个人、团队、角色）
- ORGANIZATION: 组织（公司、机构、部门、团队）
- CONCEPT: 概念（技术术语、方法论、抽象概念）
- LOCATION: 地点（地名、地址、区域）
- EVENT: 事件（会议、发布、里程碑）
- PRODUCT: 产品（软件、硬件、服务、成果）
- TECH: 技术（编程语言、框架、协议、标准）
- DATE: 日期（重要时间点、版本号）
- OTHER: 其他重要实体

要求：
1. 只提取文档中明确提及的实体
2. 每个实体给出简短描述（15字以内）
3. 关系描述用简洁的动宾结构（如"开发了"、"属于"、"包含"、"提出"、"集成"）
4. 只提取有意义的关系，不要过度提取
5. 注意识别实体的不同称呼，尽量使用最规范的名称"""

_EXTRACTION_USER = """从以下文档片段中提取实体和关系。

文档片段：
---
{chunks_text}
---

请以JSON格式返回（只返回JSON，不要其他文字）：
{{
  "entities": [
    {{"name": "实体名称", "type": "实体类型", "description": "简短描述"}}
  ],
  "relations": [
    {{"source": "源实体名称", "target": "目标实体名称", "relation": "关系描述"}}
  ]
}}

注意：
- source 和 target 必须与 entities 中的 name 完全一致
- 如果没有实体或关系，返回空列表 []
- 关系方向要准确：A 包含 B，不是 B 包含 A"""


class EntityRelationExtractor:
    """
    LLM 驱动的实体/关系联合提取器。

    用法:
        extractor = EntityRelationExtractor(batch_size=4)
        entities, relations = extractor.extract_from_chunks(chunks, "doc.pdf")
    """

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self._resolver = EntityResolver()

    def extract_from_chunks(
        self, chunks: List[Dict], source_file: str = ""
    ) -> Tuple[List[EntityDict], List[RelationDict]]:
        """
        从 chunks 中提取实体和关系。

        参数:
            chunks:      文档 chunk 列表（至少包含 "content" 键）
            source_file: 来源文件名
        返回:
            (entities, relations) — 去重后的实体列表和关系列表
        """
        if not chunks:
            return [], []

        # ── 分批提取 ──
        all_entities = []
        all_relations = []

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            try:
                ents, rels = self._extract_batch(batch)
                all_entities.extend(ents)
                all_relations.extend(rels)
            except Exception:
                # 单批失败不影响其他批次
                continue

        if not all_entities:
            return [], []

        # ── 实体去重 ──
        resolved_entities = self._resolver.resolve(all_entities)

        # ── 重命名关系中的实体引用 ──
        # 构建 原始名 → 规范名 的映射
        name_map = {}
        for e in resolved_entities:
            canonical = e.get("canonical_name", e.get("name", ""))
            name_map[e.get("name", "")] = canonical

        resolved_relations = []
        for r in all_relations:
            src = r.get("source", "")
            tgt = r.get("target", "")
            if src in name_map:
                r["source"] = name_map[src]
            if tgt in name_map:
                r["target"] = name_map[tgt]
            # 过滤自环
            if r["source"] != r["target"]:
                resolved_relations.append(r)

        return resolved_entities, resolved_relations

    def _extract_batch(
        self, batch_chunks: List[Dict]
    ) -> Tuple[List[EntityDict], List[RelationDict]]:
        """
        对一批 chunks 调用 LLM 提取实体和关系。
        """
        # ── 拼接 chunks 文本 ──
        parts = []
        for idx, c in enumerate(batch_chunks):
            heading = " > ".join(c.get("heading_path", [])) or "文档"
            parts.append(f"[片段 {idx + 1}] [{heading}]\n{c.get('content', '')}")
        chunks_text = "\n\n---\n\n".join(parts)

        # ── 调用 LLM ──
        from llm import ask_llm

        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM},
            {"role": "user", "content": _EXTRACTION_USER.format(chunks_text=chunks_text)},
        ]

        # 最多重试 2 次
        for attempt in range(2):
            try:
                response = ask_llm(messages=messages, max_retries=1)
                parsed = self._parse_llm_response(response or "")
                if parsed:
                    entities = parsed.get("entities", [])
                    relations = parsed.get("relations", [])

                    # 标准化实体字段
                    for e in entities:
                        if "canonical_name" not in e:
                            e["canonical_name"] = EntityResolver.normalize(
                                e.get("name", "")
                            )
                        if e.get("type", "") not in _VALID_ENTITY_TYPES:
                            e["type"] = "OTHER"

                    return entities, relations

            except Exception:
                if attempt < 1:
                    # 重试：追加更严格的格式要求
                    messages.append({
                        "role": "user",
                        "content": "请确保只返回有效的JSON格式。如果没有实体或关系，返回 {\"entities\": [], \"relations\": []}",
                    })

        return [], []

    @staticmethod
    def _parse_llm_response(text: str) -> dict:
        """
        解析 LLM 返回的 JSON。处理常见格式问题：
        1. 纯 JSON
        2. Markdown 代码块包裹 (```json ... ```)
        3. 前导/后置文字
        4. 尾部逗号
        """
        if not text:
            return {}

        text = text.strip()

        # 尝试提取 markdown 代码块
        code_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if code_match:
            text = code_match.group(1).strip()

        # 尝试找到 JSON 对象边界
        if not text.startswith("{"):
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1:
                text = text[start:end + 1]

        # 移除尾部逗号
        text = re.sub(r",\s*([}\]])", r"\1", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}


class EntityResolver:
    """实体去重与规范化"""

    @staticmethod
    def normalize(name: str) -> str:
        """
        规范化实体名以进行比对。

        操作:
          1. 去除首尾空白
          2. 全角转半角
          3. ASCII 字母小写
          4. 标准化空白
        """
        if not name:
            return ""

        name = name.strip()

        # 全角转半角
        result = []
        for ch in name:
            cp = ord(ch)
            if 0xFF01 <= cp <= 0xFF5E:      # 全角标点
                result.append(chr(cp - 0xFEE0))
            elif cp == 0x3000:               # 全角空格
                result.append(" ")
            elif 0x41 <= cp <= 0x5A:         # 大写 ASCII → 小写
                result.append(chr(cp + 32))
            else:
                result.append(ch)

        name = "".join(result)
        # 合并多余空白
        name = re.sub(r"\s+", " ", name).strip()

        return name

    def resolve(self, entities: List[EntityDict]) -> List[EntityDict]:
        """
        合并重复实体。

        策略:
          1. 精确 canonical_name 匹配 → 合并
          2. 同一 base name（去掉括号等修饰）→ 合并
          3. 合并时保留出现次数最多的 name、最长的 description、
             合并 chunk_refs
        """
        if not entities:
            return []

        # ── Step 1: 按 canonical_name 分组 ──
        groups: Dict[str, List[EntityDict]] = {}
        for e in entities:
            canonical = e.get("canonical_name", self.normalize(e.get("name", "")))
            if canonical not in groups:
                groups[canonical] = []
            groups[canonical].append(e)

        # ── Step 2: 合并每组 ──
        resolved = []
        for canonical, group in groups.items():
            if len(group) == 1:
                resolved.append(group[0])
                continue

            # 取最好的 name（最长的，往往是全称）
            best = max(group, key=lambda x: len(x.get("name", "")))
            # 合并 chunk_refs
            all_refs = []
            for e in group:
                all_refs.extend(e.get("chunk_refs", []))
            unique_refs = list(dict.fromkeys(all_refs))  # 去重保序

            merged = {
                "name": best.get("name", canonical),
                "canonical_name": canonical,
                "entity_type": best.get("entity_type", group[0].get("entity_type", "OTHER")),
                "description": best.get("description", ""),
                "chunk_refs": unique_refs,
            }
            resolved.append(merged)

        # ── Step 3: 模糊去重（可选，基于 embedding 相似度） ──
        # 目前仅做精确去重，避免过度合并

        return resolved
