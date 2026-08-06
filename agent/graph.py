"""agent/graph.py — LangGraph agent 运行时

基于 langchain.agents.create_agent（LangGraph 之上的一层 agent 抽象）：
模型 + 工具 + 系统提示 → ReAct 循环（agent 节点 ↔ tools 节点），
可挂 checkpointer 做多轮持久化，支持 astream 流式。

开关 → 工具集映射：
  - RAG 关：整个移除 search_documents（agent 退化为纯聊天）
  - KG 开：允许 search_documents 的 mode="kg"
  - 视觉开：允许 mode="visual"
"""

import os
from typing import List, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from agent.models import get_chat_model
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools import (
    ALL_MODES,
    allowed_modes,
    get_outline,
    list_sources,
    make_search_documents,
    memory_recall,
    memory_remember,
)

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_db")


def build_agent(
    include_rag: bool = True,
    include_kg: bool = True,
    include_visual: bool = True,
    extra_tools: Optional[Sequence[BaseTool]] = None,
    checkpointer=None,
):
    """按开关组合构建 LangGraph agent。"""
    tools: List[BaseTool] = []
    if include_rag:
        tools.append(make_search_documents(allowed_modes(include_kg, include_visual)))
        tools += [get_outline, list_sources]
    tools += [memory_recall, memory_remember]
    if extra_tools:
        tools += list(extra_tools)

    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
