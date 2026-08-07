"""agent/graph.py — LangGraph agent 运行时

基于 langchain.agents.create_agent（LangGraph 之上的一层 agent 抽象）：
模型 + 工具 + 系统提示 → ReAct 循环（agent 节点 ↔ tools 节点），
可挂 checkpointer 做多轮持久化，支持 astream 流式。

工具集映射（检索策略已由 router.py 自动路由，不再由开关决定）：
  - search_documents 全模式开放（vector/kg/visual/expanded），agent 按问题自由组合
  - get_outline / list_sources / web_search / memory_recall / memory_remember
"""

import os
from typing import List, Optional, Sequence

from langchain.agents import create_agent
from langchain_core.tools import BaseTool

from agent.models import get_chat_model
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.router import web_search
from agent.tools import (
    get_outline,
    list_sources,
    make_search_documents,
    memory_recall,
    memory_remember,
)

DEFAULT_DB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data_db")


def build_agent(
    include_rag: bool = True,
    extra_tools: Optional[Sequence[BaseTool]] = None,
    checkpointer=None,
):
    """构建 LangGraph agent（检索策略由 router 自动路由，工具全模式开放）。"""
    tools: List[BaseTool] = []
    if include_rag:
        tools.append(make_search_documents())  # 全模式开放，agent 按需选/组合
        tools += [get_outline, list_sources]
    tools += [memory_recall, memory_remember, web_search]
    if extra_tools:
        tools += list(extra_tools)

    return create_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )
