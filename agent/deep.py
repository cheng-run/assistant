"""agent/deep.py — Deep Agents harness（规划 / 子代理 / 上下文压缩 / 跨会话记忆）

在 LangGraph 之上叠加 create_deep_agent：
  - write_todos 规划（TodoListMiddleware，deepagents 0.7 起 opt-in）
  - task 子代理委派（默认 general-purpose 子代理）
  - 自动摘要压缩长对话
  - store 跨会话记忆（可选，传 InMemoryStore 点亮）

检索策略由 router.py 自动路由，工具全模式开放；AGENT_MODE=deep 只是
service.py 里的一行切换。
"""

from typing import List, Optional, Sequence

from deepagents import create_deep_agent
from langchain.agents.middleware.todo import TodoListMiddleware
from langchain_core.tools import BaseTool

from agent.models import get_chat_model
from agent.prompts import DEEP_AGENT_PROMPT
from agent.router import web_search
from agent.tools import (
    get_outline,
    list_sources,
    make_search_documents,
    memory_recall,
    memory_remember,
)


def build_deep_agent(
    include_rag: bool = True,
    extra_tools: Optional[Sequence[BaseTool]] = None,
    checkpointer=None,
    store=None,
):
    """构建 Deep Agents harness（规划 + 子代理 + 跨会话记忆）。"""
    tools: List[BaseTool] = []
    if include_rag:
        tools.append(make_search_documents())  # 全模式开放
        tools += [get_outline, list_sources]
    tools += [memory_recall, memory_remember, web_search]
    if extra_tools:
        tools += list(extra_tools)

    return create_deep_agent(
        model=get_chat_model(),
        tools=tools,
        system_prompt=DEEP_AGENT_PROMPT,
        middleware=[TodoListMiddleware()],  # 规划工具 write_todos
        checkpointer=checkpointer,
        store=store,  # 跨会话记忆（可选；对话召回仍用自定义 memory/）
    )
