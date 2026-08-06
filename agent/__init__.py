"""
agent — LangChain + LangGraph + Deep Agents 编排层

职责边界（重要）：
  - 本包只依赖 rag/ 的 RAGTool 公开方法、memory/ 的 MemoryTool、llm.py 的公开函数；
  - rag/、memory/、llm.py 绝不反向 import 本包（双向依赖会破坏解耦）。
  - 本包把现有 RAG 引擎包装成 agent 可调用的工具，并承载 LangGraph / Deep Agents 运行时。
"""

import warnings

# Python 3.14 下 langchain 的 Pydantic V1 兼容层会产生大量告警噪音，
# 抑制它们（deepagents 官方 CLI 采用相同做法）。
warnings.filterwarnings("ignore", module="langchain_core._api.deprecation")
warnings.filterwarnings("ignore", message=".*Pydantic V1.*", category=UserWarning)

from agent.prompts import SYSTEM_PROMPT, AGENT_SYSTEM_PROMPT, DEEP_AGENT_PROMPT  # noqa: E402
from agent.models import get_chat_model, get_ollama_chat_model  # noqa: E402
from agent.tools import (  # noqa: E402
    rag,
    mem,
    ALL_MODES,
    allowed_modes,
    get_outline,
    list_sources,
    memory_recall,
    memory_remember,
    make_search_documents,
)
from agent.mcp import load_mcp_tools  # noqa: E402

__all__ = [
    "SYSTEM_PROMPT",
    "AGENT_SYSTEM_PROMPT",
    "DEEP_AGENT_PROMPT",
    "get_chat_model",
    "get_ollama_chat_model",
    "rag",
    "mem",
    "ALL_MODES",
    "allowed_modes",
    "get_outline",
    "list_sources",
    "memory_recall",
    "memory_remember",
    "make_search_documents",
    "load_mcp_tools",
]
