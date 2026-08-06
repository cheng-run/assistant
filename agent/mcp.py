"""agent/mcp.py — 通过 langchain-mcp-adapters 消费外部 MCP server 工具

本应用是 MCP *客户端*：把外部 MCP server（如官方 mcp-server-langchain）
暴露的工具加载进 agent，让 agent 具备外部工具能力。

配置（.env）：
  MCP_SERVERS_JSON='{"langchain": {"transport": "stdio", "command": "python", "args": ["-m", "mcp_server_langchain"]}}'
未配置 → load_mcp_tools() 返回空列表，应用行为完全不变。
"""

import json
import os
from typing import List

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

_MCP_TOOLS: List[BaseTool] | None = None


async def load_mcp_tools() -> List[BaseTool]:
    """
    一次性加载外部 MCP server 工具（应在 demo.launch() 之前调用一次并缓存）。
    MultiServerMCPClient 是异步上下文管理器，连接生命周期随 async with 结束；
    拿到的工具对象可跨请求复用（Stateless 会话，每调用建连）。
    """
    global _MCP_TOOLS
    if _MCP_TOOLS is not None:
        return _MCP_TOOLS

    cfg = os.getenv("MCP_SERVERS_JSON")
    if not cfg:
        _MCP_TOOLS = []
        return _MCP_TOOLS

    servers = json.loads(cfg)
    async with MultiServerMCPClient(servers) as client:
        _MCP_TOOLS = client.get_tools()
    return _MCP_TOOLS
