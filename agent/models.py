"""agent/models.py — LLM 模型工厂

主模型走 DeepSeek（OpenAI 兼容接口），与重构前 llm.py 的行为一致；
Ollama 本地模型仅作可选项（默认不用于 agent 主循环）。
嵌入 / 视觉仍由 rag/ 内部走 Ollama，不经过这里。
"""

import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama


def get_chat_model(**kwargs) -> ChatOpenAI:
    """
    DeepSeek 主模型工厂，用于 agent 主循环。

    配置来源 .env：
      DEEPSEEK_MODEL_NAME / DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL
    """
    return ChatOpenAI(
        model=os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=kwargs.get("temperature", 0.2),
        streaming=True,
        max_retries=kwargs.get("max_retries", 3),
    )


def get_ollama_chat_model(model: str = None, **kwargs) -> ChatOllama:
    """
    Ollama 本地对话模型（可选）。

    目前视觉检索返回的是文本描述，agent 无需直接看图片，
    因此默认不使用 ChatOllama；保留供后续扩展（如 agent 接收图像）。
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # .env 里是 /v1（OpenAI 兼容层），ChatOllama 需要原生根地址
    host = base_url.rstrip("/").removesuffix("/v1") or "http://localhost:11434"
    return ChatOllama(
        model=model or os.getenv("OLLAMA_VLM_MODEL", "qwen3-vl:4b"),
        base_url=host,
        temperature=kwargs.get("temperature", 0.2),
    )
