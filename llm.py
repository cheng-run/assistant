import os
import threading
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None
_client_lock = threading.Lock()


def _get_client():
    """获取复用的 OpenAI client 单例（线程安全）"""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:  # double-check
                _client = OpenAI(
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                    base_url=os.getenv("DEEPSEEK_BASE_URL"),
                )
    return _client


def ask_llm(messages: list = None, question: str = None, max_retries: int = 3, model: str = None) -> str:
    """调用 LLM，支持传入完整 messages 或简单 question，带自动重试。

    自动检测视觉消息（content 为 list 的格式），使用 deepseek-v4-pro 模型。
    """
    if messages is None:
        if question is None:
            raise ValueError("必须提供 messages 或 question 参数")
        messages = [
            {"role": "system", "content": "你是一个AI小助手，可以回答用户的问题"},
            {"role": "user", "content": question},
        ]

    # 自动检测视觉消息 → 切换模型
    has_vision = any(
        isinstance(m.get("content"), list)
        for m in messages
    )
    model = model or (
        "deepseek-v4-pro" if has_vision else os.getenv("DEEPSEEK_MODEL_NAME")
    )

    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=False,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避：1s → 2s → 4s

    raise last_error


def ask_llm_stream(messages: list, max_retries: int = 3):
    """
    流式调用 LLM，逐 token 产出。支持自动重试。

    用法:
        for token in ask_llm_stream(messages):
            print(token, end="", flush=True)
    """
    client = _get_client()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=os.getenv("DEEPSEEK_MODEL_NAME"),
                messages=messages,
                stream=True,
            )
            for chunk in response:
                content = chunk.choices[0].delta.content
                if content:
                    yield content
            return  # 正常完成
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # 指数退避：1s → 2s → 4s

    raise last_error
