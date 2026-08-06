import base64
import json
import os
import threading
import time
import urllib.request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_client = None
_client_lock = threading.Lock()

_ollama_client = None
_ollama_client_lock = threading.Lock()


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


def _get_ollama_client():
    """获取复用的 Ollama OpenAI 兼容 client 单例（线程安全）"""
    global _ollama_client
    if _ollama_client is None:
        with _ollama_client_lock:
            if _ollama_client is None:  # double-check
                _ollama_client = OpenAI(
                    api_key="ollama",  # Ollama 忽略 key，占位即可
                    base_url=os.getenv(
                        "OLLAMA_BASE_URL", "http://localhost:11434/v1"
                    ),
                )
    return _ollama_client


def _ask_ollama_vlm(messages: list, model: str, max_retries: int = 3) -> str:
    """
    调用 Ollama 原生 /api/chat 接口（视觉消息）。

    Ollama 的 OpenAI 兼容 /v1/chat/completions 不支持 OpenAI 的 image_url 消息格式
    （会报 "unknown variant image_url"），原生 /api/chat 用 images 数组传 base64。
    这里把 OpenAI 风格消息转成 Ollama 原生格式。
    """
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    # /v1 是 OpenAI 兼容端点，原生 API 是 /api/chat（去掉 /v1 后缀）
    host = base_url.rstrip("/").removesuffix("/v1") or "http://localhost:11434"
    url = f"{host}/api/chat"

    last_error = None
    for attempt in range(max_retries):
        try:
            # 组装 Ollama 原生消息：content 为纯文本，图片转 images 数组
            ollama_messages = []
            images_by_turn: list[list[str]] = []

            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    # OpenAI 视觉消息：抽文本 + 抽 base64 图片
                    text_parts = []
                    imgs = []
                    for part in content:
                        if part.get("type") == "image_url":
                            url_data = part.get("image_url", {}).get("url", "")
                            if url_data.startswith("data:"):
                                # data:image/png;base64,xxxx
                                imgs.append(url_data.split(",", 1)[1])
                        elif part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    ollama_messages.append({
                        "role": m.get("role", "user"),
                        "content": "\n".join(text_parts),
                    })
                    images_by_turn.append(imgs)
                else:
                    ollama_messages.append({
                        "role": m.get("role", "user"),
                        "content": content or "",
                    })
                    images_by_turn.append([])

            body = {
                "model": model,
                "messages": ollama_messages,
                "images": images_by_turn,
                "stream": False,
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data.get("message", {}).get("content", "") or ""
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    raise last_error


def ask_llm(messages: list = None, question: str = None, max_retries: int = 3, model: str = None) -> str:
    """调用 LLM，支持传入完整 messages 或简单 question，带自动重试。

    自动检测视觉消息（content 为 list 的格式）：
      - 配置了 OLLAMA_VLM_MODEL → 路由到本地 Ollama 视觉模型（如 qwen VL）
      - 未配置 → 使用 DEEPSEEK_MODEL_NAME（当前为 flash）
    纯文本消息始终走 DeepSeek。
    """
    if messages is None:
        if question is None:
            raise ValueError("必须提供 messages 或 question 参数")
        messages = [
            {"role": "system", "content": "你是一个AI小助手，可以回答用户的问题"},
            {"role": "user", "content": question},
        ]

    # 自动检测视觉消息 → 路由到 Ollama VL / DeepSeek 视觉模型
    has_vision = any(
        isinstance(m.get("content"), list)
        for m in messages
    )
    vlm_model = os.getenv("OLLAMA_VLM_MODEL") or ""
    use_ollama_vlm = has_vision and bool(vlm_model)

    if model is None:
        if use_ollama_vlm:
            model = vlm_model
        elif has_vision:
            model = os.getenv("DEEPSEEK_MODEL_NAME")  # 与纯文本对话统一（当前为 flash）
        else:
            model = os.getenv("DEEPSEEK_MODEL_NAME")

    # 视觉消息走 Ollama 原生接口（OpenAI 兼容层不支持 image_url）
    if use_ollama_vlm:
        try:
            return _ask_ollama_vlm(messages, model, max_retries)
        except Exception as e:
            raise RuntimeError(
                f"Ollama 视觉模型调用失败（model={vlm_model}）。"
                f"请确认已运行 'ollama serve' 且已拉取模型 'ollama pull {vlm_model}'。"
            ) from e

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
