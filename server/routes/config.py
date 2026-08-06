"""server/routes/config.py — GET /api/config（只读，供前端徽章展示）"""

import os

from fastapi import APIRouter

router = APIRouter()


@router.get("/config")
async def get_config():
    return {
        "agent_mode": os.getenv("AGENT_MODE", "reactive"),
        "checkpoint_enabled": os.getenv("ENABLE_CHECKPOINT", "0") == "1",
        "model": os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v4-flash"),
        "use_langchain": os.getenv("USE_LANGCHAIN", "0") == "1",
    }
