"""launch.py — 一键启动智能文档问答助手（替代 .bat，跨平台无编码问题）

用法（在项目根目录）：
    uv run --env-file .env python launch.py              # 生产：构建前端 + 启动 + 自动打开浏览器
    uv run --env-file .env python launch.py --no-browser # 不自动打开浏览器

说明：
  - `uv run --env-file .env` 负责把 PYTHONUTF8 / DOCLING_* 等变量在解释器启动前注入
    （解决 Windows 中文系统下 docling 的 GBK 崩溃），因此必须用这个命令启动。
  - 生产模式由 FastAPI 直接托管 frontend/dist；首次启动会自动 npm install + build。
  - 开发模式（改前端热更新）请另开终端执行：cd frontend && npm run dev，访问 :5173。
"""

import os
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

BASE = Path(__file__).resolve().parent
DIST = BASE / "frontend" / "dist"
PORT = int(os.getenv("PORT", "8000"))


def build_frontend() -> None:
    """前端 dist 缺失时自动构建（仅首次）。"""
    if (DIST / "index.html").exists():
        print("[launch] 前端已构建，跳过构建步骤")
        return
    print("[launch] 首次启动，构建前端（npm install && npm run build）...")
    if subprocess.run(["npm", "install"], cwd=BASE / "frontend", shell=True).returncode != 0:
        sys.exit("❌ npm install 失败，请检查 npm 是否可用")
    if subprocess.run(["npm", "run", "build"], cwd=BASE / "frontend", shell=True).returncode != 0:
        sys.exit("❌ npm run build 失败")


def main() -> None:
    build_frontend()
    if "--no-browser" not in sys.argv:
        threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}")).start()
        print(f"[launch] 已启动，正在打开浏览器 http://127.0.0.1:{PORT} （Ctrl+C 停止）")
    else:
        print(f"[launch] 已启动 http://127.0.0.1:{PORT} （Ctrl+C 停止）")

    import uvicorn

    uvicorn.run("server.main:app", host="127.0.0.1", port=PORT)


if __name__ == "__main__":
    main()
