@echo off
REM 智能文档问答助手 启动脚本
REM 用 --env-file 加载 .env（含 PYTHONUTF8=1，解决 Windows 下 docling GBK 崩溃）
cd /d "%~dp0"
echo 启动智能文档问答助手...
echo 请在浏览器打开 http://127.0.0.1:7860
echo 按 Ctrl+C 停止服务
uv run --env-file .env python app.py
pause
