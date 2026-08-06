"""tests/run_golden.py — 金集回归运行器

在项目根目录运行（Ollama 需已启动；调 LLM 需要 DeepSeek 密钥）：

  # 只测检索 P@3（不调 LLM，快速冒烟）
  uv run --env-file .env python -m tests.run_golden --retrieve-only

  # 跑旧路径 / LangGraph agent / Deep Agents 三路回答 + 延迟
  uv run --env-file .env python -m tests.run_golden --paths legacy,reactive,deep

  # 加 LLM 裁判对回答打分（额外 DeepSeek 调用）
  uv run --env-file .env python -m tests.run_golden --paths reactive --judge --limit 5

指标：
  - 检索 P@3：seed 检索 top-3 片段的标题路径命中 expected_headings 的比例
  - 回答忠实度：LLM-as-judge 按金集预期事实对回答打分 1-5
  - 延迟：端到端耗时（秒）
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # 保证 `python tests/run_golden.py` 也能跑

GOLDEN_FILE = ROOT / "tests" / "golden.jsonl"


def load_golden() -> list:
    items = []
    with open(GOLDEN_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# ═══════════════════════════════════════════════════════════════
#  检索 P@3（确定性 seed 检索，OLD/NEW 共用）
# ═══════════════════════════════════════════════════════════════

def _retrieval_p3(item: dict) -> float:
    from tests.legacy_on_message import build_legacy_messages

    mode = item.get("mode", "vector")
    visual = mode == "visual"
    kg = mode == "kg"
    _, chunks = build_legacy_messages(item["question"], [], True, visual, kg)
    top = chunks[:3]
    if not top:
        return 0.0
    expected = item.get("expected_headings", [])
    hits = 0
    for c in top:
        head = " > ".join(c.get("heading_path", [])) or "文档"
        if head in expected:
            hits += 1
    return hits / min(3, len(top))


# ═══════════════════════════════════════════════════════════════
#  回答路径
# ═══════════════════════════════════════════════════════════════

def _old_answer(item: dict) -> str:
    from tests.legacy_on_message import legacy_stream_tokens

    mode = item.get("mode", "vector")
    visual = mode == "visual"
    kg = mode == "kg"
    return "".join(legacy_stream_tokens(item["question"], [], True, visual, kg))


async def _agent_answer(item: dict, deep: bool) -> str:
    from agent.service import stream_chat

    os.environ["AGENT_MODE"] = "deep" if deep else "reactive"
    mode = item.get("mode", "vector")
    visual = mode == "visual"
    kg = mode == "kg"
    rows = []
    async for row in stream_chat(item["question"], [], True, visual, kg):
        rows.append(row)
    # 取最后一个 assistant 消息（完整答案）
    for msg in reversed(rows[-1]):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def _judge(item: dict, answer: str) -> int | None:
    """LLM-as-judge：按金集预期事实对回答打分 1-5。"""
    from llm import ask_llm

    prompt = (
        "你是一个严格的答案裁判。请评估回答是否忠实于文档事实、正确且完整。\n"
        f"问题：{item['question']}\n"
        f"金集预期事实：{item.get('notes', '')}\n"
        f"模型回答：{answer}\n"
        "只输出一个 1-5 的整数分数（5=完全正确且忠实，1=完全错误或编造）。"
    )
    try:
        resp = ask_llm([{"role": "user", "content": prompt}])
        m = re.search(r"[1-5]", resp)
        return int(m.group()) if m else None
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════
#  主流程
# ═══════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(description="金集回归运行器")
    parser.add_argument("--paths", default="reactive",
                        help="要跑的回答路径，逗号分隔: legacy,reactive,deep")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    parser.add_argument("--judge", action="store_true", help="启用 LLM 裁判打分")
    parser.add_argument("--retrieve-only", action="store_true",
                        help="只测检索 P@3，不调用 LLM")
    args = parser.parse_args()

    items = load_golden()
    if args.limit > 0:
        items = items[:args.limit]
    print(f"加载金集 {len(items)} 条: {GOLDEN_FILE}\n")

    paths = [p.strip() for p in args.paths.split(",") if p.strip()]
    report = []

    # 检索 P@3（确定性，先算）
    p3 = {it["id"]: _retrieval_p3(it) for it in items}
    avg_p3 = sum(p3.values()) / len(p3) if p3 else 0.0

    if args.retrieve_only:
        print(f"{'ID':<4}{'类型':<12}{'P@3':>6}")
        for it in items:
            print(f"{it['id']:<4}{it.get('bucket','-'):<12}{p3[it['id']]:>6.2f}")
        print(f"\n平均检索 P@3: {avg_p3:.2f}")
        return

    for path in paths:
        if path not in ("legacy", "reactive", "deep"):
            print(f"跳过未知路径: {path}")
            continue
        print(f"\n===== 路径: {path} =====")
        scores = []
        for it in items:
            q = it["question"]
            t0 = time.time()
            if path == "legacy":
                answer = _old_answer(it)
            else:
                answer = await _agent_answer(it, deep=(path == "deep"))
            elapsed = time.time() - t0

            score = "-"
            if args.judge:
                score = _judge(it, answer)
                if score is not None:
                    scores.append(score)

            judge_str = f"{score}" if score != "-" else "  -"
            print(f"[{it['id']}] ({it.get('bucket','-'):<10}) {elapsed:5.1f}s 判{judge_str} "
                  f"P@3={p3[it['id']]:.2f} | {q[:40]}")
            print(f"     答案: {answer[:120].strip()!r}")

        if scores:
            print(f"\n{path} 平均裁判分: {sum(scores)/len(scores):.2f} (n={len(scores)})")

    print(f"\n平均检索 P@3（全部路径共享 seed）: {avg_p3:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
