from memory.base import extract_text
from memory.tool import MemoryTool
from llm import ask_llm

SYSTEM_PROMPT = "你是一个AI小助手，可以回答用户的问题"


class QAAssistant:
    """带双层记忆的问答助手：短期对话流 + 长期用户信息，记忆维护全自动"""

    def __init__(self, max_history: int = 20):
        self.memory = MemoryTool(working_capacity=max_history)
        self.max_history = max_history
        self._msg_count = 0  # 消息计数器，用于节流 auto_maintain

    def ask(self, question: str, importance: float = 0.5) -> str:
        # ── 构建 messages ────────────────────
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # 注入相关长期记忆（情景记忆中与当前问题相关的用户信息）
        episodic_context = self.memory.get_relevant_context(question, limit=3)
        if episodic_context:
            facts = "\n".join(
                f"- {self._format_context(c)}" for c in episodic_context
            )
            messages.append({
                "role": "system",
                "content": f"以下是关于用户的已知信息，可以在回答时参考：\n{facts}",
            })

        # 注入短期对话历史（最近几轮对话）
        history = self.memory.get_working()
        for item in history[-self.max_history:]:
            messages.append(item["content"])

        # 当前问题
        messages.append({"role": "user", "content": question})

        # ── 存储用户问题（自动路由） ─────────
        self.memory.add(
            {"role": "user", "content": question},
            importance=importance,
        )

        # ── 调用 LLM ─────────────────────────
        answer = ask_llm(messages=messages)

        # ── 存储助手回答 ─────────────────────
        self.memory.add(
            {"role": "assistant", "content": answer},
            importance=0.3,  # 助手回答默认低重要性
        )

        # ── 自动记忆维护：每 5 条消息执行一次整合 + 遗忘 ────────
        self._msg_count += 1
        if self._msg_count % 5 == 0:
            self.memory.auto_maintain()

        return answer

    def recall(self, keyword: str) -> list:
        """跨记忆层搜索"""
        return self.memory.search(keyword)

    def remember(self, fact: str):
        """强制存入长期情景记忆"""
        self.memory.add(fact, importance=0.9, memory_type="episodic")

    def clear_memory(self):
        """清空全部记忆"""
        self.memory.clear_all()

    @property
    def stats(self) -> dict:
        """查看记忆统计"""
        return self.memory.stats

    # ── 内部 ──────────────────────────────

    @staticmethod
    def _format_context(memory_item: dict) -> str:
        """格式化一条记忆为可读文本"""
        return extract_text(memory_item.get("content", ""))


if __name__ == "__main__":
    assistant = QAAssistant()

    # 测试：存入个人信息（高重要性 → 自动进长期记忆）
    print(assistant.ask("hello-agent是谁的项目？"))
    print("记忆统计:", assistant.stats)
