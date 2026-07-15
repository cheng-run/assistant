"""
Layer 2 — 记忆路由

根据条目内容、重要性、类型标签，决定记忆应该存入：
- working（短期工作记忆）
- episodic（长期情景记忆）
- both（两边都存）

判断逻辑：
  1. memory_type 显式指定 → 直接路由
  2. importance >= 0.7        → 长期
  3. 内容匹配个人信息模式     → 长期
  4. 其余                     → 短期
"""
import re

from .base import extract_text

# 触发长期记忆的内容模式（正则）
_PERSONAL_PATTERNS = [
    re.compile(r"我叫"), re.compile(r"我是"), re.compile(r"我的名字"), re.compile(r"我姓"),
    re.compile(r"我喜欢"), re.compile(r"我讨厌"), re.compile(r"我爱好"), re.compile(r"我不喜欢"),
    re.compile(r"我住在"), re.compile(r"我家"), re.compile(r"我的地址"),
    re.compile(r"我的生日"), re.compile(r"我今年"), re.compile(r"我属"),
    re.compile(r"我的工作"), re.compile(r"我在.*上班"), re.compile(r"我的职业"),
    re.compile(r"记住"), re.compile(r"别忘了"), re.compile(r"帮我记"),
    re.compile(r"我的电话"), re.compile(r"我的邮箱"), re.compile(r"我的微信"),
]


class MemoryRouter:
    """记忆路由器：决定记忆的去向"""

    def classify(self, content, importance: float = 0.5, memory_type: str = None) -> str:
        """
        分类记忆去向。memory_type=None 表示自动判断。

        返回:
            "working"   — 只存短期
            "episodic"  — 只存长期
            "both"      — 两边都存
        """
        # 规则 1：显式指定类型（仅当非 None）
        if memory_type == "working":
            return "working"
        if memory_type == "episodic":
            return "episodic"

        # 规则 2：高重要性 → 长期
        if importance >= 0.7:
            return "episodic"

        # 规则 3：内容命中个人信息模式 → 长期
        text = extract_text(content)
        if self._match_personal(text):
            return "episodic"

        # 规则 4：中等重要性 → 两边都存
        if importance >= 0.4:
            return "both"

        # 默认 → 短期
        return "working"

    # ── 内部辅助 ──────────────────────────

    @staticmethod
    def _match_personal(text: str) -> bool:
        """检测文本是否包含个人信息模式（正则匹配）"""
        for pattern in _PERSONAL_PATTERNS:
            if pattern.search(text):
                return True
        return False
