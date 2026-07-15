"""
Layer 3 — 工作记忆（短期记忆）

检索机制：混合检索（TF-IDF 向量化 + 关键词匹配 + 时间衰减 + 重要性加权）
遗忘机制：
- 基于时间：超过 max_age_seconds 的条目自动淘汰
- 基于重要性：低于阈值的条目被清除
- 基于容量：超限时按重要性驱逐（而非简单 FIFO）
"""
from datetime import datetime, timedelta
from .base import parse_time
from .retriever import HybridRetriever


class WorkingMemory:
    """短期工作记忆：存储最近的对话记录，容量受限，混合检索"""

    def __init__(self, capacity: int = 20, max_age_seconds: int = 3600):
        self._capacity = capacity
        self._max_age_seconds = max_age_seconds
        self._memory: list = []
        self._retriever = HybridRetriever()

    # ── 写入 ──────────────────────────────

    def add(self, content, memory_type: str = "working", importance: float = 0.5):
        """添加一条记忆"""
        self._memory.append({
            "content": content,
            "type": memory_type,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
        })
        self._trim()

    # ── 读取 ──────────────────────────────

    def get_all(self) -> list:
        """返回所有记忆的副本"""
        return list(self._memory)

    def search(self, query: str, limit: int = 5) -> list:
        """
        混合检索：TF-IDF + 关键词 + 时间衰减 + 重要性加权。
        """
        return self._retriever.retrieve(query, self._memory, limit)

    # ── 遗忘：基于重要性 ──────────────────

    def forget_by_importance(self, threshold: float = 0.3) -> int:
        """遗忘所有重要性 < threshold 的记忆。返回被遗忘的条数。"""
        before = len(self._memory)
        self._memory = [m for m in self._memory if m.get("importance", 0) >= threshold]
        return before - len(self._memory)

    # ── 遗忘：基于时间 ────────────────────

    def forget_by_time(self, max_age_seconds: int = None) -> int:
        """遗忘超过指定秒数的记忆。返回被遗忘的条数。"""
        if max_age_seconds is None:
            max_age_seconds = self._max_age_seconds

        cutoff = datetime.now() - timedelta(seconds=max_age_seconds)
        before = len(self._memory)
        self._memory = [m for m in self._memory if parse_time(m.get("timestamp", "")) >= cutoff]
        return before - len(self._memory)

    # ── 遗忘：基于容量 ────────────────────

    def forget_by_capacity(self) -> int:
        """容量超限时，驱逐最不重要的记忆。返回被遗忘的条数。"""
        if len(self._memory) <= self._capacity:
            return 0
        before = len(self._memory)
        self._memory.sort(key=lambda m: m.get("importance", 0))
        self._memory[:] = self._memory[-self._capacity:]
        return before - len(self._memory)

    # ── 综合遗忘 ──────────────────────────

    def forget(
        self,
        importance_threshold: float = 0.3,
        max_age_seconds: int = None,
    ) -> dict:
        """综合遗忘：依次执行三种策略。返回各策略遗忘的条数统计。"""
        result = {
            "by_time": self.forget_by_time(max_age_seconds),
            "by_importance": self.forget_by_importance(importance_threshold),
            "by_capacity": self.forget_by_capacity(),
        }
        result["total"] = sum(result.values())
        return result

    # ── 整合 ──────────────────────────────

    def extract_by_importance(self, threshold: float = 0.7) -> list:
        """提取并移除重要性 >= threshold 的记忆，用于固化为长期记忆。"""
        extracted = [m for m in self._memory if m.get("importance", 0) >= threshold]
        self._memory = [m for m in self._memory if m.get("importance", 0) < threshold]
        return extracted

    def add_batch(self, entries: list, memory_type: str = "working"):
        """批量添加记忆"""
        now = datetime.now().isoformat()
        for entry in entries:
            self._memory.append({
                "content": entry.get("content"),
                "type": memory_type,
                "importance": entry.get("importance", 0.5),
                "timestamp": entry.get("timestamp", now),
            })
        self._trim()

    # ── 增强 ──────────────────────────────

    def reinforce(self, index: int, boost: float = 0.2):
        """增强记忆（模拟复述效应）：提升指定记忆的重要性。"""
        if 0 <= index < len(self._memory):
            self._memory[index]["importance"] = min(
                1.0, self._memory[index].get("importance", 0.5) + boost
            )

    # ── 维护 ──────────────────────────────

    def _trim(self):
        """
        容量超限裁剪：只驱逐多余条数，不重复排序。
        仅在超限 > 10% 时做全量排序（大多数情况下只追加不触发）。
        """
        excess = len(self._memory) - self._capacity
        if excess <= 0:
            return

        # 轻度超限（≤ 3 条）：逐条驱逐最不重要的，O(n) per pop
        if excess <= 3:
            for _ in range(excess):
                min_idx, _ = min(
                    enumerate(self._memory),
                    key=lambda pair: pair[1].get("importance", 0),
                )
                self._memory.pop(min_idx)
            return

        # 重度超限：全量排序驱逐
        self._memory.sort(key=lambda m: m.get("importance", 0))
        self._memory[:] = self._memory[-self._capacity:]

    def clear(self):
        """清空工作记忆"""
        self._memory.clear()

    # ── 元信息 ────────────────────────────

    def __len__(self) -> int:
        return len(self._memory)

    @property
    def capacity(self) -> int:
        return self._capacity
