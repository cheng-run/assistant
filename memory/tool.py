"""
Layer 1 — 统一记忆接口（含自动化记忆维护）

对外暴露的顶层 API，内部通过 MemoryRouter 将记忆分发到：
- WorkingMemory（短期，容量受限）
- EpisodicMemory（长期，SQLite 持久化）

遗忘机制（三种策略，自动执行）：
  策略一：基于重要性 — 删除不重要的记忆
  策略二：基于时间     — 删除过时的记忆
  策略三：基于容量     — 超限时驱逐最不重要的记忆

记忆整合（自动执行）：
  重要性 >= 0.7 的短期记忆自动固化为长期情景记忆
"""
from .base import extract_text
from .working import WorkingMemory
from .episodic import EpisodicMemory
from .router import MemoryRouter


class MemoryTool:
    """
    统一记忆工具 — 最外层 API，整合+遗忘自动执行

    用法:
        mem = MemoryTool()

        # 添加记忆（路由自动判断短期/长期）
        mem.add("用户叫小明", importance=0.8)

        # 每次对话后自动维护（推荐在 assistant 中自动调用）
        mem.auto_maintain()
    """

    def __init__(self, working_capacity: int = 20):
        self.working = WorkingMemory(capacity=working_capacity)
        self.episodic = EpisodicMemory()
        self.router = MemoryRouter()

    # ═══════════════════════════════════════
    #  写入
    # ═══════════════════════════════════════

    def add(self, content, importance: float = 0.5, memory_type: str = None):
        """
        添加一条记忆，自动路由到对应的记忆层。

        参数:
            content:      记忆内容（str 或 {"role": ..., "content": ...}）
            importance:   重要性 0.0~1.0，越高越可能进长期记忆
            memory_type:  显式指定类型 "working" / "episodic"，None 则自动判断
        """
        target = self.router.classify(content, importance, memory_type)

        if target in ("working", "both"):
            self.working.add(content, memory_type="working", importance=importance)
        if target in ("episodic", "both"):
            self.episodic.add(content, memory_type="episodic", importance=importance)

    # ═══════════════════════════════════════
    #  读取
    # ═══════════════════════════════════════

    def get_working(self) -> list:
        """获取短期工作记忆（用于构建对话上下文）"""
        return self.working.get_all()

    def get_episodic(self) -> list:
        """获取长期情景记忆"""
        return self.episodic.get_all()

    def get_all(self) -> list:
        """获取短期记忆（对外兼容，主要用于对话历史）"""
        return self.working.get_all()

    def search(self, query: str, limit: int = 5) -> list:
        """跨记忆层搜索，长期记忆优先，搜索同时强化记忆（复述效应）"""
        episodic_results = self.episodic.search(query, limit)
        working_results = self.working.search(query, limit)

        # 复述效应：被检索到的长期记忆重要性提升
        self.episodic.reinforce_by_query(query, boost=0.05)

        # 合并去重（长期在前），用 extract_text 归一化 content 做去重 key
        seen = set()
        merged = []
        for item in episodic_results + working_results:
            key = extract_text(item["content"])
            if key not in seen:
                seen.add(key)
                merged.append(item)
        return merged[:limit]

    def get_relevant_context(self, query: str, limit: int = 3) -> list:
        """
        根据查询获取相关的情景记忆上下文。
        用于把长期记忆注入 LLM 对话。
        """
        return self.episodic.search(query, limit)

    # ═══════════════════════════════════════
    #  遗忘：策略一 — 基于重要性
    # ═══════════════════════════════════════

    def forget_by_importance(self, threshold: float = 0.3) -> dict:
        """
        删除所有重要性低于阈值的记忆。

        参数:
            threshold: 重要性阈值 0.0~1.0，低于此值的记忆被遗忘
        返回:
            {"working": 删除数, "episodic": 删除数, "total": 总数}
        """
        w = self.working.forget_by_importance(threshold)
        e = self.episodic.forget_by_importance(threshold)
        return {"working": w, "episodic": e, "total": w + e}

    # ═══════════════════════════════════════
    #  遗忘：策略二 — 基于时间
    # ═══════════════════════════════════════

    def forget_by_time(self, max_age_days: int = 30) -> dict:
        """
        删除超过指定天数的记忆。

        参数:
            max_age_days: 最大存活天数
        返回:
            {"working": 删除数, "episodic": 删除数, "total": 总数}
        """
        # 工作记忆以秒为单位，转换为秒
        working_forgotten = self.working.forget_by_time(
            max_age_seconds=max_age_days * 86400
        )
        episodic_forgotten = self.episodic.forget_by_time(max_age_days)
        return {
            "working": working_forgotten,
            "episodic": episodic_forgotten,
            "total": working_forgotten + episodic_forgotten,
        }

    # ═══════════════════════════════════════
    #  遗忘：策略三 — 基于容量
    # ═══════════════════════════════════════

    def forget_by_capacity(self, soft_limit: int = None) -> dict:
        """
        当存储接近容量上限时，删除最不重要的记忆。

        参数:
            soft_limit: 软上限，超过则驱逐最低有效重要性的记忆
        返回:
            {"working": 删除数, "episodic": 删除数, "total": 总数}
        """
        working_forgotten = self.working.forget_by_capacity()
        episodic_forgotten = self.episodic.forget_by_capacity(soft_limit)
        return {
            "working": working_forgotten,
            "episodic": episodic_forgotten,
            "total": working_forgotten + episodic_forgotten,
        }

    # ═══════════════════════════════════════
    #  遗忘：综合执行
    # ═══════════════════════════════════════

    def forget(
        self,
        importance_threshold: float = 0.3,
        max_age_days: int = 30,
        soft_limit: int = None,
    ) -> dict:
        """
        综合遗忘：一次性执行三种遗忘策略。

        执行顺序：
          1. 基于时间（淘汰过期记忆）
          2. 基于重要性（淘汰低价值记忆）
          3. 基于容量（超限驱逐）

        返回:
            {"by_time": n, "by_importance": n, "by_capacity": n, "total": n}
        """
        result: dict = {}

        # 策略 1：基于时间
        wt = self.working.forget_by_time(max_age_seconds=max_age_days * 86400)
        et = self.episodic.forget_by_time(max_age_days)
        result["by_time"] = wt + et

        # 策略 2：基于重要性
        wi = self.working.forget_by_importance(importance_threshold)
        ei = self.episodic.forget_by_importance(importance_threshold)
        result["by_importance"] = wi + ei

        # 策略 3：基于容量
        wc = self.working.forget_by_capacity()
        ec = self.episodic.forget_by_capacity(soft_limit)
        result["by_capacity"] = wc + ec

        result["total"] = sum(result.values())
        return result

    # ═══════════════════════════════════════
    #  自动化维护（整合 + 遗忘）
    # ═══════════════════════════════════════

    def auto_maintain(
        self,
        consolidate_threshold: float = 0.7,
        forget_importance: float = 0.3,
        forget_max_age_days: int = 30,
    ) -> dict:
        """
        自动记忆维护：整合 → 遗忘，推荐在每次对话后调用。

        1. 整合：重要性 >= consolidate_threshold 的短期记忆 → 长期情景记忆
        2. 遗忘：执行三种遗忘策略清理无效记忆

        返回:
            {
                "consolidated": n,           # 整合条数
                "forgotten": {
                    "by_time": n,
                    "by_importance": n,
                    "by_capacity": n,
                    "total": n,
                },
            }
        """
        # 步骤 1：记忆整合（短期 → 长期）
        consolidated = self.consolidate_memories(  # 新拼写
            from_type="working",
            to_type="episodic",
            importance_threshold=consolidate_threshold,
            remove_from_source=True,
        )

        # 步骤 2：三种策略遗忘
        forgotten = self.forget(
            importance_threshold=forget_importance,
            max_age_days=forget_max_age_days,
        )

        return {
            "consolidated": consolidated,
            "forgotten": forgotten,
        }

    # ═══════════════════════════════════════
    #  记忆整合（记忆固化）
    # ═══════════════════════════════════════

    def consolidate_memories(  # 正确拼写
        self,
        from_type: str = "working",
        to_type: str = "episodic",
        importance_threshold: float = 0.7,
        remove_from_source: bool = True,
    ) -> int:
        """
        记忆整合：将重要的短期记忆提升为长期记忆。

        模拟人脑的记忆固化过程——海马体中的短期记忆在睡眠期间
        被逐步转移到新皮层成为长期记忆。系统自动识别重要性超过
        阈值的记忆，将其从源记忆层转移到目标记忆层。

        参数:
            from_type:            源记忆类型 "working" / "episodic"
            to_type:              目标记忆类型 "episodic" / "semantic"（预留）
            importance_threshold: 重要性阈值，>= 此值的记忆被整合
            remove_from_source:   是否从源中移除（True=移动, False=复制）

        返回:
            成功整合的记忆条数

        示例:
            # 将重要的工作记忆固化为情景记忆
            count = mem.consolidate_memories(
                from_type="working",
                to_type="episodic",
                importance_threshold=0.7,
            )
            # → "🔄 已整合 3 条记忆为长期记忆"
        """
        source = self._get_store(from_type)
        target = self._get_store(to_type)

        if source is None or target is None:
            raise ValueError(
                f"不支持的记忆类型: from={from_type}, to={to_type}。"
                f"当前支持: working, episodic"
            )

        if source is target:
            return 0

        # 提取高价值记忆
        consolidated = source.extract_by_importance(importance_threshold)

        if not consolidated:
            return 0

        # 更新类型标记并写入目标
        for entry in consolidated:
            entry["type"] = to_type

        target.add_batch(consolidated, memory_type=to_type)

        # 如果只复制不移除，把条目还回源
        if not remove_from_source:
            source.add_batch(consolidated, memory_type=from_type)

        return len(consolidated)

    # 旧拼写别名（保持向后兼容）
    consolidate_memories = consolidate_memories

    def _get_store(self, memory_type: str):
        """
        将类型名称映射到实际的记忆存储实例。

        支持的类型:
            "working"  → WorkingMemory 实例
            "episodic" → EpisodicMemory 实例
            "semantic" → None（预留，后续扩展）
        """
        stores = {
            "working": self.working,
            "episodic": self.episodic,
            "semantic": None,  # 预留：语义记忆
        }
        return stores.get(memory_type)

    # ═══════════════════════════════════════
    #  维护
    # ═══════════════════════════════════════

    def clear_working(self):
        """清空短期记忆"""
        self.working.clear()

    def clear_episodic(self):
        """清空长期记忆"""
        self.episodic.clear()

    def clear_all(self):
        """清空全部记忆"""
        self.working.clear()
        self.episodic.clear()

    # ═══════════════════════════════════════
    #  元信息
    # ═══════════════════════════════════════

    def __len__(self) -> int:
        return len(self.working) + len(self.episodic)

    @property
    def stats(self) -> dict:
        """返回记忆统计信息"""
        return {
            "working_count": len(self.working),
            "working_capacity": self.working.capacity,
            "episodic_count": len(self.episodic),
            "episodic_soft_limit": self.episodic.soft_limit,
            "total": len(self.working) + len(self.episodic),
        }
