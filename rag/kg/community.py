"""
社区检测与摘要 — Louvain 算法 + LLM 摘要生成

流程:
  1. 从邻接表运行 Louvain 社区检测
  2. 计算每个社区的统计量（密度、关键实体）
  3. 用 LLM 生成每个社区的摘要
  4. 嵌入摘要用于后续检索
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple


# ── 类型别名 ──────────────────────────
CommunityDict = Dict   # {"level": int, "entity_ids": [...], "key_entities": [...],
                       #  "summary": str, "summary_embedding": [...],
                       #  "relation_count": int, "density": float}


class CommunityDetector:
    """
    基于 Louvain 算法的社区检测（greedy modularity optimization）。

    参考: Blondel et al. (2008) "Fast unfolding of communities in large networks"

    仅依赖标准库，无外部依赖。
    """

    def detect(
        self,
        adjacency: Dict[int, List[Tuple[int, float]]],
        resolution: float = 1.0,
        max_iter: int = 50,
        min_community_size: int = 3,
    ) -> List[CommunityDict]:
        """
        运行 Louvain 社区检测。

        参数:
            adjacency:           邻接表 {node_id: [(neighbor_id, weight), ...]}
            resolution:          分辨率参数（越高 = 越小/越多的社区）
            max_iter:            最大迭代次数
            min_community_size:  最小社区大小（小于此值的社区被过滤）
        返回:
            社区列表
        """
        if not adjacency:
            return []

        nodes = list(adjacency.keys())
        if len(nodes) < min_community_size:
            return []

        # ── 计算总边权 ──
        total_weight = 0.0
        node_degrees = {}
        for node, neighbors in adjacency.items():
            deg = sum(w for _, w in neighbors)
            node_degrees[node] = deg
            total_weight += deg
        total_weight /= 2.0  # 无向图每条边被算了两次

        if total_weight == 0:
            return []

        # ── Phase 1: 初始分区（贪婪模块度优化） ──
        partition = {node: node for node in nodes}  # node → community_id
        communities = self._louvain_phase1(
            adjacency, partition, node_degrees, total_weight, resolution, max_iter
        )

        # ── 转换为标准格式 ──
        # communities: {community_id: set of node_ids}
        result = []
        for comm_nodes in communities.values():
            if len(comm_nodes) >= min_community_size:
                entity_ids = sorted(comm_nodes)
                density = self._compute_density(entity_ids, adjacency)
                key_entities = self._compute_key_entities(entity_ids, adjacency)

                result.append({
                    "level": 0,
                    "entity_ids": entity_ids,
                    "key_entities": key_entities,
                    "density": round(density, 4),
                    "relation_count": self._count_internal_edges(entity_ids, adjacency),
                })

        return result

    def _louvain_phase1(
        self,
        adjacency: Dict[int, List[Tuple[int, float]]],
        partition: Dict[int, int],
        node_degrees: Dict[int, float],
        total_weight: float,
        resolution: float,
        max_iter: int,
    ) -> Dict[int, set]:
        """
        Phase 1: 局部移动阶段。

        每个节点迭代移动到最大化模块度增益的邻居社区。
        """
        nodes = list(adjacency.keys())
        n_nodes = len(nodes)

        if n_nodes <= 1:
            # 单节点图：直接返回
            if nodes:
                return {nodes[0]: {nodes[0]}}
            return {}

        # 初始化：每个节点一个社区
        communities = {node: {node} for node in nodes}
        node_to_comm = {node: node for node in nodes}

        # 社区统计
        comm_in = defaultdict(float)   # 社区内部边权
        comm_tot = defaultdict(float)  # 社区总入射边权

        for node in nodes:
            comm_in[node] = 0.0
            for neighbor, w in adjacency.get(node, []):
                if neighbor in communities[node]:
                    comm_in[node] += w
            comm_in[node] /= 2.0
            comm_tot[node] = node_degrees.get(node, 0.0)

        improved = True
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1
            random.shuffle(nodes)

            for node in nodes:
                current_comm = node_to_comm[node]
                best_gain = 0.0
                best_comm = current_comm

                # 计算邻居社区（去重）
                neighbor_comms = {}
                for neighbor, w in adjacency.get(node, []):
                    nc = node_to_comm[neighbor]
                    if nc != current_comm:
                        neighbor_comms[nc] = neighbor_comms.get(nc, 0.0) + w

                # 评估移动到每个邻居社区
                for nc, k_i_in in neighbor_comms.items():
                    # k_i_in: 节点到目标社区的边权和
                    k_i = node_degrees.get(node, 0.0)
                    sigma_tot = comm_tot.get(nc, 0.0)

                    if total_weight == 0:
                        continue

                    # ΔQ = k_i_in / m - resolution * sigma_tot * k_i / (2 * m^2)
                    # Derived from the full modularity gain formula of Blondel et al.
                    gain = (k_i_in / total_weight
                            - resolution * sigma_tot * k_i / (2.0 * total_weight * total_weight))

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = nc

                if best_comm != current_comm:
                    improved = True
                    old_comm = current_comm
                    new_comm = best_comm

                    # 从旧社区移除节点
                    communities[old_comm].discard(node)
                    if not communities[old_comm]:
                        del communities[old_comm]

                    # 加入新社区
                    if new_comm not in communities:
                        communities[new_comm] = set()
                    communities[new_comm].add(node)

                    node_to_comm[node] = new_comm

                    # 更新社区统计
                    # 从旧社区扣除
                    for neighbor, w in adjacency.get(node, []):
                        if neighbor in communities.get(old_comm, set()):
                            comm_in[old_comm] -= w / 2.0
                    comm_tot[old_comm] -= node_degrees.get(node, 0.0)

                    # 加入新社区
                    for neighbor, w in adjacency.get(node, []):
                        if neighbor in communities.get(new_comm, set()):
                            comm_in[new_comm] += w / 2.0
                    comm_tot[new_comm] += node_degrees.get(node, 0.0)

                    # 清理空社区统计
                    if old_comm not in communities:
                        comm_in.pop(old_comm, None)
                        comm_tot.pop(old_comm, None)

        return communities

    @staticmethod
    def _compute_density(
        entity_ids: List[int],
        adjacency: Dict[int, List[Tuple[int, float]]],
    ) -> float:
        """计算社区内部边密度"""
        n = len(entity_ids)
        if n <= 1:
            return 0.0

        ids_set = set(entity_ids)
        edge_count = 0

        for node in entity_ids:
            for neighbor, _ in adjacency.get(node, []):
                if neighbor in ids_set:
                    edge_count += 1

        # 无向图每条边被算了两次
        edge_count //= 2
        max_edges = n * (n - 1) // 2
        return edge_count / max_edges if max_edges > 0 else 0.0

    @staticmethod
    def _compute_key_entities(
        entity_ids: List[int],
        adjacency: Dict[int, List[Tuple[int, float]]],
        top_n: int = 5,
    ) -> List[Dict]:
        """按度中心性排序，返回 top-N 关键实体"""
        ids_set = set(entity_ids)
        scores = []

        for node in entity_ids:
            # 社区内部度数
            internal_deg = sum(
                w for n, w in adjacency.get(node, []) if n in ids_set
            )
            # 总度数
            total_deg = sum(w for _, w in adjacency.get(node, []))
            # 综合得分：内部度 + 0.5 * 外部度
            score = internal_deg + 0.5 * (total_deg - internal_deg)
            scores.append((node, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [{"id": n, "centrality": round(s, 4)} for n, s in scores[:top_n]]

    @staticmethod
    def _count_internal_edges(
        entity_ids: List[int],
        adjacency: Dict[int, List[Tuple[int, float]]],
    ) -> int:
        """统计社区内部边数"""
        ids_set = set(entity_ids)
        edge_count = 0
        for node in entity_ids:
            for neighbor, _ in adjacency.get(node, []):
                if neighbor in ids_set:
                    edge_count += 1
        return edge_count // 2  # 无向边除以 2


class CommunitySummarizer:
    """
    LLM 驱动的社区摘要生成器。
    """

    # ── Prompt 模板 ───────────────────
    _SUMMARIZE_PROMPT = """你是一个知识图谱分析助手。下面是一组互相关联的实体和它们之间的关系。

实体列表：
{entity_list}

关系列表：
{relation_list}

请用3-5句话总结这个实体组代表什么主题或内容领域，包括：
1. 核心主题或领域
2. 关键实体之间的关系结构
3. 最重要的几个实体及其角色

直接输出总结文字，不要用JSON格式。"""

    def summarize(
        self,
        entities: List[Dict],
        relations: List[Dict],
    ) -> str:
        """
        为社区生成中文摘要。

        参数:
            entities:  [{name, entity_type, description}, ...]
            relations: [{source_name, target_name, relation}, ...]
        返回:
            中文摘要文本
        """
        if not entities and not relations:
            return ""

        # ── 格式化实体列表 ──
        entity_lines = []
        for e in entities[:30]:  # 限制 30 个实体
            etype = e.get("entity_type", "")
            desc = e.get("description", "")
            name = e.get("name", "")
            line = f"- {name}"
            if etype:
                line += f"（{etype}）"
            if desc:
                line += f": {desc}"
            entity_lines.append(line)
        entity_list = "\n".join(entity_lines) if entity_lines else "（无实体）"

        # ── 格式化关系列表 ──
        relation_lines = []
        for r in relations[:20]:  # 限制 20 个关系
            line = f"- {r.get('source_name', r.get('source', ''))} → {r.get('relation', '')} → {r.get('target_name', r.get('target', ''))}"
            relation_lines.append(line)
        relation_list = "\n".join(relation_lines) if relation_lines else "（无关系）"

        # ── 调用 LLM ──
        try:
            from llm import ask_llm

            prompt = self._SUMMARIZE_PROMPT.format(
                entity_list=entity_list,
                relation_list=relation_list,
            )
            messages = [
                {"role": "system", "content": "你是一个知识图谱分析助手，善于总结结构化信息。"},
                {"role": "user", "content": prompt},
            ]
            summary = ask_llm(messages=messages)
            return summary.strip() if summary else ""

        except Exception:
            return ""

    def summarize_all(
        self,
        communities: List[CommunityDict],
        entity_map: Dict[int, Dict],
        store,  # KnowledgeGraphStore
        embedder,  # LocalEmbedder
    ) -> List[CommunityDict]:
        """
        为所有社区生成摘要并嵌入。

        参数:
            communities: 社区列表
            entity_map:  {entity_id: {name, entity_type, description, ...}}
            store:       KnowledgeGraphStore 实例（用于查关系）
            embedder:    LocalEmbedder 实例（用于嵌入摘要）
        返回:
            带 summary 和 summary_embedding 的社区列表
        """
        enriched = []

        for comm in communities:
            entity_ids = comm.get("entity_ids", [])

            # 获取实体详情
            comm_entities = []
            name_to_info = {}
            for eid in entity_ids:
                if eid in entity_map:
                    info = entity_map[eid]
                    comm_entities.append(info)
                    name_to_info[info.get("name", "")] = info

            # 获取内部关系
            comm_relations = store.get_relations_with_names(
                entity_ids=entity_ids, limit=20
            )

            # 生成摘要（实体 ≥ 3 才摘要）
            if len(comm_entities) >= 3:
                summary = self.summarize(comm_entities, comm_relations)
            else:
                summary = ""

            # 嵌入摘要
            summary_embedding = []
            if summary and embedder:
                try:
                    summary_embedding = embedder.embed(summary)
                except Exception:
                    summary_embedding = []

            comm["summary"] = summary
            comm["summary_embedding"] = summary_embedding
            enriched.append(comm)

        return enriched
