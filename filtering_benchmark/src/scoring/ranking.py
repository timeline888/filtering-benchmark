"""
评分与排序模块。

支持：
- 指标归一化（MinMax / ZScore / Rank）
- 单指标排序
- 多指标加权综合评分
- TOPSIS 多准则决策
- Borda 投票
- Pareto 前沿分析
"""

from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from ..core.types import AlgorithmRanking, MetricResult
from ..metrics.base import metric_registry


class MetricNormalizer:
    """指标归一化器"""

    @staticmethod
    def minmax(values: List[float], higher_is_better: bool = True) -> List[float]:
        """Min-Max 归一化到 [0, 1]"""
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        min_v, max_v = np.min(arr), np.max(arr)
        if max_v - min_v < 1e-12:
            return [0.5] * len(values)
        if higher_is_better:
            return ((arr - min_v) / (max_v - min_v)).tolist()
        else:
            return ((max_v - arr) / (max_v - min_v)).tolist()

    @staticmethod
    def zscore(values: List[float], higher_is_better: bool = True) -> List[float]:
        """Z-Score 标准化"""
        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0)
        mean, std = np.mean(arr), np.std(arr)
        if std < 1e-12:
            return [0.0] * len(values)
        z = (arr - mean) / std
        if not higher_is_better:
            z = -z
        # Sigmoid 映射到 [0, 1]
        return (1 / (1 + np.exp(-z / 3))).tolist()

    @staticmethod
    def rank(values: List[float], higher_is_better: bool = True) -> List[float]:
        """排名归一化 """
        arr = np.array(values)
        arr = np.nan_to_num(arr, nan=0.0)
        if higher_is_better:
            ranks = np.argsort(np.argsort(-arr))
        else:
            ranks = np.argsort(np.argsort(arr))
        return (ranks / max(len(ranks) - 1, 1)).tolist()

    @classmethod
    def normalize(cls, values: List[float], method: str = "minmax",
                  higher_is_better: bool = True) -> List[float]:
        method_map = {
            "minmax": cls.minmax,
            "zscore": cls.zscore,
            "rank": cls.rank,
        }
        return method_map.get(method, cls.minmax)(values, higher_is_better)


class ScoringEngine:
    """评分引擎"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.normalization_method = self.config.get("normalization", "minmax")
        self.scoring_method = self.config.get("method", "weighted_sum")

    def score(self, metric_scores: Dict[str, Dict[str, float]],
              algorithm_ids: List[str],
              weights: Dict[str, float],
              higher_is_better: Dict[str, bool]) -> Dict[str, float]:
        """
        评估并评分所有算法。

        Args:
            metric_scores: {algorithm_id: {metric_name: value}}
            algorithm_ids: 算法ID列表
            weights: {metric_name: weight}
            higher_is_better: {metric_name: True/False}

        Returns:
            {algorithm_id: overall_score}
        """
        if self.scoring_method == "weighted_sum":
            return self._weighted_sum(metric_scores, algorithm_ids, weights, higher_is_better)
        elif self.scoring_method == "topsis":
            return self._topsis(metric_scores, algorithm_ids, higher_is_better)
        elif self.scoring_method == "borda_count":
            return self._borda_count(metric_scores, algorithm_ids, higher_is_better)
        elif self.scoring_method == "pareto":
            return self._pareto(metric_scores, algorithm_ids, higher_is_better)
        else:
            return self._weighted_sum(metric_scores, algorithm_ids, weights, higher_is_better)

    def _weighted_sum(self, metric_scores, algorithm_ids, weights, higher_is_better):
        """加权求和法"""
        scores = {}
        for algo_id in algorithm_ids:
            ms = metric_scores.get(algo_id, {})
            total = 0.0
            total_w = 0.0
            for mname, w in weights.items():
                if mname in ms:
                    vals = [metric_scores[a].get(mname, 0) for a in algorithm_ids]
                    norm_vals = MetricNormalizer.normalize(
                        vals, self.normalization_method,
                        higher_is_better.get(mname, True)
                    )
                    norm_val = norm_vals[algorithm_ids.index(algo_id)]
                    total += w * norm_val
                    total_w += w
            scores[algo_id] = total / max(total_w, 1e-12)
        return scores

    def _topsis(self, metric_scores, algorithm_ids, higher_is_better):
        """TOPSIS 多准则决策"""
        n = len(algorithm_ids)
        metric_names = list(higher_is_better.keys())

        # 构建决策矩阵
        matrix = np.zeros((n, len(metric_names)))
        for i, algo_id in enumerate(algorithm_ids):
            for j, mname in enumerate(metric_names):
                matrix[i, j] = metric_scores.get(algo_id, {}).get(mname, 0)

        # 归一化
        norm_matrix = matrix / np.sqrt(np.sum(matrix ** 2, axis=0, keepdims=True) + 1e-12)

        # 确定理想解和负理想解
        ideal = np.where(
            np.array([higher_is_better.get(m, True) for m in metric_names]),
            np.max(norm_matrix, axis=0),
            np.min(norm_matrix, axis=0),
        )
        neg_ideal = np.where(
            np.array([higher_is_better.get(m, True) for m in metric_names]),
            np.min(norm_matrix, axis=0),
            np.max(norm_matrix, axis=0),
        )

        # 计算距离
        dist_ideal = np.sqrt(np.sum((norm_matrix - ideal) ** 2, axis=1))
        dist_neg = np.sqrt(np.sum((norm_matrix - neg_ideal) ** 2, axis=1))

        # 相对接近度
        scores = dist_neg / (dist_ideal + dist_neg + 1e-12)
        return {algo_id: float(s) for algo_id, s in zip(algorithm_ids, scores)}

    def _borda_count(self, metric_scores, algorithm_ids, higher_is_better):
        """Borda 投票法"""
        scores = {algo_id: 0 for algo_id in algorithm_ids}
        metric_names = list(higher_is_better.keys())

        for mname in metric_names:
            vals = [(algo_id, metric_scores[algo_id].get(mname, 0))
                    for algo_id in algorithm_ids]
            vals.sort(key=lambda x: x[1], reverse=higher_is_better.get(mname, True))
            for rank, (algo_id, _) in enumerate(vals):
                scores[algo_id] += n - rank  # n - rank 分

        n = len(algorithm_ids)
        max_score = len(metric_names) * n
        return {algo_id: s / max_score for algo_id, s in scores.items()}

    def _pareto(self, metric_scores, algorithm_ids, higher_is_better):
        """Pareto 前沿分析"""
        metric_names = list(higher_is_better.keys())
        n_metrics = len(metric_names)

        def dominates(a, b):
            """检查a是否支配b"""
            better_any = False
            for mname in metric_names:
                va = metric_scores[a].get(mname, 0)
                vb = metric_scores[b].get(mname, 0)
                hb = higher_is_better.get(mname, True)
                if hb:
                    if va > vb:
                        better_any = True
                    elif va < vb:
                        return False
                else:
                    if va < vb:
                        better_any = True
                    elif va > vb:
                        return False
            return better_any

        # Pareto 前沿排名：计算每个算法被支配的次数
        domination_count = {algo_id: 0 for algo_id in algorithm_ids}
        for a in algorithm_ids:
            for b in algorithm_ids:
                if a != b and dominates(b, a):
                    domination_count[a] += 1

        scores = {}
        max_count = max(domination_count.values()) if domination_count else 1
        for algo_id, count in domination_count.items():
            scores[algo_id] = 1 - count / max(max_count, 1)
        return scores


class RankingEngine:
    """排序引擎"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.scoring_engine = ScoringEngine(config)

    def rank(self,
             metric_scores: Dict[str, Dict[str, float]],
             algorithm_metas: Dict[str, dict],
             weights: Dict[str, float],
             higher_is_better: Dict[str, bool],
             top_n: int = 3) -> List[AlgorithmRanking]:
        """
        排序算法。

        Args:
            metric_scores: {algorithm_id: {metric_name: value}}
            algorithm_metas: {algorithm_id: {name, category, execution_time}}
            weights: {metric_name: weight}
            higher_is_better: {metric_name: True/False}
            top_n: 返回前N个

        Returns:
            排序后的 AlgorithmRanking 列表
        """
        algorithm_ids = list(metric_scores.keys())
        if not algorithm_ids:
            return []

        # 计算综合得分
        overall_scores = self.scoring_engine.score(
            metric_scores, algorithm_ids, weights, higher_is_better
        )

        # 计算单指标排名
        metric_names = list(higher_is_better.keys())
        per_metric_ranks: Dict[str, Dict[str, int]] = {}

        for mname in metric_names:
            vals = [(algo_id, metric_scores[algo_id].get(mname, 0))
                    for algo_id in algorithm_ids]
            vals.sort(key=lambda x: x[1], reverse=higher_is_better.get(mname, True))
            for rank, (algo_id, _) in enumerate(vals, 1):
                if mname not in per_metric_ranks:
                    per_metric_ranks[mname] = {}
                per_metric_ranks[mname][algo_id] = rank

        # 构建排名结果
        sorted_algos = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings = []
        for rank, (algo_id, score) in enumerate(sorted_algos, 1):
            meta = algorithm_metas.get(algo_id, {})
            ranking = AlgorithmRanking(
                algorithm_id=algo_id,
                algorithm_name=meta.get("name", algo_id),
                category=meta.get("category", ""),
                overall_score=float(score),
                overall_rank=rank,
                metric_scores=metric_scores.get(algo_id, {}),
                metric_ranks={m: per_metric_ranks.get(m, {}).get(algo_id, 0)
                              for m in metric_names},
                execution_time=float(meta.get("execution_time", 0)),
            )
            rankings.append(ranking)

        return rankings[:max(top_n, len(rankings))]
