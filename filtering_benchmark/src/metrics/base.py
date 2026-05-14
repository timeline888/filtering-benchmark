"""
评估指标基类定义。

所有评估指标必须继承 BaseMetric 并实现 compute() 方法。
使用 MetricRegistry 注册。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ..core.types import EnvelopeResult, MetricResult


class BaseMetric(ABC):
    """评估指标基类"""

    name: str = ""
    category: str = ""
    higher_is_better: bool = True
    description: str = ""

    @abstractmethod
    def compute(self, denoised: np.ndarray, sample_rate: float,
                reference: Optional[np.ndarray] = None,
                envelope: Optional[EnvelopeResult] = None,
                **kwargs) -> float:
        """
        计算指标值。

        Args:
            denoised: 降噪后的信号
            sample_rate: 采样率
            reference: 参考信号（有监督指标使用）
            envelope: 包络谱分析结果
            **kwargs: 额外参数（如故障频率配置）

        Returns:
            指标计算结果
        """
        pass

    def to_result(self, value: float) -> MetricResult:
        """转换为 MetricResult"""
        return MetricResult(
            metric_name=self.name,
            value=value,
            higher_is_better=self.higher_is_better,
            category=self.category,
        )


class MetricRegistry:
    """指标注册器"""

    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}

    def register(self, metric_cls):
        """装饰器：注册指标"""
        instance = metric_cls()
        self._metrics[instance.name] = instance
        return metric_cls

    def get(self, name: str) -> BaseMetric:
        if name not in self._metrics:
            raise KeyError(f"未知指标: {name}，可用指标: {list(self._metrics.keys())}")
        return self._metrics[name]

    def list_all(self) -> Dict[str, BaseMetric]:
        return dict(self._metrics)

    def list_names(self) -> List[str]:
        return list(self._metrics.keys())

    def compute_all(self,
                    denoised: np.ndarray,
                    sample_rate: float,
                    enabled: List[str],
                    reference: Optional[np.ndarray] = None,
                    envelope: Optional[EnvelopeResult] = None,
                    **kwargs) -> Dict[str, float]:
        """批量计算指标"""
        results = {}
        for name in enabled:
            if name not in self._metrics:
                continue
            try:
                metric = self._metrics[name]
                value = metric.compute(
                    denoised=denoised,
                    sample_rate=sample_rate,
                    reference=reference,
                    envelope=envelope,
                    **kwargs
                )
                results[name] = value
            except Exception as e:
                from loguru import logger
                logger.warning(f"指标计算失败 [{name}]: {e}")
                results[name] = 0.0
        return results


# 全局注册器
metric_registry = MetricRegistry()
