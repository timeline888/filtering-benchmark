"""
评估指标模块。

包含 20 种评估指标的基类定义、注册器和具体实现。
"""
from .base import BaseMetric, MetricRegistry, metric_registry
from .envelope_metrics import (
    # 特征频率直接度量
    FFAMetric, RFFIMetric, FBRMetric, ERMetric, LSNRMetric, HERMetric,
    # 统计分布特征
    KurtosisMetric, SKMetric, ESKMetric, SkewnessMetric,
    # 综合特征比与范数
    FFRMetric, HSIMetric, SLNMetric, HLNMetric,
    # 稀疏性与复杂度
    GIMetric, ESEMetric, SIMetric,
    # 时域与扩展
    CFMetric, IFMetric, SERMetric,
)

__all__ = [
    "BaseMetric", "MetricRegistry", "metric_registry",
    # 20种指标
    "FFAMetric", "RFFIMetric", "FBRMetric", "ERMetric",
    "LSNRMetric", "HERMetric",
    "KurtosisMetric", "SKMetric", "ESKMetric", "SkewnessMetric",
    "FFRMetric", "HSIMetric", "SLNMetric", "HLNMetric",
    "GIMetric", "ESEMetric", "SIMetric",
    "CFMetric", "IFMetric", "SERMetric",
]
