"""
滤波降噪算法库。

所有算法通过 @register_algorithm 装饰器自动注册。
新增算法只需在对应分类子目录下创建模块文件即可。
"""

from .base import BaseAlgorithm
from .registry import AlgorithmRegistry, AlgorithmMeta, registry
from .decorators import register_algorithm, log_execution, validate_params

__all__ = [
    "BaseAlgorithm",
    "AlgorithmRegistry",
    "AlgorithmMeta",
    "registry",
    "register_algorithm",
    "log_execution",
    "validate_params",
]
