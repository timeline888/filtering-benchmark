"""
算法装饰器工具集。

- @register_algorithm: 声明式注册算法
- @log_execution: 自动记录算法执行日志（可通过环境变量 FILTERING_LOG_ENABLED=0 关闭）
- @validate_params: 自动参数校验
"""

import functools
import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar

from loguru import logger

from ..core.types import AlgorithmCategory, AlgorithmComplexity
from .base import BaseAlgorithm
from .registry import registry

# 日志开关：可通过环境变量 FILTERING_LOG_ENABLED=0 禁用日志，提升性能
_LOG_ENABLED = os.getenv("FILTERING_LOG_ENABLED", "1") == "1"

T = TypeVar("T", bound=Type[BaseAlgorithm])


def register_algorithm(
    name: str,
    category: AlgorithmCategory,
    complexity: AlgorithmComplexity,
    tags: Optional[List[str]] = None,
    requires_gpu: bool = False,
) -> Callable[[T], T]:
    """
    算法注册装饰器。

    用法:
        @register_algorithm(
            name="小波阈值降噪",
            category=AlgorithmCategory.WAVELET,
            complexity=AlgorithmComplexity.MEDIUM,
        )
        class WaveletThresholdDenoise(BaseAlgorithm):
            ...
    """
    def decorator(cls: T) -> T:
        if not issubclass(cls, BaseAlgorithm):
            raise TypeError(f"{cls.__name__} 必须继承 BaseAlgorithm")

        # 注入元信息
        cls.name = name
        cls.category = category
        cls.complexity = complexity
        cls.requires_gpu = requires_gpu
        if tags:
            cls.tags = tags

        # 生成算法ID（类名的小写蛇形形式）
        algorithm_id = _to_snake_case(cls.__name__)

        # 注册
        registry.register(algorithm_id, cls)
        logger.info(f"注册算法: [{category.value}] {name}")

        return cls
    return decorator


def log_execution(func: Callable) -> Callable:
    """
    算法执行日志装饰器。
    自动记录输入/输出/耗时。
    可通过环境变量 FILTERING_LOG_ENABLED=0 禁用，提升性能。
    """
    if not _LOG_ENABLED:
        # 日志禁用时，直接返回原函数，零开销
        return func

    @functools.wraps(func)
    def wrapper(self, signal, sample_rate, **kwargs):
        logger.info(f"[{self.name}] 开始降噪处理, signal.shape={signal.shape}, fs={sample_rate}Hz")
        start = time.perf_counter()
        try:
            result = func(self, signal, sample_rate, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[{self.name}] 降噪完成, 耗时={elapsed:.3f}s")
            return result
        except Exception as e:
            elapsed = time.perf_counter() - start
            logger.error(f"[{self.name}] 降噪失败, 耗时={elapsed:.3f}s, 错误={e}")
            raise
    return wrapper


def validate_params(func: Callable) -> Callable:
    """
    参数校验装饰器。
    在执行前校验参数合法性（仅校验运行时传入的kwargs）。
    """
    @functools.wraps(func)
    def wrapper(self, signal, sample_rate, **kwargs):
        # 合并参数
        all_params = {**self.params, **kwargs}

        if hasattr(self, '_validate_runtime_params'):
            self._validate_runtime_params(all_params)

        return func(self, signal, sample_rate, **kwargs)
    return wrapper


def _to_snake_case(name: str) -> str:
    """将驼峰命名转为蛇形命名"""
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', name)
    name = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    return name.lower()
