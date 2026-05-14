"""
算法基类定义。

所有滤波降噪算法必须继承自 BaseAlgorithm 并实现 denoise() 方法。
通过 @register_algorithm 装饰器注册。
"""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional

import numpy as np

from ..core.types import AlgorithmCategory, AlgorithmComplexity, SignalDomain
from ..core.exceptions import SignalTooLargeError


class BaseAlgorithm(ABC):
    """滤波降噪算法基类"""

    # === 元信息（类属性，子类必须重写）===
    name: ClassVar[str] = ""                     # 算法显示名称
    category: ClassVar[AlgorithmCategory] = AlgorithmCategory.CLASSICAL_FILTER
    complexity: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.LOW
    tags: ClassVar[List[str]] = []               # 标签
    requires_gpu: ClassVar[bool] = False         # 是否需要GPU
    input_domain: ClassVar[SignalDomain] = SignalDomain.TIME_DOMAIN

    # 安全保护：算法可处理的最大信号长度，None 表示不限
    # 高复杂度算法应重写此值，超限时自动抛 SignalTooLargeError
    max_signal_length: ClassVar[Optional[int]] = None

    # === 参数管理 ===
    default_params: ClassVar[Dict[str, Any]] = {}
    param_schema: ClassVar[Optional[Dict[str, Any]]] = None
    """参数JSON Schema示例:
    {
        "type": "object",
        "properties": {
            "cutoff_freq": {"type": "number", "description": "截止频率(Hz)", "default": 1000},
            "order": {"type": "integer", "description": "滤波器阶数", "default": 4, "minimum": 1},
        },
        "required": ["cutoff_freq"]
    }
    """

    def __init__(self, **kwargs):
        self.params = self.default_params.copy()
        self.params.update(kwargs)
        self._validate_params()

    def _validate_params(self) -> None:
        """参数校验，可被子类重写"""
        pass

    def setup(self) -> None:
        """初始化钩子（如加载预训练模型）"""
        pass

    def teardown(self) -> None:
        """清理资源钩子"""
        pass

    # === 核心接口 ===

    def _check_signal_size(self, signal: np.ndarray) -> None:
        """检查信号长度是否超出算法安全上限。

        由高复杂度算法在 denoise() 入口处调用，避免在大规模
        信号上触发系统卡死或 OOM。

        Raises:
            SignalTooLargeError: 长度超限时抛出，由 orchestrator 捕获为 success=False。
        """
        if self.max_signal_length is None:
            return
        n_samples = signal.shape[-1] if signal.ndim > 0 else 0
        if n_samples > self.max_signal_length:
            algo_id = getattr(self, 'name', self.__class__.__name__) or self.__class__.__name__
            raise SignalTooLargeError(algo_id, n_samples, self.max_signal_length)

    @abstractmethod
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        """
        对信号进行滤波降噪。

        Args:
            signal: 输入信号，shape (n_samples,) 或 (n_channels, n_samples)
            sample_rate: 采样率 (Hz)
            **kwargs: 运行时参数覆盖

        Returns:
            降噪后的信号，同输入 shape
        """
        pass

    def __call__(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        return self.denoise(signal, sample_rate, **kwargs)

    def __str__(self) -> str:
        return f"[{self.category.value}] {self.name} ({self.complexity.value})"

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>"
