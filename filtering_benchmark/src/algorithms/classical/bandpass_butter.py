"""
带通/带阻巴特沃斯滤波器及 IIR 陷波滤波器。

包含三种经典滤波器实现：
1. 巴特沃斯带通滤波器 - 提取特定频段振动特征
2. 巴特沃斯带阻滤波器 - 抑制特定频率干扰
3. IIR 陷波滤波器 - 去除工频 50/60Hz 干扰

参考书：第3章第3-4节 - 带通/带阻/陷波滤波器
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="巴特沃斯带通滤波器",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "linear", "bandpass"],
)
class BandPassButterworth(BaseAlgorithm):
    """巴特沃斯带通滤波器，提取特定频段振动特征"""

    default_params: ClassVar[Dict[str, Any]] = {
        "low_cutoff": 500.0,   # 下截止频率 (Hz)
        "high_cutoff": 5000.0,  # 上截止频率 (Hz)
        "order": 4,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        low = params["low_cutoff"]
        high = params["high_cutoff"]
        order = params["order"]

        nyquist = sample_rate / 2.0
        normalized = [low / nyquist, high / nyquist]

        if normalized[0] >= 1.0 or normalized[1] >= 1.0:
            raise ValueError(f"截止频率超过奈奎斯特频率 {nyquist}Hz")

        sos = scipy_signal.butter(order, normalized, btype="band", output="sos")
        return scipy_signal.sosfiltfilt(sos, signal, axis=-1)


@register_algorithm(
    name="巴特沃斯带阻滤波器",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "linear", "notch"],
)
class BandStopButterworth(BaseAlgorithm):
    """巴特沃斯带阻滤波器"""

    default_params: ClassVar[Dict[str, Any]] = {
        "low_cutoff": 48.0,
        "high_cutoff": 52.0,
        "order": 4,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        low = params["low_cutoff"]
        high = params["high_cutoff"]
        order = params["order"]

        nyquist = sample_rate / 2.0
        normalized = [low / nyquist, high / nyquist]

        sos = scipy_signal.butter(order, normalized, btype="bandstop", output="sos")
        return scipy_signal.sosfiltfilt(sos, signal, axis=-1)


@register_algorithm(
    name="IIR陷波滤波器",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "notch", "powerline"],
)
class NotchIIR(BaseAlgorithm):
    """IIR陷波滤波器，主要用于去除工频50/60Hz干扰"""

    default_params: ClassVar[Dict[str, Any]] = {
        "notch_freq": 50.0,  # 陷波频率 (Hz)
        "quality_factor": 30.0,  # 品质因数
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        notch_freq = params["notch_freq"]
        q = params["quality_factor"]

        nyquist = sample_rate / 2.0
        w0 = notch_freq / nyquist

        b, a = scipy_signal.iirnotch(w0, q)
        return scipy_signal.filtfilt(b, a, signal, axis=-1)
