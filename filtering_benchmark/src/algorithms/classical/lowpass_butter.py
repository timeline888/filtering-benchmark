"""
低通巴特沃斯滤波器 (Low-Pass Butterworth Filter)

基于 scipy.signal.butter 的巴特沃斯低通滤波器实现，
使用 sosfiltfilt 进行零相位滤波，避免相位偏移。

参考书：第3章第1节 - 低通滤波器
"""

from typing import Any, ClassVar, Dict, List

import numpy as np
from scipy import signal as scipy_signal

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="巴特沃斯低通滤波器",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "linear"],
)
class LowPassButterworth(BaseAlgorithm):
    """巴特沃斯低通滤波器"""

    name: ClassVar[str] = "巴特沃斯低通滤波器"
    category: ClassVar[AlgorithmCategory] = AlgorithmCategory.CLASSICAL_FILTER
    complexity: ClassVar[AlgorithmComplexity] = AlgorithmComplexity.LOW
    default_params: ClassVar[Dict[str, Any]] = {
        "cutoff_freq": 1000.0,  # 截止频率 (Hz)
        "order": 4,             # 滤波器阶数
        "filter_type": "lowpass",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        cutoff = params["cutoff_freq"]
        order = params["order"]

        # 归一化截止频率
        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0:
            raise ValueError(f"截止频率 {cutoff}Hz 超过奈奎斯特频率 {nyquist}Hz")

        # 设计并应用巴特沃斯低通滤波器
        sos = scipy_signal.butter(order, normalized_cutoff, btype="low", output="sos")
        return scipy_signal.sosfiltfilt(sos, signal, axis=-1)
