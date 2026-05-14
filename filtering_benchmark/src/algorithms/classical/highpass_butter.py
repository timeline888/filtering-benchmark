"""
高通巴特沃斯滤波器。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="巴特沃斯高通滤波器",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "linear"],
)
class HighPassButterworth(BaseAlgorithm):
    """巴特沃斯高通滤波器，用于去除低频漂移和趋势项"""

    default_params: ClassVar[Dict[str, Any]] = {
        "cutoff_freq": 10.0,  # 截止频率 (Hz)
        "order": 4,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        cutoff = params["cutoff_freq"]
        order = params["order"]

        nyquist = sample_rate / 2.0
        normalized_cutoff = cutoff / nyquist

        if normalized_cutoff >= 1.0:
            raise ValueError(f"截止频率 {cutoff}Hz 超过奈奎斯特频率 {nyquist}Hz")

        sos = scipy_signal.butter(order, normalized_cutoff, btype="high", output="sos")
        return scipy_signal.sosfiltfilt(sos, signal, axis=-1)
