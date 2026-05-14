"""
时域降噪算法集合：滑动平均、中值滤波、S-G滤波、EWMA、高斯滤波。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="滑动平均滤波",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "smooth", "simple"],
)
class MovingAverageFilter(BaseAlgorithm):
    """滑动平均滤波，用窗口内均值替代中心值"""

    default_params: ClassVar[Dict[str, Any]] = {
        "window_size": 5,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        w = int(params["window_size"])
        if w < 2:
            return signal.copy()
        kernel = np.ones(w) / w
        return scipy_signal.convolve(signal, kernel[np.newaxis, :], mode="same", method="auto")


@register_algorithm(
    name="中值滤波",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "impulse-noise", "robust"],
)
class MedianFilter(BaseAlgorithm):
    """中值滤波，对脉冲/极盐噪声有极好的抑制效果"""

    default_params: ClassVar[Dict[str, Any]] = {
        "kernel_size": 5,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        k = int(params["kernel_size"])
        k = k + 1 if k % 2 == 0 else k  # 确保奇数
        return scipy_signal.medfilt(signal, kernel_size=k)


@register_algorithm(
    name="Savitzky-Golay滤波",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["smooth", "peak-preserving", "polynomial"],
)
class SavitzkyGolayFilter(BaseAlgorithm):
    """Savitzky-Golay平滑滤波，保留信号峰值特征的同时平滑噪声"""

    default_params: ClassVar[Dict[str, Any]] = {
        "window_length": 11,
        "polyorder": 3,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        w = int(params["window_length"])
        p = int(params["polyorder"])
        w = w + 1 if w % 2 == 0 else w
        w = max(w, p + 2)
        return scipy_signal.savgol_filter(signal, w, p, axis=-1)


@register_algorithm(
    name="指数加权移动平均",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "recursive", "trend"],
)
class EWMABase(BaseAlgorithm):
    """指数加权移动平均 (EWMA)，对近期数据赋予更大权重"""

    default_params: ClassVar[Dict[str, Any]] = {
        "alpha": 0.3,  # 平滑因子 [0, 1]，越大对近期数据越敏感
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        alpha = float(params["alpha"])
        alpha = np.clip(alpha, 0.01, 0.99)

        result = np.zeros_like(signal)
        result[..., 0] = signal[..., 0]
        for i in range(1, signal.shape[-1]):
            result[..., i] = alpha * signal[..., i] + (1 - alpha) * result[..., i - 1]
        return result


@register_algorithm(
    name="高斯滤波",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["smooth", "gaussian", "lowpass"],
)
class GaussianFilter(BaseAlgorithm):
    """高斯滤波，使用高斯核的1D卷积平滑"""

    default_params: ClassVar[Dict[str, Any]] = {
        "sigma": 2.0,  # 高斯核标准差
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        sigma = float(params["sigma"])
        return gaussian_filter1d(signal, sigma=sigma, axis=-1, mode="reflect")
