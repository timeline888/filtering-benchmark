"""
时域降噪算法集合：滑动平均、中值滤波、S-G滤波、EWMA、高斯滤波。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import gaussian_filter1d

from ...algorithms.base import BaseAlgorithm, _to_stereo, _from_stereo
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

        # medfilt对多维输入做多维滤波，因此需逐通道处理1D信号
        orig_signal = signal
        signal = _to_stereo(signal)
        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)
        for ch in range(n_channels):
            output[ch] = scipy_signal.medfilt(signal[ch], kernel_size=k)
        return _from_stereo(output, orig_signal)


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


@register_algorithm(
    name="自适应中值滤波",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "impulse-noise", "adaptive-window"],
)
class AdaptiveMedianFilter(BaseAlgorithm):
    """自适应中值滤波，窗口大小根据局部统计特性动态调整。

    比固定中值滤波更好地处理非平稳噪声，在高密度脉冲噪声
    环境下保持边缘细节。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "max_window_size": 15,
        "initial_window_size": 3,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        max_win = int(params["max_window_size"])
        init_win = int(params["initial_window_size"])
        max_win = max_win + 1 if max_win % 2 == 0 else max_win
        init_win = max(3, init_win + 1 if init_win % 2 == 0 else init_win)

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            output[ch] = self._adaptive_median_1d(signal[ch], max_win, init_win)

        return _from_stereo(output, orig_signal)

    def _adaptive_median_1d(self, sig: np.ndarray, max_win: int, init_win: int) -> np.ndarray:
        n = len(sig)
        output = sig.copy()
        half_init = init_win // 2

        for i in range(half_init, n - half_init):
            win_size = init_win
            while win_size <= max_win:
                half = win_size // 2
                window = sig[max(0, i - half):min(n, i + half + 1)]
                z_med = float(np.median(window))
                z_min = float(np.min(window))
                z_max = float(np.max(window))

                # 层级 A: 中值不是脉冲
                if z_med > z_min and z_med < z_max:
                    # 层级 B: 当前像素不是脉冲
                    if sig[i] > z_min and sig[i] < z_max:
                        output[i] = sig[i]
                    else:
                        output[i] = z_med
                    break
                win_size += 2

            if win_size > max_win:
                half = max_win // 2
                output[i] = float(np.median(
                    sig[max(0, i - half):min(n, i + half + 1)]
                ))

        return output


@register_algorithm(
    name="加权滑动平均",
    category=AlgorithmCategory.TIME_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "smooth", "weighted", "lowpass"],
)
class WeightedMovingAverage(BaseAlgorithm):
    """加权滑动平均滤波，使用非均匀权重获得更好的频域特性。

    相比标准滑动平均(均匀权重)，三角窗权重的旁瓣衰减
    可达-25dB（标准MA仅-13dB）。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "window_size": 5,
        "weight_type": "triangular",  # triangular | cosine | hamming
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        window_size = int(params["window_size"])
        weight_type = str(params["weight_type"])

        if weight_type == "triangular":
            weights = np.array([min(k + 1, window_size - k)
                                for k in range(window_size)], dtype=float)
        elif weight_type == "cosine":
            weights = np.cos(np.linspace(-np.pi / 2, np.pi / 2, window_size)) + 1
        elif weight_type == "hamming":
            weights = np.hamming(window_size)
        else:
            weights = np.ones(window_size)
        weights /= weights.sum()

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)
        for ch in range(n_channels):
            output[ch] = scipy_signal.convolve(signal[ch], weights,
                                                 mode="same", method="auto")

        return _from_stereo(output, orig_signal)
