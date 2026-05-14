"""
频域降噪算法集合：FFT理想滤波器、频谱减法、倒谱分析、同态滤波。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="FFT理想低通滤波",
    category=AlgorithmCategory.FREQ_DOMAIN,
    complexity=AlgorithmComplexity.LOW,
    tags=["frequency-domain", "ideal", "simple"],
)
class FftIdealLowPass(BaseAlgorithm):
    """FF域的理想低通滤波器，直接置零高频分量"""

    default_params: ClassVar[Dict[str, Any]] = {
        "cutoff_freq": 1000.0,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        cutoff = float(params["cutoff_freq"])

        n = signal.shape[-1]
        freqs = fftfreq(n, 1.0 / sample_rate)
        mask = np.abs(freqs) <= cutoff

        spectrum = fft(signal, axis=-1)
        spectrum[..., ~mask] = 0
        return np.real(ifft(spectrum, axis=-1))


@register_algorithm(
    name="频谱减法",
    category=AlgorithmCategory.FREQ_DOMAIN,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["audio", "noise-reduction", "speech"],
)
class SpectralSubtraction(BaseAlgorithm):
    """频谱减法降噪，适用于平稳噪声环境"""

    default_params: ClassVar[Dict[str, Any]] = {
        "noise_estimation_fraction": 0.1,  # 使用前百分之几估计噪声谱
        "subtraction_factor": 1.0,        # 减除系数
        "floor_factor": 0.01,             # 频谱下限
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        noise_frac = float(params["noise_estimation_fraction"])
        sub_factor = float(params["subtraction_factor"])
        floor = float(params["floor_factor"])

        n = signal.shape[-1]
        n_noise = max(int(n * noise_frac), 10)

        spectrum = fft(signal, axis=-1)
        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        # 用前 n_noise 个点估计噪声幅度谱
        noise_mag = np.mean(magnitude[..., :n_noise], axis=-1, keepdims=True)

        # 频谱减除
        magnitude_clean = np.maximum(magnitude - sub_factor * noise_mag, floor * noise_mag)
        spectrum_clean = magnitude_clean * np.exp(1j * phase)
        return np.real(ifft(spectrum_clean, axis=-1))


@register_algorithm(
    name="倒谱分析滤波",
    category=AlgorithmCategory.FREQ_DOMAIN,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["cepstral", "homomorphic", "echo-removal"],
)
class CepstralFilter(BaseAlgorithm):
    """倒谱分析滤波，在倒谱域分离信号与噪声"""

    default_params: ClassVar[Dict[str, Any]] = {
        "quefrency_cutoff": 0.001,  # 倒频率截止（秒），保留低倒频率成分
        "lifter_type": "lowpass",   # lowpass / highpass
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        q_cutoff = float(params["quefrency_cutoff"])
        lift_type = params["lifter_type"]

        n = signal.shape[-1]
        spectrum = fft(signal, axis=-1)
        log_spectrum = np.log(np.abs(spectrum) + 1e-12)
        cepstrum = np.real(fft(log_spectrum, axis=-1))

        # 倒谱滤波（低时/liftering）
        cutoff_idx = int(q_cutoff * sample_rate)
        if lift_type == "lowpass":
            cepstrum[..., cutoff_idx:] = 0
        else:
            cepstrum[..., :cutoff_idx] = 0

        log_spectrum_clean = np.real(ifft(cepstrum, axis=-1))
        spectrum_clean = np.exp(log_spectrum_clean) * np.exp(1j * np.angle(spectrum))
        return np.real(ifft(spectrum_clean, axis=-1))


@register_algorithm(
    name="同态滤波",
    category=AlgorithmCategory.FREQ_DOMAIN,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["homomorphic", "multiplicative-noise", "illumination"],
)
class HomomorphicFilter(BaseAlgorithm):
    """同态滤波，处理乘性噪声"""

    default_params: ClassVar[Dict[str, Any]] = {
        "low_cutoff": 0.1,
        "high_cutoff": 0.9,
        "order": 2,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        low = float(params["low_cutoff"])
        high = float(params["high_cutoff"])
        order = int(params["order"])

        # 取对数（乘性→加性）
        log_signal = np.log(np.abs(signal) + 1e-12)

        # 设计高通滤波器（突出高频细节）
        sos = scipy_signal.butter(order, [low, high], btype="band", output="sos")
        filtered = scipy_signal.sosfiltfilt(sos, log_signal, axis=-1)

        # 指数变换还原
        return np.exp(filtered) * np.sign(signal + 1e-12)
