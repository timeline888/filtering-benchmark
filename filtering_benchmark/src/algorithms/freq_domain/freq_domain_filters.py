"""
频域降噪算法集合：FFT理想滤波器、频谱减法、倒谱分析、同态滤波。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import fft, ifft, fftfreq

from ...algorithms.base import BaseAlgorithm, _to_stereo, _from_stereo
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


@register_algorithm(
    name="多带频谱减法",
    category=AlgorithmCategory.FREQ_DOMAIN,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["audio", "noise-reduction", "multi-band", "speech"],
)
class MultiBandSpectralSubtraction(BaseAlgorithm):
    """多带频谱减法降噪，将频谱划分为多个子带独立处理。

    相比标准频谱减法（4.2节）在全频带使用统一过减因子，
    多带频谱减法对各子带独立估计噪声和过减因子，
    更好地适应非平坦噪声频谱，减少音乐噪声。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "n_bands": 4,
        "over_subtraction_factor": 2.0,
        "noise_floor": 0.01,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_bands = int(params["n_bands"])
        over_sub = float(params["over_subtraction_factor"])
        noise_floor = float(params["noise_floor"])

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels, n_samples = signal.shape
        output = np.zeros_like(signal)

        for ch in range(n_channels):
            output[ch] = self._multi_band_sub(signal[ch], sample_rate,
                                                n_bands, over_sub, noise_floor)

        return _from_stereo(output, orig_signal)

    def _multi_band_sub(
        self, sig: np.ndarray, fs: float,
        n_bands: int, over_sub: float, noise_floor: float
    ) -> np.ndarray:
        N = len(sig)
        f, t, Zxx = scipy_signal.stft(sig, fs=fs, nperseg=256, noverlap=128)
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)

        # 噪声估计（取前10帧）
        noise_est = np.mean(mag[:, :10]**2, axis=1, keepdims=True)

        # 划分子带
        n_freq = len(f)
        band_size = n_freq // n_bands
        enhanced = np.zeros_like(mag)

        for b in range(n_bands):
            idx_low = b * band_size
            idx_high = n_freq if b == n_bands - 1 else (b + 1) * band_size

            sub_mag = mag[idx_low:idx_high, :]
            sub_noise = noise_est[idx_low:idx_high]

            # 子带 SNR 估计（取整个子带的平均功率）
            sub_power = np.mean(sub_mag**2)  # scalar
            noise_power = np.mean(sub_noise) + 1e-10  # scalar
            snr_db = 10 * np.log10(sub_power / noise_power)

            # 自适应过减因子（整带一致）
            alpha = np.clip(over_sub - 0.15 * snr_db, 1.0, 5.0)
            enhanced[idx_low:idx_high, :] = np.maximum(
                sub_mag**2 - alpha * sub_noise,
                noise_floor * sub_noise
            )

        mag_enhanced = np.sqrt(enhanced)
        _, x_est = scipy_signal.istft(mag_enhanced * np.exp(1j * phase), fs=fs)
        return x_est[:N]
