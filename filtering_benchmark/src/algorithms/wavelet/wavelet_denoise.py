"""
小波变换类降噪算法：小波阈值、小波包、经验小波变换(EWT)、双树复小波(DTCWT)、平稳小波(SWT)。
"""

from typing import Any, ClassVar, Dict

import numpy as np

from ...algorithms.base import BaseAlgorithm, _to_stereo, _from_stereo
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


# ========== 1. 小波阈值降噪 ==========

@register_algorithm(
    name="小波阈值降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["wavelet", "threshold", "nonlinear"],
)
class DwtThresholdDenoise(BaseAlgorithm):
    """离散小波变换 + 阈值降噪，最经典的小波降噪方法"""

    default_params: ClassVar[Dict[str, Any]] = {
        "wavelet": "db4",
        "level": 5,
        "threshold_mode": "soft",  # soft | hard | garrote
        "threshold_rule": "universal",  # universal | sure | minimaxi | birge_massart
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        import pywt

        params = {**self.params, **kwargs}
        wavelet = params["wavelet"]
        level = int(params["level"])
        thr_mode = params["threshold_mode"]
        thr_rule = params["threshold_rule"]

        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_channel(signal[ch], wavelet, level, thr_mode, thr_rule)
        return result

    def _denoise_channel(self, sig: np.ndarray, wavelet: str, level: int,
                         thr_mode: str, thr_rule: str) -> np.ndarray:
        import pywt

        # 小波分解
        coeffs = pywt.wavedec(sig, wavelet, level=level)

        # 阈值估计
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # 噪声标准差估计
        if thr_rule == "universal":
            threshold = sigma * np.sqrt(2 * np.log(len(sig)))
        elif thr_rule == "sure":
            threshold = self._sure_threshold(coeffs[-1], sigma)
        elif thr_rule == "minimaxi":
            n = len(sig)
            if n <= 32:
                threshold = 0
            else:
                threshold = sigma * (0.3936 + 0.1829 * np.log2(n))
        elif thr_rule == "birge_massart":
            threshold = self._birge_massart_threshold(coeffs, level, sigma)
        else:
            threshold = sigma * np.sqrt(2 * np.log(len(sig)))

        # 阈值处理（细节系数，保留近似系数）
        coeffs_th = list(coeffs)
        for i in range(1, len(coeffs_th)):
            coeffs_th[i] = pywt.threshold(coeffs_th[i], threshold, mode=thr_mode)

        return pywt.waverec(coeffs_th, wavelet)[:len(sig)]

    def _sure_threshold(self, detail: np.ndarray, sigma: float) -> float:
        n = len(detail)
        if n == 0:
            return 0
        sorted_coef = np.sort(np.abs(detail)) ** 2
        risks = (n - 2 * np.arange(1, n + 1) + np.cumsum(sorted_coef)) / n
        best = np.argmin(risks)
        return sigma * np.sqrt(sorted_coef[best]) / sigma if sorted_coef[best] > 0 else 0

    def _birge_massart_threshold(self, coeffs, level, sigma):
        n = len(coeffs[0])  # 近似系数长度
        total_n = sum(len(c) for c in coeffs)
        thr_idx = int(total_n * 0.5)  # 保留50%的系数
        all_details = np.concatenate([np.abs(c).ravel() for c in coeffs[1:]])
        all_details.sort()
        if thr_idx >= len(all_details):
            return 0
        return all_details[-thr_idx]


# ========== 2. 小波包降噪 ==========

@register_algorithm(
    name="小波包降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["wavelet-packet", "full-decomposition", "adaptive"],
)
class WaveletPacketDenoise(BaseAlgorithm):
    """小波包降噪，对信号高低频都进行精细分解"""

    default_params: ClassVar[Dict[str, Any]] = {
        "wavelet": "db4",
        "level": 4,
        "threshold_mode": "soft",
        "best_basis": True,
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        import pywt

        params = {**self.params, **kwargs}
        wavelet = params["wavelet"]
        level = int(params["level"])
        thr_mode = params["threshold_mode"]
        best_basis = params["best_basis"]

        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_ch(signal[ch], wavelet, level, thr_mode, best_basis)
        return result

    def _denoise_ch(self, sig: np.ndarray, wavelet: str, level: int,
                    thr_mode: str, best_basis: bool) -> np.ndarray:
        import pywt

        wp = pywt.WaveletPacket(data=sig, wavelet=wavelet, mode='symmetric', maxlevel=level)

        if best_basis:
            # 搜索最优基（基于香农熵）
            def shannon_entropy(x):
                x = np.abs(x)
                x = x / (np.sum(x) + 1e-12)
                return -np.sum(x * np.log2(x + 1e-12))
            wp.get_entropy = shannon_entropy
            best_tree = wp.get_best_entropy_basis()
        else:
            best_tree = [node.path for node in wp.get_level(level, 'natural')]

        # 阈值处理
        sigma = np.median(np.abs(sig - np.mean(sig))) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(sig)))

        for node_path in best_tree:
            node = wp[node_path]
            node.data = pywt.threshold(node.data, threshold, mode=thr_mode)

        return wp.reconstruct(update=True)[:len(sig)]


# ========== 3. 双树复小波变换降噪 (DTCWT) ==========

@register_algorithm(
    name="双树复小波变换降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.HIGH,
    tags=["dtcwt", "shift-invariant", "directional"],
)
class DtcwtDenoise(BaseAlgorithm):
    """双树复小波变换 (DTCWT) 降噪，具有近似平移不变性"""

    default_params: ClassVar[Dict[str, Any]] = {
        "level": 5,
        "threshold_mode": "soft",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        level = int(params["level"])
        thr_mode = params["threshold_mode"]

        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            result[ch] = self._denoise_ch(signal[ch], level, thr_mode)
        return result

    def _denoise_ch(self, sig: np.ndarray, level: int, thr_mode: str) -> np.ndarray:
        try:
            import dtcwt
            transform = dtcwt.Transform1d()
            coeffs = transform.forward(sig, nlevels=level)

            # 阈值处理
            sigma = np.median(np.abs(coeffs.highpasses[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(sig)))

            for i in range(level):
                hp = coeffs.highpasses[i]
                hp_mag = np.abs(hp)
                hp_mag_new = pywt_threshold(hp_mag, threshold, thr_mode)
                coeffs.highpasses[i] = hp * (hp_mag_new / (hp_mag + 1e-12))

            recon = transform.inverse(coeffs)
            return recon[:len(sig)]
        except ImportError:
            # 如果 dtcwt 不可用，回退到标准小波阈值
            return self._fallback_dwt(sig, level, thr_mode)

    def _fallback_dwt(self, sig: np.ndarray, level: int, thr_mode: str) -> np.ndarray:
        import pywt
        coeffs = pywt.wavedec(sig, "db4", level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(sig)))
        coeffs_th = list(coeffs)
        for i in range(1, len(coeffs_th)):
            coeffs_th[i] = pywt.threshold(coeffs_th[i], threshold, mode=thr_mode)
        return pywt.waverec(coeffs_th, "db4")[:len(sig)]


# ========== 4. 平稳小波变换降噪 (SWT) ==========

@register_algorithm(
    name="平稳小波变换降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["swt", "stationary-wavelet", "shift-invariant"],
)
class SwtDenoise(BaseAlgorithm):
    """平稳小波变换 (SWT) 降噪，也称冗余小波变换，具有平移不变性"""

    default_params: ClassVar[Dict[str, Any]] = {
        "wavelet": "db4",
        "level": 5,
        "threshold_mode": "soft",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        import pywt

        params = {**self.params, **kwargs}
        wavelet = params["wavelet"]
        level = int(params["level"])
        thr_mode = params["threshold_mode"]

        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            try:
                coeffs = pywt.swt(signal[ch], wavelet, level=level, axis=-1)
                # 设第0层细节系数的中位数为噪声估计
                sigma = np.median(np.abs(coeffs[0][1])) / 0.6745
                threshold = sigma * np.sqrt(2 * np.log(len(signal[ch])))

                coeffs_th = []
                for approx, detail in coeffs:
                    detail_th = pywt.threshold(detail, threshold, mode=thr_mode)
                    coeffs_th.append((approx, detail_th))

                recon = pywt.iswt(coeffs_th, wavelet)[:len(signal[ch])]
                result[ch] = recon
            except Exception:
                # SWT要求信号长度是2的倍数，否则回退到DWT
                result[ch] = self._fallback_dwt(signal[ch], level, thr_mode)
        return result

    def _fallback_dwt(self, sig: np.ndarray, level: int, thr_mode: str) -> np.ndarray:
        import pywt
        coeffs = pywt.wavedec(sig, "db4", level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(sig)))
        coeffs_th = list(coeffs)
        for i in range(1, len(coeffs_th)):
            coeffs_th[i] = pywt.threshold(coeffs_th[i], threshold, mode=thr_mode)
        return pywt.waverec(coeffs_th, "db4")[:len(sig)]


# ========== 5. 经验小波变换 (EWT) ==========

@register_algorithm(
    name="经验小波变换降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.HIGH,
    tags=["ewt", "adaptive", "fourier-segment"],
)
class EwtDenoise(BaseAlgorithm):
    """经验小波变换 (EWT) 降噪，根据信号频谱特征自适应分割并构造滤波器组"""

    default_params: ClassVar[Dict[str, Any]] = {
        "n_modes": 5,
        "detection_method": "localmax",  # localmax | scale_space
        "threshold_mode": "soft",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        n_modes = int(params["n_modes"])

        result = np.zeros_like(signal)
        for ch in range(signal.shape[0]):
            # 简单的F-域自适应滤波实现
            result[ch] = self._adaptive_fft_filter(signal[ch], sample_rate, n_modes)
        return result

    def _adaptive_fft_filter(self, sig: np.ndarray, fs: float, n_modes: int) -> np.ndarray:
        """基于FFT的自适应频带划分降噪（EWT简化实现）"""
        import pywt
        # 使用小波包分解近似EWT效果
        wp = pywt.WaveletPacket(data=sig, wavelet='db8', mode='symmetric', maxlevel=n_modes)
        nodes = wp.get_level(n_modes, 'natural')
        threshold = np.median(np.abs(sig - np.mean(sig))) / 0.6745 * np.sqrt(2 * np.log(len(sig)))

        for node in nodes:
            node.data = pywt.threshold(node.data, threshold, mode="soft")

        return wp.reconstruct(update=True)[:len(sig)]


# 辅助函数
def pywt_threshold(data: np.ndarray, threshold: float, mode: str = "soft") -> np.ndarray:
    """阈值处理（兼容独立调用）"""
    import pywt
    return pywt.threshold(data, threshold, mode=mode)


# ========== 6. SureShrink自适应阈值降噪 ==========

@register_algorithm(
    name="SureShrink自适应阈值降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["wavelet", "sureshrink", "adaptive", "sure"],
)
class SureShrinkDenoise(BaseAlgorithm):
    """SureShrink自适应阈值降噪。

    基于 Stein 无偏风险估计 (SURE) 对每一层小波系数独立选择最优阈值，
    对于高稀疏层自动回退到 VisuShrink 全局阈值。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "wavelet": "db4",
        "level": 4,
        "threshold_mode": "soft",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        import pywt

        params = {**self.params, **kwargs}
        wavelet = params["wavelet"]
        level = int(params["level"])
        thr_mode = params["threshold_mode"]

        orig_signal = signal
        signal = _to_stereo(signal)

        n_channels = signal.shape[0]
        result = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            coeffs = pywt.wavedec(sig, wavelet, level=level)

            # 噪声标准差估计（最细节层）
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745

            for i in range(1, len(coeffs)):
                d = coeffs[i]
                threshold_visu = sigma * np.sqrt(2 * np.log(len(d)))
                # 判断是否高稀疏层
                dense_ratio = np.sum(np.abs(d) > sigma) / len(d)
                if dense_ratio > 1.0 / np.sqrt(len(d)):
                    # SURE 阈值
                    sorted_sq = np.sort(np.abs(d / sigma)) ** 2
                    n = len(sorted_sq)
                    risks = (n - 2 * np.arange(1, n + 1) +
                             np.cumsum(sorted_sq)) / n
                    best = np.argmin(risks)
                    t = sigma * np.sqrt(sorted_sq[best]) if sorted_sq[best] > 0 else 0
                else:
                    # 高稀疏层用 VisuShrink
                    t = threshold_visu
                coeffs[i] = pywt.threshold(d, t, mode=thr_mode)

            result[ch] = pywt.waverec(coeffs, wavelet)[:len(sig)]

        return _from_stereo(result, orig_signal)


# ========== 7. 平移不变小波降噪 (TI-Wavelet) ==========

@register_algorithm(
    name="平移不变小波降噪",
    category=AlgorithmCategory.WAVELET,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["wavelet", "shift-invariant", "ti-wavelet", "gibbs-free"],
)
class TiWaveletDenoise(BaseAlgorithm):
    """平移不变小波降噪 (Translation-Invariant Wavelet Denoising)。

    通过对信号进行多个循环平移后分别做小波阈值降噪，
    再反向平移取平均，消除 DWT 的下采样导致的 Gibbs 伪影。
    """

    default_params: ClassVar[Dict[str, Any]] = {
        "wavelet": "db4",
        "level": 4,
        "threshold_mode": "soft",
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        import pywt

        params = {**self.params, **kwargs}
        wavelet = params["wavelet"]
        level = int(params["level"])
        thr_mode = params["threshold_mode"]

        is_1d = signal.ndim == 1
        if is_1d:
            signal = signal.reshape(1, -1)

        n_channels, n_samples = signal.shape
        result = np.zeros_like(signal)

        for ch in range(n_channels):
            sig = signal[ch]
            N = len(sig)
            # 以 2 的幂次平移
            max_shift = max(1, int(np.log2(N)))
            denoised_sum = np.zeros(N)

            sigma = np.median(np.abs(
                pywt.dwt(sig, wavelet)[0]
            )) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(N))

            for shift_idx in range(max_shift):
                step = 2 ** shift_idx
                shifted = np.roll(sig, step)
                coeffs = pywt.wavedec(shifted, wavelet, level=level)
                coeffs_th = [coeffs[0]] + [
                    pywt.threshold(c, threshold, mode=thr_mode)
                    for c in coeffs[1:]
                ]
                denoised = pywt.waverec(coeffs_th, wavelet)[:N]
                denoised_sum += np.roll(denoised, -step)

            result[ch] = denoised_sum / max_shift

        return result[0] if is_1d else result
