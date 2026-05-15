"""
包络谱分析模块。

核心功能：
- Hilbert变换求取信号包络
- 包络的FFT得到包络谱
- 故障特征频率匹配与识别
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as scipy_signal
from loguru import logger

from ..core.types import EnvelopeResult
from ..core.constants import BEARING_PARAMETERS, FaultFrequencyType


class EnvelopeAnalyzer:
    """包络谱分析器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def analyze(self, signal: np.ndarray, sample_rate: float) -> EnvelopeResult:
        """
        对信号进行包络谱分析。

        Args:
            signal: 输入信号，shape (n_samples,) 或 (n_channels, n_samples)
            sample_rate: 采样率 (Hz)

        Returns:
            EnvelopeResult 包络谱分析结果
        """
        if signal.ndim == 1:
            signal = signal.reshape(1, -1)

        # 1. 可选带通滤波
        bp_config = self.config.get("bandpass", {})
        if bp_config.get("enabled", False):
            low = bp_config.get("low_cutoff", 2000)
            high = bp_config.get("high_cutoff", 5000)
            order = bp_config.get("order", 4)
            signal = self._bandpass_filter(signal, low, high, sample_rate, order)

        # 2. Hilbert变换 → 包络信号
        envelope = self._hilbert_envelope(signal)

        # 3. FFT → 包络谱
        n_fft = self.config.get("n_fft", None)
        spectrum, freq_axis = self._compute_spectrum(envelope, sample_rate, n_fft)

        # 4. 提取主导频率
        dominant_freqs = self._find_dominant_freqs(spectrum, freq_axis)

        # 5. 匹配故障特征频率
        fault_matches = self._match_fault_freqs(dominant_freqs, sample_rate)

        return EnvelopeResult(
            algorithm_id="",
            envelope_signal=envelope[0] if envelope.shape[0] == 1 else envelope,
            envelope_spectrum=spectrum[0] if spectrum.shape[0] == 1 else spectrum,
            freq_axis=freq_axis,
            dominant_freqs=dominant_freqs,
            fault_freq_matches=fault_matches,
        )

    def analyze_batch(self, signals: Dict[str, np.ndarray], sample_rate: float) -> Dict[str, EnvelopeResult]:
        """批量分析多个信号的包络谱"""
        results = {}
        for algo_id, sig in signals.items():
            try:
                results[algo_id] = self.analyze(sig, sample_rate)
                results[algo_id].algorithm_id = algo_id
            except Exception as e:
                logger.error(f"包络谱分析失败 [{algo_id}]: {e}")
        return results

    @staticmethod
    def _bandpass_filter(signal: np.ndarray, low: float, high: float,
                         fs: float, order: int = 4) -> np.ndarray:
        """带通滤波"""
        nyq = fs / 2.0
        if low <= 0 or high >= nyq or high <= low:
            return signal
        sos = scipy_signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")
        return scipy_signal.sosfiltfilt(sos, signal, axis=-1)

    @staticmethod
    def _hilbert_envelope(signal: np.ndarray) -> np.ndarray:
        """Hilbert变换 → 解析信号幅值 → 包络"""
        analytic = scipy_signal.hilbert(signal, axis=-1)
        return np.abs(analytic)

    @staticmethod
    def _compute_spectrum(envelope: np.ndarray, fs: float,
                          n_fft: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算包络信号的幅值谱（自动去除直流分量）"""
        n = envelope.shape[-1]
        if n_fft is not None and n_fft > n:
            n = n_fft
        elif n_fft is not None:
            n = n_fft

        # 去除包络的直流分量，避免DC峰值淹没频谱细节
        envelope_ac = envelope - np.mean(envelope, axis=-1, keepdims=True)
        spectrum = np.abs(np.fft.rfft(envelope_ac, n=n, axis=-1))
        freq_axis = np.fft.rfftfreq(n, 1.0 / fs)
        return spectrum, freq_axis

    @staticmethod
    def _find_dominant_freqs(spectrum: np.ndarray, freq_axis: np.ndarray,
                              n_peaks: int = 20) -> List[Tuple[float, float]]:
        """寻找主导频率（峰值检测）"""
        peaks = []
        for ch_spec in spectrum:
            from scipy.signal import find_peaks
            # 忽略DC分量
            half_idx = len(freq_axis) // 2
            peaks_idx, properties = find_peaks(ch_spec[:half_idx], height=np.mean(ch_spec[:half_idx]) * 2)
            if len(peaks_idx) == 0:
                peaks_idx = [np.argmax(ch_spec[:half_idx])]
            ch_peaks = [
                (freq_axis[idx], ch_spec[idx])
                for idx in peaks_idx[:n_peaks]
            ]
            peaks.extend(ch_peaks)

        # 按幅值降序排列
        peaks.sort(key=lambda x: x[1], reverse=True)
        return peaks[:n_peaks]

    @staticmethod
    def calculate_fault_freqs(params: dict, shaft_rpm: float) -> Dict[str, float]:
        """
        根据轴承参数计算故障特征频率。

        Args:
            params: 包含 pitch_diameter, roller_diameter, n_rollers, contact_angle
            shaft_rpm: 转速 (RPM)

        Returns:
            {freq_name: freq_value} 字典
        """
        shaft_freq = shaft_rpm / 60.0
        pd = params.get("pitch_diameter", 0)
        rd = params.get("roller_diameter", 0)
        n = params.get("n_rollers", 0)
        angle = np.deg2rad(params.get("contact_angle", 0))

        if pd <= 0 or rd <= 0 or n <= 0:
            return {"fr": shaft_freq}

        bpfo = n * shaft_freq / 2 * (1 - rd / pd * np.cos(angle))
        bpfi = n * shaft_freq / 2 * (1 + rd / pd * np.cos(angle))
        bsf = pd * shaft_freq / (2 * rd) * (1 - (rd / pd * np.cos(angle)) ** 2)
        ftf = shaft_freq / 2 * (1 - rd / pd * np.cos(angle))

        return {
            "bpfo": bpfo,
            "bpfi": bpfi,
            "bsf": bsf,
            "ftf": ftf,
            "fr": shaft_freq,
        }

    def _match_fault_freqs(self, dominant_freqs: List[Tuple[float, float]],
                           sample_rate: float) -> Dict[str, float]:
        """匹配故障特征频率"""
        fault_config = self.config.get("fault_frequencies", {})
        mode = fault_config.get("mode", "manual")

        target_freqs = {}
        if mode == "manual":
            manual = fault_config.get("manual", {})
            for key in ["bpfo", "bpfi", "bsf", "ftf", "gmf", "fr"]:
                val = manual.get(key)
                if val:
                    target_freqs[key] = float(val)
        elif mode == "auto":
            auto_config = fault_config.get("auto", {})
            bearing_params = auto_config.get("bearing_params", {})
            shaft_rpm = bearing_params.get("shaft_rpm", 0)

            if bearing_params.get("bearing_type"):
                std_type = bearing_params["bearing_type"]
                if std_type in BEARING_PARAMETERS:
                    std_params = BEARING_PARAMETERS[std_type]
                    bearing_params = {**std_params, **bearing_params}

            if shaft_rpm > 0:
                target_freqs = self.calculate_fault_freqs(bearing_params, shaft_rpm)

        matches = {}
        for name, target_freq in target_freqs.items():
            best_amp = 0
            for freq, amp in dominant_freqs:
                if abs(freq - target_freq) / max(target_freq, 1) < 0.02:
                    best_amp = max(best_amp, amp)
            if best_amp > 0:
                matches[name] = best_amp

        return matches
