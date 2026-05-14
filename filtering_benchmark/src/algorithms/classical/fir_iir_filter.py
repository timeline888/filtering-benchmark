"""
FIR/IIR 通用滤波器。
"""

from typing import Any, ClassVar, Dict

import numpy as np
from scipy import signal as scipy_signal

from ...algorithms.base import BaseAlgorithm
from ...algorithms.decorators import register_algorithm, log_execution, validate_params
from ...core.types import AlgorithmCategory, AlgorithmComplexity


@register_algorithm(
    name="FIR带通滤波器",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.MEDIUM,
    tags=["linear-phase", "bandpass", "stable"],
)
class FirBandPass(BaseAlgorithm):
    """FIR带通滤波器，采用Kaiser窗设计，线性相位特性，适合相位保真度要求高的场景"""

    default_params: ClassVar[Dict[str, Any]] = {
        "low_cutoff": 500.0,
        "high_cutoff": 5000.0,
        "transition_width": 200.0,  # 过渡带宽度 (Hz)
        "attenuation": 60.0,  # 阻带衰减 (dB)
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        low = params["low_cutoff"]
        high = params["high_cutoff"]
        tw = params["transition_width"]
        atten = params["attenuation"]

        nyquist = sample_rate / 2.0
        # 计算所需的滤波器阶数（Kaiser窗经验公式）
        n, beta = scipy_signal.kaiserord(atten, tw / nyquist)

        # 确保阶数为奇数（线性相位）
        if n % 2 == 0:
            n += 1
        n = max(n, 3)

        taps = scipy_signal.firwin(n, [low, high], window=("kaiser", beta),
                                    pass_zero="bandpass", fs=sample_rate)
        return scipy_signal.filtfilt(taps, [1.0], signal, axis=-1)


@register_algorithm(
    name="IIR带通滤波器（椭圆）",
    category=AlgorithmCategory.CLASSICAL_FILTER,
    complexity=AlgorithmComplexity.LOW,
    tags=["real-time", "bandpass", "efficient"],
)
class IirBandPass(BaseAlgorithm):
    """IIR带通滤波器（椭圆滤波器），比Butterworth更陡的过渡带"""

    default_params: ClassVar[Dict[str, Any]] = {
        "low_cutoff": 500.0,
        "high_cutoff": 5000.0,
        "order": 4,
        "passband_ripple": 0.1,  # 通带纹波 (dB)
        "stopband_attenuation": 40.0,  # 阻带衰减 (dB)
    }

    @log_execution
    @validate_params
    def denoise(self, signal: np.ndarray, sample_rate: float, **kwargs) -> np.ndarray:
        params = {**self.params, **kwargs}
        low = params["low_cutoff"]
        high = params["high_cutoff"]
        order = params["order"]
        rp = params["passband_ripple"]
        rs = params["stopband_attenuation"]

        nyquist = sample_rate / 2.0
        normalized = [low / nyquist, high / nyquist]

        sos = scipy_signal.ellip(order, rp, rs, normalized, btype="band", output="sos")
        return scipy_signal.sosfiltfilt(sos, signal, axis=-1)
