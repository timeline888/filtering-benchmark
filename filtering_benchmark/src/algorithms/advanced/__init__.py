"""
前沿/高级滤波降噪算法集合。

包含以下前沿方法:
1. 压缩感知降噪 (Compressed Sensing)
2. 随机共振 (Stochastic Resonance)
3. 形态学滤波 (Morphological Filtering)
4. 全变分去噪 (Total Variation)
5. 非局部均值降噪 (Non-Local Means)
6. 图信号处理降噪 (Graph Signal Processing)
7. 分形降噪 (Fractal Denoising)
"""

from .advanced_denoise import (
    CompressedSensingDenoise,
    StochasticResonance,
    MorphologicalFilter,
    TotalVariationDenoise,
    NonLocalMeansDenoise,
    GraphSignalDenoise,
    FractalDenoise,
)

__all__ = [
    "CompressedSensingDenoise",
    "StochasticResonance",
    "MorphologicalFilter",
    "TotalVariationDenoise",
    "NonLocalMeansDenoise",
    "GraphSignalDenoise",
    "FractalDenoise",
]
