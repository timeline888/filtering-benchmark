"""
信号预处理流水线。

支持：去直流、重采样、分段、标准化。
"""

from typing import Optional

import numpy as np
from scipy import signal as scipy_signal
from loguru import logger

from ..core.types import SignalData


class SignalPreprocessor:
    """信号预处理器"""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}

    def process(self, signal_data: SignalData) -> SignalData:
        """
        运行预处理流水线。

        Args:
            signal_data: 原始信号数据

        Returns:
            预处理后的信号数据
        """
        data = signal_data.data.copy()
        fs = signal_data.sample_rate

        # 1. 去直流
        if self.config.get("remove_dc", True):
            data = self._remove_dc(data)
            logger.debug(f"去直流完成")

        # 2. 重采样
        target_fs = self.config.get("resample")
        if target_fs and target_fs != fs:
            data, fs = self._resample(data, fs, target_fs)
            logger.debug(f"重采样: {fs}Hz -> {target_fs}Hz")

        # 3. 分段
        segment_duration = self.config.get("segment_duration_sec")
        if segment_duration:
            overlap = self.config.get("segment_overlap", 0.0)
            segments = self._segment(data, fs, segment_duration, overlap)
            # 处理第一个分段
            data = segments[0] if len(segments) > 0 else data

        # 4. 标准化
        if self.config.get("normalize", False):
            data = self._normalize(data)
            logger.debug("标准化完成")

        return SignalData(
            data=data,
            sample_rate=fs,
            channel_names=signal_data.channel_names,
            metadata=signal_data.metadata,
            source_path=signal_data.source_path,
            unit=signal_data.unit,
        )

    @staticmethod
    def _remove_dc(data: np.ndarray) -> np.ndarray:
        """去除直流分量"""
        return data - np.mean(data, axis=-1, keepdims=True)

    @staticmethod
    def _resample(data: np.ndarray, old_fs: float, new_fs: float) -> tuple:
        """重采样"""
        from scipy import signal as scipy_signal
        n_samples = int(data.shape[-1] * new_fs / old_fs)
        resampled = scipy_signal.resample(data, n_samples, axis=-1)
        return resampled.astype(data.dtype), new_fs

    @staticmethod
    def _segment(data: np.ndarray, fs: float, duration: float, overlap: float) -> list:
        """信号分段"""
        n_per_seg = int(duration * fs)
        step = int(n_per_seg * (1 - overlap))
        segments = []
        for start in range(0, data.shape[-1] - n_per_seg + 1, step):
            segments.append(data[..., start:start + n_per_seg])
        return segments

    @staticmethod
    def _normalize(data: np.ndarray) -> np.ndarray:
        """Z-score 标准化"""
        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std = np.where(std < 1e-12, 1.0, std)
        return (data - mean) / std
