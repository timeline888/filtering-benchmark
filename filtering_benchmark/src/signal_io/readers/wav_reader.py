"""
WAV/FLAC 格式信号读取器（声学信号）。
"""

from pathlib import Path
from typing import Set

import numpy as np

from ...core.types import SignalData
from ..base_reader import BaseSignalReader


class WavSignalReader(BaseSignalReader):
    """WAV 声学信号读取器"""

    supported_extensions: Set[str] = {".wav", ".flac", ".mp3"}

    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取 WAV 文件。
        """
        import soundfile as sf

        data, sample_rate = sf.read(file_path, **kwargs)

        # 确保为 2D (n_channels, n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T

        data = data.astype(np.float64)

        n_chan = data.shape[0]
        channel_names = [f"ch_{i}" for i in range(n_chan)]

        return SignalData(
            data=data,
            sample_rate=float(sample_rate),
            channel_names=channel_names,
            source_path=file_path,
            unit="Pa",
        )


from ..reader_factory import SignalReaderFactory
SignalReaderFactory.register(WavSignalReader)
