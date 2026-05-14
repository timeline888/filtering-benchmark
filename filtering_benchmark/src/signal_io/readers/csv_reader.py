"""
CSV/TXT 格式信号读取器。
"""

from pathlib import Path
from typing import Optional, Set

import numpy as np

from ...core.types import SignalData
from ..base_reader import BaseSignalReader


class CsvSignalReader(BaseSignalReader):
    """CSV/TXT 信号读取器"""

    supported_extensions: Set[str] = {".csv", ".txt"}

    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取 CSV/TXT 文件。
        默认每列为一个通道。
        """
        import pandas as pd

        df = pd.read_csv(file_path, **kwargs)
        data = df.values.T  # 转置为 (n_channels, n_samples)
        data = data.astype(np.float64)

        # 列名作为通道名
        channel_names = list(df.columns)

        sample_rate = self._infer_sample_rate({}, file_path)

        return SignalData(
            data=data,
            sample_rate=sample_rate,
            channel_names=channel_names,
            source_path=file_path,
            unit="m/s²",
        )


# 注册到工厂
from ..reader_factory import SignalReaderFactory
SignalReaderFactory.register(CsvSignalReader)
