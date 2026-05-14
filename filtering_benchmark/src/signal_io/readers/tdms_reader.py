"""
TDMS (NI LabVIEW) 格式信号读取器。
"""

from pathlib import Path
from typing import Dict, Set

import numpy as np

from ...core.types import SignalData
from ..base_reader import BaseSignalReader


class TdmsSignalReader(BaseSignalReader):
    """TDMS 信号读取器"""

    supported_extensions: Set[str] = {".tdms"}

    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取 NI TDMS 文件。
        """
        from nptdms import TdmsFile

        tdms = TdmsFile.read(file_path, **kwargs)

        # 收集所有通道数据
        all_data = []
        channel_names = []
        sample_rate = 1.0

        for group in tdms.groups():
            for ch in group.channels():
                data = ch.data.astype(np.float64)
                all_data.append(data)
                channel_names.append(f"{group.name}/{ch.name}")

                # 获取采样率属性
                for key in ["sample_rate", "sampling_rate", "fs", "sr"]:
                    if hasattr(ch.properties, key):
                        sample_rate = float(ch.properties[key])
                        break
                if hasattr(ch, "properties"):
                    for key in ["sample_rate", "sampling_rate", "fs", "sr"]:
                        val = ch.properties.get(key)
                        if val is not None:
                            sample_rate = float(val)
                            break

        if not all_data:
            raise ValueError(f"TDMS 文件中未找到通道数据: {file_path}")

        # 对齐到相同长度
        min_len = min(len(d) for d in all_data)
        data = np.array([d[:min_len] for d in all_data])

        return SignalData(
            data=data,
            sample_rate=sample_rate,
            channel_names=channel_names,
            source_path=file_path,
            unit="m/s²",
            metadata={"tdms_channels": len(all_data)},
        )


from ..reader_factory import SignalReaderFactory
SignalReaderFactory.register(TdmsSignalReader)
