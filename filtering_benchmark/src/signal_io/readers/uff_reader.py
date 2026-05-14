"""
UFF (Universal File Format) 格式信号读取器。
"""

from pathlib import Path
from typing import Set

import numpy as np

from ...core.types import SignalData
from ..base_reader import BaseSignalReader


class UffSignalReader(BaseSignalReader):
    """UFF 信号读取器"""

    supported_extensions: Set[str] = {".uff", ".unv"}

    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取 UFF 58/58b/15 格式文件。
        """
        import pyuff

        uff = pyuff.UFF(file_path)
        data_sets = uff.read_sets()

        if not data_sets:
            raise ValueError(f"UFF 文件中未找到数据: {file_path}")

        all_data = []
        metadata = {}

        for ds in data_sets:
            if hasattr(ds, 'data'):
                d = np.array(ds.data, dtype=np.float64)
                if d.ndim == 1:
                    d = d.reshape(1, -1)
                all_data.append(d)
            elif isinstance(ds, dict) and 'data' in ds:
                d = np.array(ds['data'], dtype=np.float64)
                if d.ndim == 1:
                    d = d.reshape(1, -1)
                all_data.append(d)

        if not all_data:
            raise ValueError(f"UFF 文件中未找到数值数据: {file_path}")

        data = np.vstack(all_data)

        # 尝试推断采样率
        sample_rate = 1.0
        for ds in data_sets:
            props = ds if isinstance(ds, dict) else ds.__dict__
            for key in ['sample_rate', 'fs', 'abscissa_spacing', 'delta_x']:
                val = props.get(key)
                if val is not None and float(val) > 0:
                    sample_rate = 1.0 / float(val)
                    break

        n_chan = data.shape[0]
        channel_names = [f"uff_{i}" for i in range(n_chan)]

        return SignalData(
            data=data,
            sample_rate=sample_rate,
            channel_names=channel_names,
            source_path=file_path,
            unit="m/s²",
            metadata={"n_datasets": len(data_sets)},
        )


from ..reader_factory import SignalReaderFactory
SignalReaderFactory.register(UffSignalReader)
