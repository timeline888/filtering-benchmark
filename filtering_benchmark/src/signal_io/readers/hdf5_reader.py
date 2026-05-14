"""
HDF5 格式信号读取器。
"""

from pathlib import Path
from typing import Set

import numpy as np

from ...core.types import SignalData
from ..base_reader import BaseSignalReader


class Hdf5SignalReader(BaseSignalReader):
    """HDF5 信号读取器"""

    supported_extensions: Set[str] = {".h5", ".hdf5"}

    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取 HDF5 文件。
        自动检测包含数据的路径。
        """
        import h5py

        with h5py.File(file_path, 'r') as f:
            candidates = []

            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim <= 2 and obj.size > 0:
                    candidates.append((name, np.array(obj, dtype=np.float64)))
            f.visititems(find_datasets)

            if not candidates:
                raise ValueError(f"HDF5 文件中未找到数据集: {file_path}")

            # 取值最大的数据集
            name, data = max(candidates, key=lambda x: x[1].size)

            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T

            # 读取采样率属性
            ds = f[name]
            sample_rate = 1.0
            for key in ["sample_rate", "fs", "sr"]:
                val = ds.attrs.get(key)
                if val is not None:
                    sample_rate = float(val)
                    break

            n_chan = data.shape[0]
            channel_names = [f"{name}_ch{i}" for i in range(n_chan)]

        return SignalData(
            data=data,
            sample_rate=sample_rate,
            channel_names=channel_names,
            source_path=file_path,
            unit="m/s²",
            metadata={"h5_path": name},
        )


from ..reader_factory import SignalReaderFactory
SignalReaderFactory.register(Hdf5SignalReader)
