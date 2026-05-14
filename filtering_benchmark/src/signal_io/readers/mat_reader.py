"""
MAT (MATLAB) 格式信号读取器。
"""

from pathlib import Path
from typing import Set

import numpy as np

from ...core.types import SignalData
from ..base_reader import BaseSignalReader


class MatSignalReader(BaseSignalReader):
    """MAT 信号读取器"""

    supported_extensions: Set[str] = {".mat"}

    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取 MATLAB .mat 文件。
        自动检测信号变量名（取第一个最大2D数组）。
        """
        import scipy.io as sio
        import h5py

        # 尝试 v5/v7 MAT 格式
        try:
            mat_data = sio.loadmat(file_path, squeeze_me=False)
        except NotImplementedError:
            # v7.3 格式（HDF5）
            return self._read_h5(file_path, channel)
        except Exception:
            return self._read_h5(file_path, channel)

        # 过滤掉 MATLAB 内部变量
        meta_vars = {'__header__', '__version__', '__globals__'}
        candidates = {k: v for k, v in mat_data.items()
                      if k not in meta_vars and isinstance(v, np.ndarray)}

        if not candidates:
            raise ValueError(f"未找到数值变量: {file_path}")

        # 选择最大的非标量数组
        best_var = max(candidates, key=lambda k: candidates[k].size)
        data = candidates[best_var].astype(np.float64).squeeze()

        # 确保 shape 为 (n_channels, n_samples)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim == 2 and data.shape[0] > data.shape[1]:
            data = data.T

        sample_rate = self._infer_sample_rate(mat_data, file_path)

        return SignalData(
            data=data,
            sample_rate=sample_rate,
            source_path=file_path,
            unit="m/s²",
            metadata={"mat_variable": best_var},
        )

    def _read_h5(self, file_path: str, channel: int) -> SignalData:
        """读取 HDF5 格式 .mat (v7.3)"""
        import h5py

        with h5py.File(file_path, 'r') as f:
            # 查找第一个包含数据集的组
            candidates = []
            def find_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.ndim <= 2:
                    candidates.append((name, obj))
            f.visititems(find_datasets)

            if not candidates:
                raise ValueError(f"未找到数据集: {file_path}")

            # 选择最大的数据集
            name, ds = max(candidates, key=lambda x: x[1].size)
            data = ds[()].astype(np.float64)

            if data.ndim == 1:
                data = data.reshape(1, -1)
            elif data.ndim == 2 and data.shape[0] > data.shape[1]:
                data = data.T

        return SignalData(
            data=data,
            sample_rate=1.0,
            source_path=file_path,
            unit="m/s²",
            metadata={"h5_dataset": name},
        )


from ..reader_factory import SignalReaderFactory
SignalReaderFactory.register(MatSignalReader)
