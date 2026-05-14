"""
信号读取器基类。

每种格式对应一个 Reader 子类，统一返回 SignalData 对象。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from ..core.types import SignalData
from ..core.constants import SignalFormat


class BaseSignalReader(ABC):
    """信号读取器抽象基类"""

    # 支持的扩展名集合
    supported_extensions: Set[str] = set()

    @abstractmethod
    def read(self, file_path: str, channel: int = 0, **kwargs) -> SignalData:
        """
        读取信号文件。

        Args:
            file_path: 文件路径
            channel: 要读取的通道索引
            **kwargs: 格式相关参数

        Returns:
            SignalData 对象
        """
        pass

    @classmethod
    def supports(cls, file_path: str) -> bool:
        """检查是否支持该文件格式"""
        ext = Path(file_path).suffix.lower()
        return ext in cls.supported_extensions

    def _infer_sample_rate(self, data: dict, file_path: str) -> float:
        """尝试从元数据中推断采样率（子类可重写）"""
        for key in ["fs", "sample_rate", "sampling_rate", "srate", "sr", "rate"]:
            val = data.get(key, None)
            if val is not None:
                return float(val)
        return 1.0  # 默认值
