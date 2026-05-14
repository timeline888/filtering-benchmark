"""
读取器工厂模式实现。

根据文件扩展名自动选择对应的 Reader。
"""

from pathlib import Path
from typing import Dict, Optional, Type

from loguru import logger

from ..core.exceptions import UnsupportedFormatError
from .base_reader import BaseSignalReader


class SignalReaderFactory:
    """信号读取器工厂"""

    _readers: Dict[str, Type[BaseSignalReader]] = {}

    @classmethod
    def register(cls, reader_cls: Type[BaseSignalReader]) -> None:
        """注册读取器"""
        for ext in reader_cls.supported_extensions:
            cls._readers[ext] = reader_cls
        logger.debug(f"注册读取器: {reader_cls.__name__} -> {reader_cls.supported_extensions}")

    @classmethod
    def create_reader(cls, file_path: str) -> BaseSignalReader:
        """
        根据文件扩展名创建对应的读取器。

        Args:
            file_path: 文件路径

        Returns:
            读取器实例

        Raises:
            UnsupportedFormatError: 不支持的格式
        """
        ext = Path(file_path).suffix.lower()
        if ext not in cls._readers:
            raise UnsupportedFormatError(ext)

        reader_cls = cls._readers[ext]
        return reader_cls()

    @classmethod
    def list_supported_formats(cls) -> list:
        """列出所有支持的格式"""
        return sorted(cls._readers.keys())

    @classmethod
    def read(cls, file_path: str, channel: int = 0, **kwargs):
        """便捷方法：直接读取信号"""
        reader = cls.create_reader(file_path)
        return reader.read(file_path, channel=channel, **kwargs)
