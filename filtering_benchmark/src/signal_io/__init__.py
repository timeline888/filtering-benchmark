"""
信号输入模块。

支持多种格式的读取，输出统一 SignalData 对象。
"""

# 导入 readers 子包（触发所有读取器向工厂注册）
from . import readers  # noqa: F401

from .base_reader import BaseSignalReader
from .reader_factory import SignalReaderFactory
from .preprocessor import SignalPreprocessor

__all__ = [
    "BaseSignalReader",
    "SignalReaderFactory",
    "SignalPreprocessor",
]
