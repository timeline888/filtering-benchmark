"""
信号读取器注册模块。

导入所有读取器模块以触发 SignalReaderFactory 注册。
"""

# 导入所有读取器，触发底部的 factory.register() 调用
from . import mat_reader       # .mat
from . import csv_reader       # .csv, .txt
from . import wav_reader       # .wav, .flac, .mp3
from . import hdf5_reader      # .h5, .hdf5
from . import tdms_reader      # .tdms
from . import uff_reader       # .uff, .unv
