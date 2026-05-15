"""
信号输入组件。

支持文件模式 (MAT/WAV/CSV/TXT/HDF5) 和仿真模式。
嵌入 matplotlib 波形预览。
"""

import os
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup, QComboBox, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QRadioButton, QSizePolicy, QSpinBox, QStackedWidget,
    QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class SignalPreviewCanvas(FigureCanvasQTAgg):
    """信号预览画布"""
    def __init__(self, parent=None):
        # 确保中文字体可渲染
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
        plt.rcParams['axes.unicode_minus'] = False

        self.figure = Figure(figsize=(5, 2), dpi=80)
        self.axes = self.figure.add_subplot(111)
        super().__init__(self.figure)
        self.setParent(parent)
        self.figure.tight_layout(pad=1.0)

    def plot_signal(self, data: np.ndarray, sample_rate: float, title: str = "信号预览"):
        self.axes.clear()
        time_axis = np.arange(len(data)) / sample_rate
        self.axes.plot(time_axis, data, linewidth=0.5)
        self.axes.set_xlabel("时间 (s)")
        self.axes.set_ylabel("幅值")
        self.axes.set_title(title)
        self.figure.tight_layout(pad=1.0)
        self.draw()


class SignalInputWidget(QWidget):
    """信号输入面板"""
    signalLoaded = pyqtSignal(object, float, str)  # data, sample_rate, source_path

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_data: Optional[np.ndarray] = None
        self._current_fs: float = 25600.0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 信号源选择
        source_group = QGroupBox("信号源")
        source_layout = QVBoxLayout(source_group)

        self._mode_stack = QStackedWidget()

        # ---- 文件模式 ----
        file_widget = QWidget()
        file_layout = QVBoxLayout(file_widget)
        file_layout.setContentsMargins(0, 0, 0, 0)

        file_row = QHBoxLayout()
        self._file_path_edit = QLineEdit()
        self._file_path_edit.setPlaceholderText("选择信号文件...")
        self._file_path_edit.setReadOnly(True)
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self._file_path_edit)
        file_row.addWidget(browse_btn)
        file_layout.addLayout(file_row)

        self._file_info_label = QLabel("未选择文件")
        self._file_info_label.setStyleSheet("color: #888;")
        file_layout.addWidget(self._file_info_label)

        self._mode_stack.addWidget(file_widget)

        # ---- 仿真模式 ----
        sim_widget = QWidget()
        sim_layout = QFormLayout(sim_widget)
        sim_layout.setContentsMargins(0, 0, 0, 0)

        self._sim_type_combo = QComboBox()
        self._sim_type_combo.addItems([
            "bearing_fault (轴承故障)",
            "gear_fault (齿轮故障)",
            "chirp (线性调频)",
            "multi_harmonic (多谐波)",
        ])
        sim_layout.addRow("仿真类型:", self._sim_type_combo)

        self._sim_fs_spin = QSpinBox()
        self._sim_fs_spin.setRange(1000, 256000)
        self._sim_fs_spin.setValue(25600)
        self._sim_fs_spin.setSingleStep(1000)
        self._sim_fs_spin.setSuffix(" Hz")
        sim_layout.addRow("采样率:", self._sim_fs_spin)

        self._sim_duration_spin = QSpinBox()
        self._sim_duration_spin.setRange(1, 30)
        self._sim_duration_spin.setValue(1)
        self._sim_duration_spin.setSuffix(" s")
        sim_layout.addRow("持续时间:", self._sim_duration_spin)

        self._sim_noise_spin = QSpinBox()
        self._sim_noise_spin.setRange(0, 100)
        self._sim_noise_spin.setValue(30)
        self._sim_noise_spin.setSuffix(" %")
        sim_layout.addRow("噪声水平:", self._sim_noise_spin)

        generate_btn = QPushButton("生成仿真信号")
        generate_btn.clicked.connect(self._generate_simulation)
        sim_layout.addRow("", generate_btn)

        self._mode_stack.addWidget(sim_widget)

        source_layout.addWidget(self._mode_stack)
        layout.addWidget(source_group)

        # 模式切换
        mode_group = QGroupBox("输入模式")
        mode_layout = QHBoxLayout(mode_group)
        self._file_radio = QRadioButton("文件")
        self._sim_radio = QRadioButton("仿真")
        self._file_radio.setChecked(True)
        self._file_radio.toggled.connect(self._on_mode_changed)
        mode_layout.addWidget(self._file_radio)
        mode_layout.addWidget(self._sim_radio)
        mode_layout.addStretch()
        layout.addWidget(mode_group)

        # 参数设置
        param_group = QGroupBox("参数设置")
        param_layout = QFormLayout(param_group)

        self._channel_spin = QSpinBox()
        self._channel_spin.setRange(0, 63)
        self._channel_spin.setValue(0)
        param_layout.addRow("通道:", self._channel_spin)

        layout.addWidget(param_group)

        # 信号预览（已隐藏，保留代码结构以便后续恢复）
        preview_group = QGroupBox("信号预览")
        preview_layout = QVBoxLayout(preview_group)
        self._preview_canvas = SignalPreviewCanvas()
        preview_layout.addWidget(self._preview_canvas)
        preview_group.setVisible(False)  # 隐藏预览区域
        layout.addWidget(preview_group)

    def _on_mode_changed(self, checked: bool):
        if self._file_radio.isChecked():
            self._mode_stack.setCurrentIndex(0)
        else:
            self._mode_stack.setCurrentIndex(1)

    def _browse_file(self):
        from PyQt6.QtWidgets import QFileDialog
        default_dir = r"E:\Qoder项目\滤波基石系统设计\cwru"
        path, _ = QFileDialog.getOpenFileName(
            self, "选择信号文件", default_dir,
            "信号文件 (*.mat *.wav *.csv *.txt *.h5 *.hdf5);;MATLAB (*.mat);;WAV (*.wav);;CSV (*.csv);;TXT (*.txt);;HDF5 (*.h5 *.hdf5);;所有文件 (*.*)"
        )
        if path:
            self._file_path_edit.setText(path)
            self._load_file(path)

    def _get_format_label(self, path: str) -> str:
        """根据文件扩展名返回格式标签"""
        ext = os.path.splitext(path)[1].lower()
        fmt_map = {
            '.mat': 'MATLAB',
            '.wav': 'WAV',
            '.csv': 'CSV',
            '.txt': 'TXT',
            '.h5': 'HDF5',
            '.hdf5': 'HDF5',
        }
        return fmt_map.get(ext, ext.upper().lstrip('.'))

    def _load_file(self, path: str):
        try:
            from ...signal_io import SignalReaderFactory
            reader = SignalReaderFactory.create_reader(path)
            channel = self._channel_spin.value()
            signal_data = reader.read(path, channel=channel)

            # 对于采样率为 1.0 的 .mat 文件，使用 GUI 默认采样率
            fs = signal_data.sample_rate
            if fs <= 1.0 and path.lower().endswith('.mat'):
                fs = float(self._sim_fs_spin.value())
            self._current_fs = fs
            self._current_data = signal_data.get_channel(0)

            fmt = self._get_format_label(path)
            shape_str = f"{signal_data.data.shape}"
            fs_str = f"{fs:.0f} Hz"
            dur_str = f"{signal_data.duration:.2f} s"
            self._file_info_label.setText(f"[{fmt}] {shape_str}, {fs_str}, {dur_str}")
            self._file_info_label.setStyleSheet("color: #888;")

            self._preview_canvas.plot_signal(self._current_data, fs, f"[{fmt}] {os.path.basename(path)}")
            self.signalLoaded.emit(self._current_data, fs, path)
        except Exception as e:
            self._file_info_label.setText(f"加载失败: {e}")
            self._file_info_label.setStyleSheet("color: red;")

    def _generate_simulation(self):
        try:
            from ...signal_simulation.simulator import SignalSimulator
            sim = SignalSimulator()
            fs = float(self._sim_fs_spin.value())
            duration = float(self._sim_duration_spin.value())
            noise_level = self._sim_noise_spin.value() / 100.0

            sim_type = self._sim_type_combo.currentIndex()
            if sim_type == 0:
                data = sim.bearing_fault(fs, duration, fault_freq=78.5, noise_level=noise_level)
            elif sim_type == 1:
                data = sim.gear_fault(fs, duration, mesh_freq=500, noise_level=noise_level)
            elif sim_type == 2:
                data = sim.chirp(fs, duration, f0=100, f1=2000, noise_level=noise_level)
            else:
                data = sim.multi_harmonic(fs, duration, base_freq=50, harmonics=5, noise_level=noise_level)

            self._current_data = data.flatten()
            self._current_fs = fs

            self._preview_canvas.plot_signal(self._current_data, fs, f"仿真: {self._sim_type_combo.currentText()}")
            self.signalLoaded.emit(self._current_data, fs, "simulated")
        except Exception as e:
            self._file_info_label.setText(f"仿真失败: {e}")
            self._file_info_label.setStyleSheet("color: red;")

    @property
    def signal_data(self) -> Optional[np.ndarray]:
        return self._current_data

    @property
    def sample_rate(self) -> float:
        return self._current_fs

    @property
    def source_path(self) -> str:
        if self._file_radio.isChecked():
            return self._file_path_edit.text() or "simulated"
        return "simulated"
