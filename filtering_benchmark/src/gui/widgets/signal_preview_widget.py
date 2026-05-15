"""
信号预览组件 - 集成到右侧面板。

提供两个独立面板：
1. OriginalSignalPanel  — 原始信号的时域/FFT/包络谱
2. FilteredSignalPanel  — 滤波后信号的时域/FFT/包络谱（带算法选择下拉框）

支持中文标签。
支持matplotlib导航工具栏（缩放、平移、重置）。
"""

from typing import Any, Dict, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout,
    QLabel, QVBoxLayout, QWidget,
)

# ── 确保中文字体全局可用 ──
import matplotlib.pyplot as _plt
_plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
_plt.rcParams['axes.unicode_minus'] = False


class SignalPreviewCanvas(FigureCanvasQTAgg):
    """三面板信号画布 - 时域 / FFT / 包络谱"""

    def __init__(self, parent=None, width: int = 8, height: int = 6, dpi: int = 100):
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.figure)
        self.setParent(parent)
        self._setup_axes()

    def _setup_axes(self):
        """创建三个子图，最大化利用空间"""
        self.figure.clear()
        # 使用 gridspec 精确控制布局，减小边距和间距
        gs = self.figure.add_gridspec(3, 1, left=0.08, right=0.98, top=0.94, bottom=0.06, hspace=0.35)
        self._ax_time = self.figure.add_subplot(gs[0])
        self._ax_freq = self.figure.add_subplot(gs[1])
        self._ax_env = self.figure.add_subplot(gs[2])

    def update_plots(self, data: np.ndarray, sample_rate: float,
                     title: str = "信号预览"):
        """更新三个子图"""
        self._ax_time.clear()
        self._ax_freq.clear()
        self._ax_env.clear()

        n = len(data)

        # ── 1. 时域波形 ──
        time_axis = np.arange(n) / sample_rate
        self._ax_time.plot(time_axis, data, linewidth=0.5, color='#2c3e50')
        self._ax_time.set_xlabel("时间 (s)")
        self._ax_time.set_ylabel("幅值")
        self._ax_time.set_title("时域波形")
        self._ax_time.grid(True, alpha=0.3)

        # ── 2. FFT 频谱 ──
        fft_spec = np.abs(np.fft.rfft(data))
        freq_axis = np.fft.rfftfreq(n, 1.0 / sample_rate)
        self._ax_freq.plot(freq_axis, fft_spec, linewidth=0.5, color='#2980b9')
        self._ax_freq.set_xlabel("频率 (Hz)")
        self._ax_freq.set_ylabel("幅值")
        self._ax_freq.set_title("频谱 (FFT)")
        self._ax_freq.grid(True, alpha=0.3)

        # ── 3. 全频带包络谱（去除直流后再FFT） ──
        from scipy import signal
        analytic = signal.hilbert(data)
        envelope = np.abs(analytic)
        envelope_ac = envelope - np.mean(envelope)  # 去除直流分量
        env_spec = np.abs(np.fft.rfft(envelope_ac))
        self._ax_env.plot(freq_axis, env_spec, linewidth=0.5, color='#e67e22')
        self._ax_env.set_xlabel("频率 (Hz)")
        self._ax_env.set_ylabel("幅值")
        self._ax_env.set_title("全频带包络谱")
        self._ax_env.grid(True, alpha=0.3)

        self.figure.suptitle(title, fontsize=10, fontweight='bold', y=0.98)
        self.draw()

    def clear(self):
        """清空所有子图"""
        self._setup_axes()
        self.draw()


class OriginalSignalPanel(QWidget):
    """原始信号谱图展示面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # 信息标签
        self._info_label = QLabel("请加载信号以查看预览")
        self._info_label.setStyleSheet(
            "color: #888; padding: 4px; font-size: 11px;")
        layout.addWidget(self._info_label)

        # 画布（填满剩余空间）
        self._canvas = SignalPreviewCanvas(self)
        self._canvas.setVisible(False)
        layout.addWidget(self._canvas, stretch=1)

        # 工具栏
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.setStyleSheet("QToolBar { border: none; }")
        self._toolbar.setVisible(False)
        layout.addWidget(self._toolbar)

    def update_signal(self, data: np.ndarray, sample_rate: float,
                      source_path: str = ""):
        """更新原始信号谱图"""
        import os
        basename = os.path.basename(source_path) if source_path else ""
        title_parts = [part for part in [basename] if part]
        title = "原始信号"
        if title_parts:
            title = f"原始信号 - {' | '.join(title_parts)}"

        self._canvas.update_plots(data, sample_rate, title)
        self._canvas.setVisible(True)
        self._toolbar.setVisible(True)

        duration = len(data) / sample_rate
        self._info_label.setText(
            f"采样率: {sample_rate:.0f} Hz | "
            f"数据长度: {len(data)} | "
            f"时长: {duration:.2f} s"
        )

    def clear(self):
        """清空"""
        self._canvas.clear()
        self._canvas.setVisible(False)
        self._toolbar.setVisible(False)
        self._info_label.setText("请加载信号以查看预览")


class FilteredSignalPanel(QWidget):
    """滤波后信号谱图展示面板

    通过下拉框切换查看不同算法的滤波结果。
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._denoised_results: Dict[str, Any] = {}
        self._sample_rate: float = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ── 算法选择区 ──
        selector_group = QGroupBox("算法选择")
        selector_layout = QHBoxLayout(selector_group)

        selector_layout.addWidget(QLabel("查看结果:"))
        self._algo_combo = QComboBox()
        self._algo_combo.setMinimumWidth(200)
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        selector_layout.addWidget(self._algo_combo)

        selector_layout.addStretch()
        layout.addWidget(selector_group)

        # ── 信息标签 ──
        self._info_label = QLabel("请先运行评估以查看滤波结果")
        self._info_label.setStyleSheet(
            "color: #888; padding: 4px; font-size: 11px;")
        layout.addWidget(self._info_label)

        # ── 画布（填满剩余空间） ──
        self._canvas = SignalPreviewCanvas(self)
        self._canvas.setVisible(False)
        layout.addWidget(self._canvas, stretch=1)

        # ── 工具栏 ──
        self._toolbar = NavigationToolbar2QT(self._canvas, self)
        self._toolbar.setStyleSheet("QToolBar { border: none; }")
        self._toolbar.setVisible(False)
        layout.addWidget(self._toolbar)

    # ── 公共接口 ──

    def load_results(self, denoised_results: Dict[str, Any],
                     sample_rate: float):
        """加载评估结果并填充算法选择框"""
        self._denoised_results = denoised_results
        self._sample_rate = sample_rate

        # 更新下拉框，阻断信号避免频繁重绘
        self._algo_combo.blockSignals(True)
        self._algo_combo.clear()
        for algo_id, result in denoised_results.items():
            self._algo_combo.addItem(result.algorithm_name, algo_id)
        self._algo_combo.blockSignals(False)

        if self._algo_combo.count() > 0:
            self._algo_combo.setCurrentIndex(0)
            self._show_current_algo()

    def clear(self):
        """清空全部"""
        self._denoised_results = {}
        self._sample_rate = 0
        self._algo_combo.blockSignals(True)
        self._algo_combo.clear()
        self._algo_combo.blockSignals(False)
        self._canvas.clear()
        self._canvas.setVisible(False)
        self._toolbar.setVisible(False)
        self._info_label.setText("请先运行评估以查看滤波结果")

    # ── 内部方法 ──

    def _on_algo_changed(self, index: int):
        self._show_current_algo()

    def _show_current_algo(self):
        """显示当前选中算法的三面板谱图"""
        algo_id = self._algo_combo.currentData()
        if algo_id is None or algo_id not in self._denoised_results:
            return

        result = self._denoised_results[algo_id]
        signal_1d = result.denoised_signal
        if signal_1d.ndim > 1:
            signal_1d = signal_1d[0]

        title = f"滤波结果 - {result.algorithm_name}"
        self._canvas.update_plots(signal_1d, self._sample_rate, title)
        self._canvas.setVisible(True)
        self._toolbar.setVisible(True)

        self._info_label.setText(
            f"算法: {result.algorithm_name} | "
            f"类别: {result.category.value} | "
            f"执行时间: {result.execution_time:.3f}s"
            f"{' | 执行失败' if not result.success else ''}"
        )
