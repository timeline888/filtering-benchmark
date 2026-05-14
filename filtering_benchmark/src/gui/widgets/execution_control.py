"""
执行控制组件。

包含运行/停止按钮、进度条、模式选择等控件。
"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)


class ExecutionControlWidget(QWidget):
    """执行控制面板"""
    runClicked = pyqtSignal()
    stopClicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 运行控制
        ctrl_group = QGroupBox("运行控制")
        ctrl_layout = QHBoxLayout(ctrl_group)

        self._run_btn = QPushButton("▶ 运行评估")
        self._run_btn.setMinimumHeight(36)
        self._run_btn.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71; color: white;
                font-weight: bold; border-radius: 4px;
                padding: 6px 20px;
            }
            QPushButton:hover { background-color: #27ae60; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self._run_btn.clicked.connect(self.runClicked.emit)
        ctrl_layout.addWidget(self._run_btn)

        self._stop_btn = QPushButton("■ 停止")
        self._stop_btn.setMinimumHeight(36)
        self._stop_btn.setEnabled(False)
        self._stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c; color: white;
                font-weight: bold; border-radius: 4px;
                padding: 6px 20px;
            }
            QPushButton:hover { background-color: #c0392b; }
            QPushButton:disabled { background-color: #95a5a6; }
        """)
        self._stop_btn.clicked.connect(self.stopClicked.emit)
        ctrl_layout.addWidget(self._stop_btn)

        layout.addWidget(ctrl_group)

        # 进度
        progress_group = QGroupBox("进度")
        progress_layout = QVBoxLayout(progress_group)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        progress_layout.addWidget(self._progress_bar)

        self._status_label = QLabel("就绪")
        self._status_label.setStyleSheet("color: #555;")
        progress_layout.addWidget(self._status_label)

        layout.addWidget(progress_group)

        # 评估设置
        config_group = QGroupBox("评估设置")
        config_layout = QFormLayout(config_group)

        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["全部算法 (all)", "选定算法 (selected)", "按类别 (category)"])
        self._mode_combo.setCurrentIndex(1)
        config_layout.addRow("算法模式:", self._mode_combo)

        self._scoring_combo = QComboBox()
        self._scoring_combo.addItems(["weighted_sum (加权和)", "topsis (TOPSIS)", "borda_count (Borda计数)"])
        config_layout.addRow("评分方法:", self._scoring_combo)

        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(1, 20)
        self._top_n_spin.setValue(3)
        config_layout.addRow("TOP-N:", self._top_n_spin)

        self._backend_combo = QComboBox()
        self._backend_combo.addItems(["sequential (顺序)", "multiprocessing (多进程)"])
        config_layout.addRow("并行后端:", self._backend_combo)

        layout.addWidget(config_group)

        # 资源控制
        throttle_group = QGroupBox("资源控制")
        throttle_layout = QFormLayout(throttle_group)

        self._enable_throttle_cb = QCheckBox("启用资源节制（降低系统卡顿）")
        self._enable_throttle_cb.setChecked(True)
        self._enable_throttle_cb.setToolTip(
            "启用后，在执行高计算量算法时自动降低进程优先级和CPU使用。"
            "这会让算法运行更慢，但系统不会卡顿，您可以同时做其他工作。"
        )
        throttle_layout.addRow(self._enable_throttle_cb)

        self._priority_combo = QComboBox()
        self._priority_combo.addItems(["normal (正常)", "below_normal (低于正常)", "idle (空闲)"])
        self._priority_combo.setCurrentIndex(1)  # 默认低于正常
        self._priority_combo.setToolTip(
            "normal: 正常优先级，可能导致系统卡顿\n"
            "below_normal: 降低优先级，系统保持流畅（推荐）\n"
            "idle: 仅CPU空闲时运行，最友好但最慢"
        )
        throttle_layout.addRow("进程优先级:", self._priority_combo)

        cores_layout = QHBoxLayout()
        self._max_cpu_spin = QSpinBox()
        self._max_cpu_spin.setRange(0, 64)
        self._max_cpu_spin.setValue(0)
        self._max_cpu_spin.setToolTip("0=使用所有核心; N>0=限制使用前N个核心")
        cores_layout.addWidget(self._max_cpu_spin)
        cores_layout.addWidget(QLabel("(0=不限)"))
        throttle_layout.addRow("最大CPU核心:", cores_layout)

        workers_layout = QHBoxLayout()
        self._max_workers_spin = QSpinBox()
        self._max_workers_spin.setRange(1, 16)
        self._max_workers_spin.setValue(2)
        self._max_workers_spin.setToolTip("同时运行的算法数量，越大CPU占用越高")
        workers_layout.addWidget(self._max_workers_spin)
        workers_layout.addWidget(QLabel("个并行"))
        throttle_layout.addRow("最大并行数:", workers_layout)

        self._throttle_note = QLabel(
            "💡 建议: 勾选「启用资源节制」并选择「低于正常」优先级，"
            "可在后台运行高复杂度算法时保持系统流畅。"
        )
        self._throttle_note.setStyleSheet("color: #888; font-size: 11px;")
        self._throttle_note.setWordWrap(True)
        throttle_layout.addRow(self._throttle_note)

        layout.addWidget(throttle_group)
        layout.addStretch()

    def set_running(self, running: bool):
        """切换运行/停止状态"""
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        if running:
            self._status_label.setText("正在运行...")

    def update_progress(self, percent: int, message: str):
        """更新进度"""
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def reset(self):
        """重置到就绪状态"""
        self._progress_bar.setValue(0)
        self._status_label.setText("就绪")
        self._run_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    @property
    def algorithm_mode(self) -> str:
        idx = self._mode_combo.currentIndex()
        return ["all", "selected", "category"][idx]

    @property
    def scoring_method(self) -> str:
        idx = self._scoring_combo.currentIndex()
        return ["weighted_sum", "topsis", "borda_count"][idx]

    @property
    def top_n(self) -> int:
        return self._top_n_spin.value()

    @property
    def backend(self) -> str:
        idx = self._backend_combo.currentIndex()
        return ["sequential", "multiprocessing"][idx]

    # ---- 资源控制属性 ----

    @property
    def enable_throttle(self) -> bool:
        """是否启用资源节制"""
        return self._enable_throttle_cb.isChecked()

    @property
    def process_priority(self) -> str:
        """进程优先级: normal | below_normal | idle"""
        idx = self._priority_combo.currentIndex()
        return ["normal", "below_normal", "idle"][idx]

    @property
    def max_cpu_cores(self) -> int:
        """最大CPU核心数，0=不限"""
        return self._max_cpu_spin.value()

    @property
    def max_workers_setting(self) -> int:
        """最大并行工作线程数"""
        return self._max_workers_spin.value()
