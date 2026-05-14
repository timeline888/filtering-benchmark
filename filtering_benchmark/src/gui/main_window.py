"""
主窗口 - 滤波基石系统 GUI 主界面。

组装所有子组件，管理信号/槽连接和工作线程。
"""

from typing import Optional

from PyQt6.QtCore import QThread, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QGroupBox, QMainWindow,
    QMessageBox, QScrollArea, QSizePolicy, QSplitter, QStatusBar,
    QVBoxLayout, QWidget,
)

from ..algorithms.registry import registry as algo_registry
from ..core.types import EvaluationReport
from .dialogs.config_editor_dialog import ConfigEditorDialog
from .utils.config_bridge import ConfigBridge
from .widgets.algorithm_selector import AlgorithmSelectorWidget
from .widgets.execution_control import ExecutionControlWidget
from .widgets.results_tab_widget import ResultsTabWidget
from .widgets.signal_input_widget import SignalInputWidget
from .workers.pipeline_worker import PipelineWorker


class MainWindow(QMainWindow):
    """滤波基石系统主窗口"""

    def __init__(self):
        super().__init__()
        self._worker: Optional[PipelineWorker] = None
        self._thread: Optional[QThread] = None
        self._setup_ui()

    def _setup_ui(self):
        self.setWindowTitle("滤波基石系统 - 滤波降噪算法评估平台")
        self.setMinimumSize(1200, 800)

        # 菜单栏
        self._setup_menu()

        # 中央组件
        central = QWidget()
        self.setCentralWidget(central)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ---- 左侧面板 ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(380)
        left_scroll.setMaximumWidth(500)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)

        # 信号输入
        self._signal_widget = SignalInputWidget()
        left_layout.addWidget(self._signal_widget)

        # 算法选择
        self._algo_selector = AlgorithmSelectorWidget()
        algo_group = QGroupBox("算法选择")
        algo_layout = QVBoxLayout(algo_group)
        algo_layout.addWidget(self._algo_selector)
        left_layout.addWidget(algo_group)

        # 执行控制
        self._exec_ctrl = ExecutionControlWidget()
        left_layout.addWidget(self._exec_ctrl)

        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)

        # ---- 右侧面板 ----
        self._results_widget = ResultsTabWidget()
        splitter.addWidget(self._results_widget)

        splitter.setStretchFactor(0, 0)  # 左侧不伸缩
        splitter.setStretchFactor(1, 1)  # 右侧伸缩

        layout = QVBoxLayout(central)
        layout.addWidget(splitter)

        # 状态栏
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("就绪")

        # 信号连接
        self._exec_ctrl.runClicked.connect(self._on_run)
        self._exec_ctrl.stopClicked.connect(self._on_stop)
        self._signal_widget.signalLoaded.connect(self._on_signal_loaded)

    def _setup_menu(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        load_action = QAction("加载配置...", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self._load_config)
        file_menu.addAction(load_action)

        save_action = QAction("保存配置...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        edit_config_action = QAction("编辑配置...", self)
        edit_config_action.triggered.connect(self._edit_config)
        file_menu.addAction(edit_config_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图")

        expand_action = QAction("展开所有算法", self)
        expand_action.triggered.connect(self._expand_algorithms)
        view_menu.addAction(expand_action)

        collapse_action = QAction("折叠所有算法", self)
        collapse_action.triggered.connect(self._collapse_algorithms)
        view_menu.addAction(collapse_action)

        # 帮助菜单
        help_menu = menubar.addMenu("帮助")

        about_action = QAction("关于", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _on_signal_loaded(self, data, sample_rate, source_path):
        """信号加载完成"""
        self._status_bar.showMessage(f"信号已加载: {source_path}, {sample_rate:.0f} Hz")
        self._results_widget.update_signal_preview(data, sample_rate, source_path)

    def _on_run(self):
        """运行评估"""
        # 验证信号
        if self._signal_widget.signal_data is None:
            QMessageBox.warning(self, "警告", "请先加载或生成信号！")
            return

        # 获取选中算法
        selected_ids = self._algo_selector.get_selected_ids()
        mode = self._exec_ctrl.algorithm_mode
        if mode == "selected" and not selected_ids:
            QMessageBox.warning(self, "警告", "请先选择至少一个算法！")
            return

        # 复杂度警告检查
        if mode == "selected" and selected_ids:
            warnings = algo_registry.get_selected_complexity_warnings(list(selected_ids))
            if warnings:
                msg = QMessageBox(self)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("计算资源消耗警告")
                msg.setText(
                    "⚠️ 您选择的算法中包含以下高计算复杂度算法，"
                    "可能导致系统运行卡顿："
                )
                details = "\n\n".join(warnings)
                msg.setInformativeText(details)
                tips_lines = [
                    "💡 建议：",
                    "• 减少高复杂度算法的选择数量",
                    "• 先试运行低复杂度算法评估效果",
                    "• 查看复杂度分析详情了解具体优化方案",
                ]
                msg.setDetailedText("\n".join(tips_lines))
                yes_btn = msg.addButton("仍然继续运行", QMessageBox.ButtonRole.YesRole)
                msg.addButton("取消运行", QMessageBox.ButtonRole.RejectRole)
                msg.setDefaultButton(msg.buttons()[1])  # 默认选中「取消」
                msg.exec()
                if msg.clickedButton() == yes_btn:
                    self._status_bar.showMessage("用户确认高风险继续运行...")
                else:
                    self._status_bar.showMessage("已取消运行")
                    return

        # 生成配置
        config = ConfigBridge.ui_to_appconfig(
            signal_data=self._signal_widget.signal_data,
            sample_rate=self._signal_widget.sample_rate,
            source_path=self._signal_widget.source_path,
            algorithm_selected_ids=selected_ids if mode == "selected" else [],
            algorithm_mode=mode,
            scoring_method=self._exec_ctrl.scoring_method,
            top_n=self._exec_ctrl.top_n,
            backend=self._exec_ctrl.backend,
            channel=0,
            # 资源控制
            enable_throttle=self._exec_ctrl.enable_throttle,
            process_priority=self._exec_ctrl.process_priority,
            max_cpu_cores=self._exec_ctrl.max_cpu_cores,
            max_workers=self._exec_ctrl.max_workers_setting,
        )

        # 更新状态
        self._exec_ctrl.set_running(True)
        self._status_bar.showMessage("正在运行...")
        self._results_widget.clear()

        # 创建工作线程
        self._thread = QThread(self)
        self._worker = PipelineWorker()
        self._worker.moveToThread(self._thread)

        # 连接信号
        self._worker.progressUpdated.connect(self._exec_ctrl.update_progress)
        self._worker.stageChanged.connect(lambda s: self._status_bar.showMessage(s))
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        # 线程生命周期
        self._thread.started.connect(self._worker.run)
        self._thread.finished.connect(self._cleanup_thread)

        # 设置配置并启动
        self._worker.set_config(config)
        self._thread.start()

    def _on_stop(self):
        """停止运行"""
        if self._worker:
            self._worker.abort()
            # 立即反馈：禁用停止按钮，防止重复点击
            self._exec_ctrl._stop_btn.setEnabled(False)
            self._exec_ctrl._stop_btn.setText("⏳ 正在中止...")
            self._exec_ctrl._stop_btn.setStyleSheet("""
                QPushButton {
                    background-color: #95a5a6; color: white;
                    font-weight: bold; border-radius: 4px;
                    padding: 6px 20px;
                }
            """)
        self._status_bar.showMessage("正在中止...")

    def _on_finished(self, report: EvaluationReport):
        """运行完成"""
        self._results_widget.load_report(report)

        # 更新滤波结果标签页
        if report.top_denoised_results:
            self._results_widget.update_filtered_signal(
                denoised_results=report.top_denoised_results,
                sample_rate=self._signal_widget.sample_rate,
            )

        self._exec_ctrl.reset()
        self._status_bar.showMessage(f"评估完成 - TOP-1: {report.top_algorithms[0].algorithm_name}" if report.top_algorithms else "评估完成")
        self._cleanup_thread()

    def _on_error(self, error_msg: str):
        """运行出错"""
        self._exec_ctrl.reset()
        # 用户主动中止不弹错误框
        if "用户中止" in error_msg or "流水线被用户中止" in error_msg:
            self._status_bar.showMessage("已中止")
        else:
            self._status_bar.showMessage("运行失败")
            QMessageBox.critical(self, "运行错误", f"评估执行失败:\n{error_msg}")
        self._cleanup_thread()

    def _cleanup_thread(self):
        """清理线程资源"""
        if self._thread:
            if self._thread.isRunning():
                self._thread.quit()
                self._thread.wait(1000)
            self._thread.deleteLater()
            self._thread = None
        if self._worker:
            self._worker.deleteLater()
            self._worker = None

    def _load_config(self):
        """加载 YAML 配置"""
        path, _ = QFileDialog.getOpenFileName(
            self, "加载配置", "", "YAML (*.yaml *.yml);;所有文件 (*.*)"
        )
        if path:
            try:
                config_dict = ConfigBridge.load_yaml_to_dict(path)
                QMessageBox.information(self, "加载成功",
                    f"配置已加载: {path}\n可在\"编辑配置\"中查看详情。")
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载配置: {e}")

    def _save_config(self):
        """保存当前配置到 YAML"""
        path, _ = QFileDialog.getSaveFileName(
            self, "保存配置", "config.yaml", "YAML (*.yaml *.yml)"
        )
        if path:
            config_dict = ConfigBridge.ui_to_config(
                signal_data=self._signal_widget.signal_data,
                sample_rate=self._signal_widget.sample_rate,
                source_path=self._signal_widget.source_path,
                algorithm_selected_ids=list(self._algo_selector.get_selected_ids()),
                algorithm_mode=self._exec_ctrl.algorithm_mode,
                scoring_method=self._exec_ctrl.scoring_method,
                top_n=self._exec_ctrl.top_n,
                backend=self._exec_ctrl.backend,
            )
            try:
                ConfigBridge.save_dict_to_yaml(config_dict, path)
                QMessageBox.information(self, "保存成功", f"配置已保存到: {path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"无法保存配置: {e}")

    def _edit_config(self):
        """编辑配置"""
        config_dict = ConfigBridge.ui_to_config(
            signal_data=self._signal_widget.signal_data,
            sample_rate=self._signal_widget.sample_rate,
            source_path=self._signal_widget.source_path,
            algorithm_selected_ids=list(self._algo_selector.get_selected_ids()),
            algorithm_mode=self._exec_ctrl.algorithm_mode,
            scoring_method=self._exec_ctrl.scoring_method,
            top_n=self._exec_ctrl.top_n,
            backend=self._exec_ctrl.backend,
        )
        dialog = ConfigEditorDialog(config_dict, self)
        if dialog.exec() == ConfigEditorDialog.DialogCode.Accepted:
            self._status_bar.showMessage("配置已更新")

    def _expand_algorithms(self):
        """展开算法树"""
        self._algo_selector.expand_all()

    def _collapse_algorithms(self):
        """折叠算法树"""
        self._algo_selector.collapse_all()

    def _show_about(self):
        QMessageBox.about(self, "关于 滤波基石系统",
            "<h3>滤波基石系统 v1.0</h3>"
            "<p>旋转机械振动/声学信号滤波降噪评估平台</p>"
            "<p>包含 71 种滤波降噪算法、20 项评估指标</p>"
            "<hr>"
            "<p><i>PyQt6 图形用户界面</i></p>"
        )

    def closeEvent(self, event):
        """关闭窗口时清理线程"""
        self._cleanup_thread()
        event.accept()
