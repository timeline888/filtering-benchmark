"""
异步工作线程 - 在 QThread 中执行 PipelineOrchestrator。

使用 QObject + moveToThread 模式，通过信号与主线程通信。
"""

from typing import Any, Dict, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class PipelineWorker(QObject):
    """管线执行工作器"""
    progressUpdated = pyqtSignal(int, str)  # percent, message
    stageChanged = pyqtSignal(str)          # stage name
    finished = pyqtSignal(object)           # EvaluationReport
    error = pyqtSignal(str)                 # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._orchestrator = None
        self._config = None

    def set_config(self, config: Any):
        """设置要执行的配置"""
        self._config = config

    def run(self):
        """执行管线（在 QThread 中调用）"""
        try:
            from ...pipeline.orchestrator import PipelineOrchestrator

            def progress_cb(percent: int, message: str):
                self.progressUpdated.emit(percent, message)

            self._orchestrator = PipelineOrchestrator(
                self._config,
                progress_callback=progress_cb
            )

            self.stageChanged.emit("开始运行评估流水线")
            report = self._orchestrator.run()
            self.finished.emit(report)

        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            self.error.emit(str(e))

    def abort(self):
        """请求中止"""
        if self._orchestrator:
            self._orchestrator.abort()
