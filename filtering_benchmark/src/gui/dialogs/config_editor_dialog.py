"""
配置编辑器对话框 - YAML 配置查看与编辑。
"""

from typing import Any, Dict

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout,
)


class ConfigEditorDialog(QDialog):
    """YAML 配置编辑器对话框"""

    def __init__(self, config_dict: Dict[str, Any] = None, parent=None):
        super().__init__(parent)
        self._config_dict = config_dict or {}
        self.setWindowTitle("配置编辑器")
        self.setMinimumSize(700, 500)
        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # 工具栏
        toolbar = QHBoxLayout()
        load_btn = QPushButton("加载 YAML...")
        load_btn.clicked.connect(self._load_yaml)
        toolbar.addWidget(load_btn)

        save_btn = QPushButton("保存到 YAML...")
        save_btn.clicked.connect(self._save_yaml)
        toolbar.addWidget(save_btn)

        toolbar.addStretch()
        layout.addLayout(toolbar)

        # 编辑器
        self._editor = QTextEdit()
        self._editor.setFont(
            self._editor.font().family() if hasattr(self._editor.font(), 'family') else None
        )
        self._editor.setStyleSheet("font-family: 'Consolas', 'Courier New', monospace; font-size: 12px;")
        layout.addWidget(self._editor)

        # 按钮
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_config(self):
        """加载配置到编辑器"""
        import yaml
        try:
            text = yaml.dump(self._config_dict, default_flow_style=False, allow_unicode=True, sort_keys=False)
            self._editor.setText(text)
        except Exception as e:
            self._editor.setText(f"# 配置加载失败: {e}")

    def _load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "加载 YAML 配置", "", "YAML (*.yaml *.yml);;所有文件 (*.*)"
        )
        if path:
            try:
                import yaml
                with open(path, 'r', encoding='utf-8') as f:
                    self._config_dict = yaml.safe_load(f)
                self._load_config()
            except Exception as e:
                QMessageBox.warning(self, "加载失败", f"无法加载配置: {e}")

    def _save_yaml(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "保存 YAML 配置", "config.yaml", "YAML (*.yaml *.yml)"
        )
        if path:
            try:
                import yaml
                config = self.get_config_dict()
                with open(path, 'w', encoding='utf-8') as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
                QMessageBox.information(self, "保存成功", f"配置已保存到: {path}")
            except Exception as e:
                QMessageBox.warning(self, "保存失败", f"无法保存配置: {e}")

    def get_config_dict(self) -> Dict[str, Any]:
        """从编辑器获取配置字典"""
        import yaml
        try:
            return yaml.safe_load(self._editor.toPlainText()) or {}
        except Exception:
            return self._config_dict
