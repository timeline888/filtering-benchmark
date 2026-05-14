"""
应用入口 - 滤波基石系统 GUI。

启动 PyQt6 应用程序，初始化主窗口。
"""

import sys
import os

# 确保项目根目录在 path 中
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    """GUI 入口函数"""
    # 延迟导入，确保路径已设置
    from PyQt6.QtGui import QFont
    from PyQt6.QtWidgets import QApplication
    from src.gui.main_window import MainWindow

    # 配置 matplotlib 中文字体（必须在任何绘图前设置）
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False

    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei UI", 9)
    app.setFont(font)

    # 全局样式
    app.setStyleSheet("""
        QGroupBox {
            font-weight: bold;
            border: 1px solid #ccc;
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 12px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 4px;
        }
        QToolTip {
            background-color: #2c3e50;
            color: white;
            border: none;
            padding: 4px;
        }
        QTableView, QTableWidget {
            gridline-color: #ddd;
            selection-background-color: #3498db;
        }
        QHeaderView::section {
            background-color: #2c3e50;
            color: white;
            padding: 4px;
            border: none;
        }
        QTreeView {
            alternate-background-color: #f5f5f5;
        }
    """)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
