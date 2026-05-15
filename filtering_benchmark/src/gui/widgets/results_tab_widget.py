"""
结果展示组件 - 多标签页结果查看器。

包含排名表、指标对比、图表可视化、报告摘要。
支持导出 CSV/HTML。
"""

from typing import Dict, List, Optional

import numpy as np
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTabWidget, QTableWidget, QTableWidgetItem,
    QTableView, QTextBrowser, QVBoxLayout, QWidget,
)

from ...core.types import AlgorithmRanking, EvaluationReport
from .chart_canvas import ChartCanvas
from .signal_preview_widget import OriginalSignalPanel, FilteredSignalPanel


class RankingTableModel(QAbstractTableModel):
    """排名表模型"""

    def __init__(self, rankings: List[AlgorithmRanking] = None, parent=None):
        super().__init__(parent)
        self._rankings = rankings or []
        self._headers = ["排名", "算法名称", "类别", "综合得分", "执行时间(s)"]

    def set_rankings(self, rankings: List[AlgorithmRanking]):
        self.beginResetModel()
        self._rankings = rankings
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._rankings)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._headers)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._rankings):
            return None
        rank = self._rankings[index.row()]
        col = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0:
                return rank.overall_rank
            elif col == 1:
                return rank.algorithm_name
            elif col == 2:
                return rank.category.value if hasattr(rank.category, 'value') else str(rank.category)
            elif col == 3:
                return f"{rank.overall_score:.4f}"
            elif col == 4:
                return f"{rank.execution_time:.3f}"
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return self._headers[section] if section < len(self._headers) else ""
        return None


class ResultsTabWidget(QWidget):
    """结果展示多标签页"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._report: Optional[EvaluationReport] = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 导出按钮
        export_layout = QHBoxLayout()
        export_layout.addStretch()
        self._export_csv_btn = QPushButton("导出 CSV")
        self._export_csv_btn.clicked.connect(self._export_csv)
        self._export_csv_btn.setEnabled(False)
        export_layout.addWidget(self._export_csv_btn)

        self._export_html_btn = QPushButton("导出 HTML")
        self._export_html_btn.clicked.connect(self._export_html)
        self._export_html_btn.setEnabled(False)
        export_layout.addWidget(self._export_html_btn)
        layout.addLayout(export_layout)

        # 标签页
        self._tabs = QTabWidget()

        # 原始信号谱图
        self._original_panel = OriginalSignalPanel()
        self._tabs.addTab(self._original_panel, "原始信号")

        # 滤波结果谱图
        self._filtered_panel = FilteredSignalPanel()
        self._tabs.addTab(self._filtered_panel, "滤波结果")

        # 排名表
        rank_widget = QWidget()
        rank_layout = QVBoxLayout(rank_widget)
        self._rank_table = QTableView()
        self._rank_model = RankingTableModel()
        self._rank_table.setModel(self._rank_model)
        self._rank_table.setSortingEnabled(True)
        self._rank_table.horizontalHeader().setStretchLastSection(True)
        self._rank_table.setAlternatingRowColors(True)
        rank_layout.addWidget(self._rank_table)
        self._tabs.addTab(rank_widget, "排名")

        # 指标对比
        metric_widget = QWidget()
        metric_layout = QVBoxLayout(metric_widget)
        self._metric_table = QTableWidget()
        metric_layout.addWidget(self._metric_table)
        self._tabs.addTab(metric_widget, "指标")

        # 图表
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        self._chart_canvas = ChartCanvas(width=6, height=4)
        chart_layout.addWidget(self._chart_canvas)
        self._tabs.addTab(chart_widget, "图表")

        # 报告
        report_widget = QWidget()
        report_layout = QVBoxLayout(report_widget)
        self._report_browser = QTextBrowser()
        self._report_browser.setOpenExternalLinks(False)
        report_layout.addWidget(self._report_browser)
        self._tabs.addTab(report_widget, "报告")

        layout.addWidget(self._tabs)

        # 状态显示
        self._info_label = QLabel("暂无评估结果")
        self._info_label.setStyleSheet("color: #888; padding: 4px;")
        layout.addWidget(self._info_label)

    def load_report(self, report: EvaluationReport):
        """加载评估报告到各标签页"""
        self._report = report
        self._export_csv_btn.setEnabled(True)
        self._export_html_btn.setEnabled(True)

        # 更新排名表
        self._rank_model.set_rankings(report.rankings)
        self._rank_table.resizeColumnsToContents()

        # 更新指标表
        self._populate_metric_table(report)

        # 更新图表
        self._populate_charts(report)

        # 更新报告
        self._populate_report_text(report)

        # 更新信息
        top = report.top_algorithms
        info = f"共 {len(report.rankings)} 个算法 | TOP-1: {top[0].algorithm_name} ({top[0].overall_score:.4f})" if top else "无结果"
        self._info_label.setText(info)

    def _populate_metric_table(self, report: EvaluationReport):
        """填充指标对比表"""
        if not report.rankings:
            return

        # 收集所有指标名
        all_metrics: set = set()
        for rank in report.rankings:
            all_metrics.update(rank.metric_scores.keys())
        metric_names = sorted(all_metrics)

        headers = ["算法名称"] + metric_names + ["综合得分"]
        self._metric_table.setColumnCount(len(headers))
        self._metric_table.setHorizontalHeaderLabels(headers)
        self._metric_table.setRowCount(len(report.rankings))

        for row, rank in enumerate(report.rankings):
            self._metric_table.setItem(row, 0, QTableWidgetItem(rank.algorithm_name))
            for col, mname in enumerate(metric_names):
                val = rank.metric_scores.get(mname, 0)
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._metric_table.setItem(row, col + 1, item)
            self._metric_table.setItem(
                row, len(metric_names) + 1,
                QTableWidgetItem(f"{rank.overall_score:.4f}")
            )

        self._metric_table.resizeColumnsToContents()

    def _populate_charts(self, report: EvaluationReport):
        """填充图表"""
        if not report.rankings:
            return

        # 排名柱状图
        names = [r.algorithm_name for r in report.rankings]
        scores = [r.overall_score for r in report.rankings]
        self._chart_canvas.draw_ranking_bar(names, scores)

    def _populate_report_text(self, report: EvaluationReport):
        """填充报告文本"""
        lines = []
        lines.append(f"# 滤波降噪评估报告\n")
        lines.append(f"**项目**: {report.project_name}\n")
        lines.append(f"**信号**: {report.signal_info.get('path', 'N/A')}")
        lines.append(f"**采样率**: {report.signal_info.get('sample_rate', 'N/A')} Hz")
        lines.append(f"**通道数**: {report.signal_info.get('channels', 'N/A')}")
        lines.append(f"**时长**: {report.signal_info.get('duration', 'N/A'):.2f} s\n")
        lines.append("---\n")
        lines.append("## 排名结果\n")
        lines.append("| 排名 | 算法 | 类别 | 综合得分 | 执行时间 |")
        lines.append("|------|------|------|----------|----------|")
        for r in report.rankings:
            cat = r.category.value if hasattr(r.category, 'value') else str(r.category)
            lines.append(f"| {r.overall_rank} | {r.algorithm_name} | {cat} | {r.overall_score:.4f} | {r.execution_time:.3f}s |")
        lines.append("\n---\n")
        lines.append("*报告由滤波基石系统自动生成*")
        self._report_browser.setMarkdown("\n".join(lines))

    def _export_csv(self):
        if not self._report:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 CSV", "evaluation_report.csv", "CSV (*.csv)"
        )
        if path:
            try:
                import csv
                with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.writer(f)
                    writer.writerow(["排名", "算法ID", "算法名称", "类别", "综合得分", "执行时间(s)"])
                    for r in self._report.rankings:
                        cat = r.category.value if hasattr(r.category, 'value') else str(r.category)
                        writer.writerow([r.overall_rank, r.algorithm_id, r.algorithm_name,
                                        cat, round(r.overall_score, 4), round(r.execution_time, 3)])
                self._info_label.setText(f"已导出: {path}")
            except Exception as e:
                self._info_label.setText(f"导出失败: {e}")

    def _export_html(self):
        if not self._report:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "导出 HTML", "evaluation_report.html", "HTML (*.html)"
        )
        if path:
            try:
                html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>",
                        "<title>滤波降噪评估报告</title>",
                        "<style>body{font-family:sans-serif;margin:20px}",
                        "table{border-collapse:collapse;width:100%}",
                        "th,td{border:1px solid #ddd;padding:8px;text-align:center}",
                        "th{background:#2c3e50;color:white}",
                        "tr:nth-child(even){background:#f2f2f2}</style></head><body>",
                        f"<h1>滤波降噪评估报告</h1>",
                        f"<p><b>项目:</b> {self._report.project_name}</p>",
                        f"<p><b>信号:</b> {self._report.signal_info.get('path', 'N/A')}</p>",
                        "<table><tr><th>排名</th><th>算法</th><th>类别</th><th>得分</th><th>时间</th></tr>"]
                for r in self._report.rankings:
                    cat = r.category.value if hasattr(r.category, 'value') else str(r.category)
                    html.append(f"<tr><td>{r.overall_rank}</td><td>{r.algorithm_name}</td>"
                                f"<td>{cat}</td><td>{r.overall_score:.4f}</td>"
                                f"<td>{r.execution_time:.3f}s</td></tr>")
                html.append("</table></body></html>")
                with open(path, 'w', encoding='utf-8') as f:
                    f.write("\n".join(html))
                self._info_label.setText(f"已导出: {path}")
            except Exception as e:
                self._info_label.setText(f"导出失败: {e}")

    def update_signal_preview(self, data: np.ndarray, sample_rate: float,
                                source_path: str = ""):
        """更新原始信号谱图标签页"""
        self._original_panel.update_signal(data, sample_rate, source_path)
        # 同时清空滤波结果（信号变更后之前的评估结果不再有效）
        self._filtered_panel.clear()

    def update_filtered_signal(self, denoised_results: Dict,
                                sample_rate: float):
        """更新滤波结果标签页（评估完成后调用）"""
        self._filtered_panel.load_results(denoised_results, sample_rate)
        # 切换到滤波结果标签页
        self._tabs.setCurrentWidget(self._filtered_panel)

    def clear(self):
        """清空结果（保留原始信号谱图不变）"""
        self._report = None
        self._rank_model.set_rankings([])
        self._metric_table.setRowCount(0)
        self._metric_table.setColumnCount(0)
        self._chart_canvas.clear()
        self._chart_canvas.draw()
        # 不清空 _original_panel——原始信号图应在整个会话期间保持可见
        self._filtered_panel.clear()
        self._report_browser.clear()
        self._info_label.setText("暂无评估结果")
        self._export_csv_btn.setEnabled(False)
        self._export_html_btn.setEnabled(False)
