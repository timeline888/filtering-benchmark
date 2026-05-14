"""
算法选择器组件。

使用 QTreeView + AlgorithmTreeModel 展示按类别分组的算法列表，
支持搜索过滤、全选/取消全选、复杂度警告。
"""

from typing import List, Optional, Set

from PyQt6.QtCore import QModelIndex, Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSizePolicy, QTreeView, QVBoxLayout, QWidget,
)

from ...algorithms.registry import registry as algo_registry
from ...core.types import AlgorithmComplexity, ResourceConsumption
from ..models.algorithm_tree_model import AlgorithmFilterProxyModel, AlgorithmTreeModel


class _AlgorithmTreeView(QTreeView):
    """自定义 QTreeView，避免内置 checkbox 处理干扰手动 toggle"""

    def mouseReleaseEvent(self, event):
        index = self.indexAt(event.pos())
        if index.isValid() and index.column() == 0:
            model = self.model()
            # 通过代理模型映射到源模型获取节点信息
            source_idx = model.mapToSource(index) if hasattr(model, 'mapToSource') else index
            node = source_idx.internalPointer()
            if node and getattr(node, 'node_type', None) == "algorithm":
                # 手动 toggle，不调用 super()，阻止 QTreeView 内置 checkbox 处理
                current = node.checked
                new_state = Qt.CheckState.Unchecked if current else Qt.CheckState.Checked
                model.setData(index, new_state, Qt.ItemDataRole.CheckStateRole)
                return
        super().mouseReleaseEvent(event)


class AlgorithmSelectorWidget(QWidget):
    """算法选择器面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress_warning = False  # 批量操作时抑制警告
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 搜索框
        search_layout = QHBoxLayout()
        self._search_edit = QLineEdit()
        self._search_edit.setPlaceholderText("搜索算法名称或标签...")
        self._search_edit.textChanged.connect(self._on_search)
        search_layout.addWidget(self._search_edit)
        layout.addLayout(search_layout)

        # 算法树
        self._tree_model = AlgorithmTreeModel()
        self._proxy_model = AlgorithmFilterProxyModel()
        self._proxy_model.setSourceModel(self._tree_model)

        self._tree_view = _AlgorithmTreeView()
        self._tree_view.setModel(self._proxy_model)
        self._tree_view.setAnimated(True)
        self._tree_view.setIndentation(20)
        self._tree_view.setHeaderHidden(False)
        self._tree_view.setSortingEnabled(False)
        self._tree_view.setSelectionMode(QTreeView.SelectionMode.SingleSelection)
        self._tree_view.expandAll()
        layout.addWidget(self._tree_view)

        # 操作按钮
        btn_layout = QHBoxLayout()

        self._select_all_btn = QPushButton("全选")
        self._select_all_btn.clicked.connect(self._on_select_all)
        btn_layout.addWidget(self._select_all_btn)

        self._deselect_all_btn = QPushButton("取消全选")
        self._deselect_all_btn.clicked.connect(self._on_deselect_all)
        btn_layout.addWidget(self._deselect_all_btn)

        btn_layout.addStretch()

        self._count_label = QLabel("已选: 0")
        btn_layout.addWidget(self._count_label)

        layout.addLayout(btn_layout)

        # 复杂度分析按钮
        analysis_layout = QHBoxLayout()
        self._analysis_btn = QPushButton("📊 复杂度分析")
        self._analysis_btn.setToolTip("查看已选算法的计算资源消耗详情和优化建议")
        self._analysis_btn.clicked.connect(self._show_complexity_analysis)
        analysis_layout.addWidget(self._analysis_btn)
        analysis_layout.addStretch()
        layout.addLayout(analysis_layout)

        # 监听数据变化：更新计数 + 检查复杂度警告
        self._tree_model.dataChanged.connect(self._update_count)
        self._tree_model.dataChanged.connect(self._on_data_changed)
        self._tree_model.modelReset.connect(self._update_count)
        self._update_count()

    def _on_select_all(self):
        """全选所有算法（批量操作，抑制单个警告）"""
        self._suppress_warning = True
        self._tree_model.set_all_checked(True)
        self._suppress_warning = False

    def _on_deselect_all(self):
        """取消全选（批量操作，抑制警告）"""
        self._suppress_warning = True
        self._tree_model.set_all_checked(False)
        self._suppress_warning = False

    def _on_search(self, text: str):
        self._proxy_model.set_filter_text(text)
        self._tree_view.expandAll()

    def _update_count(self):
        total = len(self._tree_model.get_all_ids())
        selected = len(self._tree_model.get_selected_ids())
        self._count_label.setText(f"已选: {selected}/{total}")

    def _on_data_changed(self, top_left: QModelIndex, bottom_right: QModelIndex, roles=None):
        """当树模型数据变化时，检查新选中的高复杂度算法并弹出警告。"""
        if self._suppress_warning:
            return  # 批量操作时不弹窗
        node = top_left.internalPointer()
        if not node or node.node_type != "algorithm":
            return
        if not node.checked:
            return  # 取消选中不弹警告

        meta = node.meta
        if not meta:
            return

        analysis = meta.complexity_analysis
        if not analysis:
            return

        # 高复杂度或高资源消耗才弹警告
        is_high_complexity = analysis.complexity == AlgorithmComplexity.HIGH
        is_high_resource = analysis.resource_level in (
            ResourceConsumption.HIGH, ResourceConsumption.VERY_HIGH
        )
        if not is_high_complexity and not is_high_resource:
            return

        self._show_algorithm_warning(analysis)

    def _show_algorithm_warning(self, analysis):
        """显示单个算法复杂度警告对话框。"""
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("算法复杂度警告")
        msg.setText(
            f"⚠️ 算法「{analysis.algorithm_name}」计算资源消耗较大，"
            f"可能导致运行卡顿！"
        )
        detail_lines = [
            f"复杂度等级: {analysis.complexity.value}",
            f"资源消耗: {analysis.resource_level.value}",
            f"时间复杂度: {analysis.time_complexity}",
            f"空间复杂度: {analysis.space_complexity}",
            f"CPU使用: {analysis.cpu_usage}  |  内存: {analysis.memory_usage}",
            f"预估耗时(10k点): {analysis.estimated_time_per_10k}",
            f"主要瓶颈: {analysis.bottlenecks}",
        ]
        msg.setInformativeText("\n".join(detail_lines))
        msg.setDetailedText(f"优化建议:\n{analysis.optimization_tips}")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _show_complexity_analysis(self):
        """显示已选算法的复杂度分析汇总对话框。"""
        selected_ids = self._tree_model.get_selected_ids()
        if not selected_ids:
            QMessageBox.information(self, "复杂度分析", "当前未选择任何算法，请先选择算法。")
            return

        warnings = algo_registry.get_selected_complexity_warnings(list(selected_ids))
        all_count = len(selected_ids)
        high_count = len(warnings)

        # 构建汇总文本
        lines = [f"已选算法总数: {all_count}", f"高复杂度算法数: {high_count}", ""]

        if warnings:
            lines.append("⚠️ 高复杂度算法详情:")
            lines.append("─" * 40)
            for i, w in enumerate(warnings, 1):
                lines.append(f"\n[{i}] {w}")
                lines.append("─" * 40)

        # 统计各资源等级的数量
        resource_counts: dict = {}
        for aid in selected_ids:
            try:
                meta = algo_registry.get(aid)
                analysis = meta.complexity_analysis
                if analysis:
                    rl = analysis.resource_level.value
                    resource_counts[rl] = resource_counts.get(rl, 0) + 1
            except KeyError:
                pass

        if resource_counts:
            lines.append("\n📊 资源消耗分布:")
            for level in ["极低", "低", "中", "高", "极高"]:
                cnt = resource_counts.get(level, 0)
                if cnt:
                    bar = "█" * cnt
                    lines.append(f"  {level}: {bar} ({cnt})")

        msg = QMessageBox(self)
        msg.setIcon(
            QMessageBox.Icon.Warning if high_count > 0 else QMessageBox.Icon.Information
        )
        msg.setWindowTitle("算法复杂度分析")
        msg.setText(
            f"已选 {all_count} 个算法，其中 {high_count} 个高复杂度算法。"
        )
        msg.setInformativeText("查看详情了解具体每个算法的资源消耗。")
        msg.setDetailedText("\n".join(lines))
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def get_selected_ids(self) -> Set[str]:
        return self._tree_model.get_selected_ids()

    def get_all_ids(self):
        return self._tree_model.get_all_ids()

    def expand_all(self):
        """展开所有节点"""
        self._tree_view.expandAll()

    def collapse_all(self):
        """折叠所有节点"""
        self._tree_view.collapseAll()

    def model(self):
        return self._tree_model
