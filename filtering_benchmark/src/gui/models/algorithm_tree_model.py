"""
算法树模型 - QAbstractItemModel。

两层树结构:
  第一层: 算法类别 (AlgorithmCategory)
  第二层: 具体算法 (带复选框)
"""

from typing import Dict, List, Optional, Set

from PyQt6.QtCore import QAbstractItemModel, QModelIndex, QSortFilterProxyModel, Qt

from ...algorithms.registry import AlgorithmMeta


class AlgorithmTreeNode:
    """算法树节点"""
    def __init__(self, name: str, node_type: str = "category",
                 meta: Optional[AlgorithmMeta] = None,
                 parent: Optional['AlgorithmTreeNode'] = None):
        self.name = name
        self.node_type = node_type  # "category" or "algorithm"
        self.meta = meta
        self.parent_node = parent
        self.children: List['AlgorithmTreeNode'] = []
        self.checked = False

    def append_child(self, child: 'AlgorithmTreeNode'):
        self.children.append(child)

    def child(self, row: int) -> Optional['AlgorithmTreeNode']:
        if row < len(self.children):
            return self.children[row]
        return None

    def child_count(self) -> int:
        return len(self.children)

    def row(self) -> int:
        if self.parent_node:
            return self.parent_node.children.index(self)
        return 0

    def column_count(self) -> int:
        return 4  # name, complexity, tags, resource

    def data(self, column: int) -> str:
        if column == 0:
            return self.name
        if column == 1 and self.meta:
            return self.meta.complexity.value if hasattr(self.meta.complexity, 'value') else str(self.meta.complexity)
        if column == 2 and self.meta:
            return ", ".join(self.meta.tags) if self.meta.tags else ""
        if column == 3 and self.meta:
            analysis = self.meta.complexity_analysis
            if analysis:
                return analysis.resource_level.value
            return ""
        return ""


class AlgorithmTreeModel(QAbstractItemModel):
    """算法树模型"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.root_node = AlgorithmTreeNode("root", "root")
        self._algorithm_map: Dict[str, AlgorithmTreeNode] = {}
        self._build_tree()

    def _build_tree(self):
        """从 registry 构建算法树"""
        from ...algorithms import registry as algo_registry
        algo_registry.ensure_discovered()
        algorithms = algo_registry.list_algorithms()

        # 按类别分组
        categories: Dict[str, List[AlgorithmMeta]] = {}
        for meta in algorithms:
            cat_name = meta.category.value if hasattr(meta.category, 'value') else str(meta.category)
            if cat_name not in categories:
                categories[cat_name] = []
            categories[cat_name].append(meta)

        # 构建树
        from ...core.types import AlgorithmCategory
        category_order = [e.value for e in AlgorithmCategory]
        for cat_value in category_order:
            if cat_value not in categories:
                continue
            cat_node = AlgorithmTreeNode(cat_value, "category", parent=self.root_node)
            self.root_node.append_child(cat_node)
            metas = sorted(categories[cat_value], key=lambda m: m.name)
            for meta in metas:
                algo_node = AlgorithmTreeNode(
                    meta.name, "algorithm", meta, parent=cat_node
                )
                cat_node.append_child(algo_node)
                self._algorithm_map[meta.algorithm_id] = algo_node

    def algorithm_ids_by_category(self, cat_name: str) -> List[str]:
        """获取某类别下的所有算法 ID"""
        ids = []
        for i in range(self.root_node.child_count()):
            cat = self.root_node.child(i)
            if cat and cat.name == cat_name:
                for j in range(cat.child_count()):
                    algo = cat.child(j)
                    if algo and algo.meta:
                        ids.append(algo.meta.algorithm_id)
        return ids

    def get_selected_ids(self) -> Set[str]:
        """获取所有选中的算法 ID"""
        selected: Set[str] = set()
        for i in range(self.root_node.child_count()):
            cat = self.root_node.child(i)
            if cat:
                for j in range(cat.child_count()):
                    algo = cat.child(j)
                    if algo and algo.checked and algo.meta:
                        selected.add(algo.meta.algorithm_id)
        return selected

    def get_all_ids(self) -> List[str]:
        """获取所有算法 ID"""
        ids: List[str] = []
        for i in range(self.root_node.child_count()):
            cat = self.root_node.child(i)
            if cat:
                for j in range(cat.child_count()):
                    algo = cat.child(j)
                    if algo and algo.meta:
                        ids.append(algo.meta.algorithm_id)
        return ids

    def set_all_checked(self, checked: bool):
        """全选/全不选"""
        for i in range(self.root_node.child_count()):
            cat = self.root_node.child(i)
            if cat:
                for j in range(cat.child_count()):
                    algo = cat.child(j)
                    if algo:
                        algo.checked = checked
                        idx = self.createIndex(j, 0, algo)
                        self.dataChanged.emit(idx, idx)

    # ---- QAbstractItemModel interface ----

    def index(self, row: int, column: int, parent: QModelIndex = QModelIndex()) -> QModelIndex:
        if not self.hasIndex(row, column, parent):
            return QModelIndex()
        parent_node = parent.internalPointer() if parent.isValid() else self.root_node
        child_node = parent_node.child(row)
        if child_node:
            return self.createIndex(row, column, child_node)
        return QModelIndex()

    def parent(self, index: QModelIndex) -> QModelIndex:
        if not index.isValid():
            return QModelIndex()
        node = index.internalPointer()
        if not node or node == self.root_node:
            return QModelIndex()
        parent_node = node.parent_node
        if parent_node == self.root_node or parent_node is None:
            return QModelIndex()
        return self.createIndex(parent_node.row(), 0, parent_node)

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        node = parent.internalPointer() if parent.isValid() else self.root_node
        return node.child_count()

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 3

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        node = index.internalPointer()
        if role == Qt.ItemDataRole.ToolTipRole and index.column() == 0:
            if node.node_type == "algorithm" and node.meta:
                analysis = node.meta.complexity_analysis
                if analysis:
                    lines = [
                        f"算法: {node.name}",
                        f"复杂度等级: {analysis.complexity.value}",
                        f"资源消耗: {analysis.resource_level.value}",
                        f"时间复杂度: {analysis.time_complexity}",
                        f"空间复杂度: {analysis.space_complexity}",
                        f"CPU使用: {analysis.cpu_usage}  |  内存: {analysis.memory_usage}",
                        f"预估耗时(10k点): {analysis.estimated_time_per_10k}",
                        f"主要瓶颈: {analysis.bottlenecks}",
                    ]
                    return "\n".join(lines)
                return f"{node.name} (复杂度: {node.data(1)})"
            if node.node_type == "category":
                return node.name
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return node.data(index.column())
        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            if node.node_type == "algorithm":
                return Qt.CheckState.Checked if node.checked else Qt.CheckState.Unchecked
            return None
        if role == Qt.ItemDataRole.UserRole:
            return node
        return None

    def setData(self, index: QModelIndex, value, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            node = index.internalPointer()
            if node and node.node_type == "algorithm":
                node.checked = (value == Qt.CheckState.Checked)
                self.dataChanged.emit(index, index)
                return True
        return False

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        flags = super().flags(index)
        if index.isValid():
            node = index.internalPointer()
            if node and node.node_type == "algorithm":
                flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            headers = ["算法名称", "复杂度", "标签", "资源消耗"]
            return headers[section] if section < len(headers) else ""
        return None


class AlgorithmFilterProxyModel(QSortFilterProxyModel):
    """算法过滤代理模型"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._filter_text = ""

    def set_filter_text(self, text: str):
        self._filter_text = text
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        if not self._filter_text:
            return True
        source_model = self.sourceModel()
        if not source_model:
            return True
        idx = source_model.index(source_row, 0, source_parent)
        node = idx.internalPointer()
        if not node:
            return False
        if node.node_type == "category":
            # 类别节点：如果子节点有匹配则显示
            for i in range(node.child_count()):
                child = node.child(i)
                if child and self._filter_text.lower() in child.name.lower():
                    return True
                if child and child.meta:
                    for tag in child.meta.tags:
                        if self._filter_text.lower() in tag.lower():
                            return True
            return False
        # 算法节点
        return (self._filter_text.lower() in node.name.lower()
                or any(self._filter_text.lower() in t.lower() for t in (node.meta.tags if node.meta else [])))
