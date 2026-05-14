import importlib
import os as _os
from typing import Dict, List, Optional

from loguru import logger

from ..core.exceptions import AlgorithmNotFoundError
from ..core.types import AlgorithmCategory, AlgorithmComplexity, ComplexityAnalysis
from .base import BaseAlgorithm

from .complexity_analyzer import (
    get_complexity_analysis,
    get_high_complexity_ids,
    get_optimization_tips,
    get_very_high_resource_ids,
)


class AlgorithmMeta:
    def __init__(self, algorithm_id, algorithm_class):
        self.algorithm_id = algorithm_id
        self.algorithm_class = algorithm_class
        self.name = algorithm_class.name
        self.category = algorithm_class.category
        self.complexity = algorithm_class.complexity
        self.tags = algorithm_class.tags
        self.requires_gpu = algorithm_class.requires_gpu
        self.default_params = algorithm_class.default_params
        self._complexity_analysis: Optional[ComplexityAnalysis] = None

    @property
    def complexity_analysis(self) -> Optional[ComplexityAnalysis]:
        """获取算法的详细复杂度分析（延迟加载）。"""
        if self._complexity_analysis is None:
            self._complexity_analysis = get_complexity_analysis(self.algorithm_id)
            if self._complexity_analysis is not None:
                self._complexity_analysis.algorithm_name = self.name
                self._complexity_analysis.category = self.category
        return self._complexity_analysis

    def get_optimization_tips(self) -> str:
        """获取该算法的优化建议。"""
        analysis = self.complexity_analysis
        if analysis and analysis.optimization_tips:
            return analysis.optimization_tips
        return get_optimization_tips(
            self.category.value if hasattr(self.category, 'value') else str(self.category)
        )


class AlgorithmRegistry:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._registry = {}
        self._discovered = False

    def register(self, algorithm_id, algorithm_class):
        if algorithm_id in self._registry:
            return
        self._registry[algorithm_id] = AlgorithmMeta(algorithm_id, algorithm_class)

    def get(self, algorithm_id):
        self.ensure_discovered()
        if algorithm_id not in self._registry:
            raise KeyError(algorithm_id)
        return self._registry[algorithm_id]

    def create_instance(self, algorithm_id, **params):
        meta = self.get(algorithm_id)
        return meta.algorithm_class(**params)

    def list_ids(self):
        self.ensure_discovered()
        return list(self._registry.keys())

    def list_algorithms(self):
        self.ensure_discovered()
        return list(self._registry.values())

    def filter(self, category=None, complexity=None, tags=None, requires_gpu=None):
        results = self.list_algorithms()
        if category:
            if isinstance(category, list):
                cat_values = []
                from ..core.types import AlgorithmCategory
                for c in category:
                    if isinstance(c, str):
                        # Try enum name match, then value match
                        found = False
                        for e in AlgorithmCategory:
                            if e.name.lower() == c.lower() or e.value == c:
                                cat_values.append(e)
                                found = True
                                break
                        if not found:
                            cat_values.append(c)
                    else:
                        cat_values.append(c)
                results = [m for m in results if m.category in cat_values]
            else:
                results = [m for m in results if m.category == category]
        if complexity:
            results = [m for m in results if m.complexity == complexity]
        if tags:
            results = [m for m in results if any(t in m.tags for t in tags)]
        if requires_gpu is not None:
            results = [m for m in results if m.requires_gpu == requires_gpu]
        return results

    def count(self):
        return len(self._registry)

    def get_selected_complexity_warnings(self, algorithm_ids: List[str]) -> List[str]:
        """检查选中的算法中是否有高复杂度的，返回警告信息列表。

        Args:
            algorithm_ids: 选中的算法ID列表

        Returns:
            警告信息列表，每条包含算法名、资源消耗等级和优化建议
        """
        warnings: List[str] = []
        high_ids = get_high_complexity_ids()
        for aid in algorithm_ids:
            if aid in high_ids:
                meta = self._registry.get(aid)
                if meta:
                    analysis = meta.complexity_analysis
                    estimated = analysis.estimated_time_per_10k if analysis else "较长"
                    warnings.append(
                        f"• {meta.name}\n"
                        f"  资源消耗: {analysis.resource_level.value if analysis else '高'}\n"
                        f"  预计耗时: {estimated}\n"
                        f"  瓶颈: {analysis.bottlenecks if analysis else '计算量大'}"
                    )
        return warnings

    def ensure_discovered(self):
        if self._discovered:
            return
        self._discover_algorithms()
        self._discovered = True

    def _discover_algorithms(self):
        current_dir = _os.path.dirname(_os.path.abspath(__file__))
        modules = []
        for item in sorted(_os.listdir(current_dir)):
            item_path = _os.path.join(current_dir, item)
            if item.endswith(".py") and not item.startswith("_") and not item.startswith("."):
                modules.append(item[:-3])
            elif _os.path.isdir(item_path) and not item.startswith("_") and not item.startswith("."):
                init = _os.path.join(item_path, "__init__.py")
                if _os.path.exists(init):
                    for sub in sorted(_os.listdir(item_path)):
                        if sub.endswith(".py") and not sub.startswith("_") and not sub.startswith("."):
                            modules.append(item + "." + sub[:-3])
        prefixes = ["src.algorithms", "algorithms"]
        for mod_name in modules:
            imported = False
            for prefix in prefixes:
                full = prefix + "." + mod_name
                try:
                    importlib.import_module(full)
                    imported = True
                    break
                except ModuleNotFoundError:
                    continue
                except Exception:
                    continue
            if not imported:
                try:
                    importlib.import_module(mod_name)
                except Exception:
                    pass


registry = AlgorithmRegistry()