"""Script to write registry.py (uses Python open() to bypass Write tool path encoding issue)"""
import os

content = r'''"""
算法注册器（单例模式）。

AlgorithmRegistry 维护全局唯一的算法注册表。
支持：
- 通过装饰器声明式注册
- 批量自动发现（扫描 algorithms 子包）
- 按分类/标签/复杂度查询
"""

import importlib
import os as _os
from typing import Any, Dict, List, Optional, Type

from loguru import logger

from ..core.exceptions import AlgorithmNotFoundError
from ..core.types import AlgorithmCategory, AlgorithmComplexity
from .base import BaseAlgorithm


class AlgorithmMeta:
    """算法的注册元信息"""
    def __init__(self, algorithm_id: str, algorithm_class: Type[BaseAlgorithm]):
        self.algorithm_id = algorithm_id
        self.algorithm_class = algorithm_class
        self.name = algorithm_class.name
        self.category = algorithm_class.category
        self.complexity = algorithm_class.complexity
        self.tags = algorithm_class.tags
        self.requires_gpu = algorithm_class.requires_gpu
        self.default_params = algorithm_class.default_params


class AlgorithmRegistry:
    """全局算法注册表单例"""

    _instance: Optional["AlgorithmRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "AlgorithmRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._registry: Dict[str, AlgorithmMeta] = {}
        self._discovered = False

    def register(self, algorithm_id: str, algorithm_class: Type[BaseAlgorithm]) -> None:
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise TypeError(f"{algorithm_class} must inherit BaseAlgorithm")
        if algorithm_id in self._registry:
            logger.warning(f"Algorithm '{algorithm_id}' already registered, will be overwritten")
        self._registry[algorithm_id] = AlgorithmMeta(algorithm_id, algorithm_class)
        logger.debug(f"Registered algorithm: [{algorithm_class.category.value}] {algorithm_class.name} ({algorithm_id})")

    def get(self, algorithm_id: str) -> AlgorithmMeta:
        self.ensure_discovered()
        if algorithm_id not in self._registry:
            raise AlgorithmNotFoundError(algorithm_id)
        return self._registry[algorithm_id]

    def create_instance(self, algorithm_id: str, **params) -> BaseAlgorithm:
        meta = self.get(algorithm_id)
        return meta.algorithm_class(**params)

    def list_ids(self) -> List[str]:
        self.ensure_discovered()
        return list(self._registry.keys())

    def list_algorithms(self) -> List[AlgorithmMeta]:
        self.ensure_discovered()
        return list(self._registry.values())

    def filter(self, category=None, complexity=None, tags=None, requires_gpu=None):
        results = self.list_algorithms()
        if category:
            results = [m for m in results if m.category == category]
        if complexity:
            results = [m for m in results if m.complexity == complexity]
        if tags:
            results = [m for m in results if any(t in m.tags for t in tags)]
        if requires_gpu is not None:
            results = [m for m in results if m.requires_gpu == requires_gpu]
        return results

    def count(self) -> int:
        return len(self._registry)

    def ensure_discovered(self) -> None:
        if self._discovered:
            return
        self._discover_algorithms()
        self._discovered = True
        logger.info("Algorithm auto-discovery complete, " + str(self.count()) + " algorithms")

    def _discover_algorithms(self) -> None:
        """Discover and import all algorithm modules via filesystem scan"""
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
                except Exception as e:
                    logger.debug("Import " + full + ": " + str(e))
                    continue
            if not imported:
                try:
                    importlib.import_module(mod_name)
                except Exception:
                    pass


# Global singleton
registry = AlgorithmRegistry()
'''

target = r"e:\Qoder项目\滤波基石系统设计\filtering_benchmark\src\algorithms\registry.py"
with open(target, 'w', encoding='utf-8') as f:
    f.write(content)
print("Written to", target)
print("Size:", os.path.getsize(target))
